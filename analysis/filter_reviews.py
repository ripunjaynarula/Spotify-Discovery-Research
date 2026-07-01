from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from analysis.llm_client import AnalysisError, generate_json_content, parse_json_response
from analysis.schema import clamp_confidence
from reviews.models import RAW_REVIEW_FIELDS


from config import (
    RAW_REVIEWS_CSV,
    FILTERED_REVIEWS_CSV,
    REJECTED_REVIEWS_CSV,
    FILTER_SUMMARY_JSON,
    OPENROUTER_MODEL,
    DEFAULT_BATCH_SIZE_FILTER,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
    DEFAULT_MIN_REVIEW_LENGTH,
    DEFAULT_MAX_TOKENS_RELEVANCE,
    DEFAULT_MAX_REVIEW_CHARACTERS,
)

DEFAULT_INPUT_PATH = RAW_REVIEWS_CSV
DEFAULT_OUTPUT_PATH = FILTERED_REVIEWS_CSV
DEFAULT_REJECTED_OUTPUT_PATH = REJECTED_REVIEWS_CSV
DEFAULT_SUMMARY_OUTPUT_PATH = FILTER_SUMMARY_JSON
DEFAULT_MODEL = OPENROUTER_MODEL

RELEVANCE_FIELDS = ["relevant", "reason", "confidence"]

SYSTEM_PROMPT = """
You are helping a Product Manager research ONLY Spotify's music discovery experience.

A review is relevant ONLY if it discusses one or more of:
- discovering music
- finding new artists
- recommendations
- recommendation quality
- recommendation diversity
- recommendation personalization
- recommendation trust
- recommendation effort
- Discover Weekly
- Daily Mix
- AI DJ
- Smart Shuffle
- Spotify Radio
- playlist recommendations
- repetitive listening

Reject reviews primarily about the following topics, UNLESS they also discuss discovery:
- ads
- premium
- billing
- crashes
- playback
- offline
- downloads
- login
- account
- subscriptions
- widgets
- Android Auto
- connectivity
- pricing

Return JSON only.

{
  "reviews": [
    {
      "id": "...",
      "relevant": true,
      "reason": "...",
      "confidence": 0.95
    }
  ]
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter raw Spotify reviews for discovery/recommendation relevance."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE_FILTER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay-seconds", type=float, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--min-review-length", type=int, default=DEFAULT_MIN_REVIEW_LENGTH)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only keep relevant reviews at or above this confidence threshold.",
    )
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.min_review_length < 1:
        raise ValueError("--min-review-length must be at least 1")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1")

    raw_reviews = read_raw_reviews(args.input)
    reviews_for_llm, removal_counts = prepare_reviews_for_llm(
        raw_reviews,
        min_review_length=args.min_review_length,
    )
    print_removal_counts(removal_counts)

    relevance_rows = classify_relevance(reviews_for_llm, args)
    filtered_reviews = filter_relevant_reviews(
        reviews_for_llm,
        relevance_rows,
        min_confidence=args.min_confidence,
    )
    rejected_reviews = build_rejected_reviews(reviews_for_llm, relevance_rows)
    write_filtered_reviews(filtered_reviews, args.output)
    write_rejected_reviews(rejected_reviews, DEFAULT_REJECTED_OUTPUT_PATH)
    write_filter_summary(
        summary=build_filter_summary(
            total_reviews=len(raw_reviews),
            reviews_sent_to_llm=len(reviews_for_llm),
            filtered_reviews=filtered_reviews,
            relevance_rows=relevance_rows,
            model=args.model,
            processing_time_seconds=time.perf_counter() - start_time,
        ),
        output_path=DEFAULT_SUMMARY_OUTPUT_PATH,
    )
    print(f"Wrote {len(filtered_reviews)} relevant reviews to {args.output}")
    print(f"Wrote {len(rejected_reviews)} rejected reviews to {DEFAULT_REJECTED_OUTPUT_PATH}")
    print(f"Wrote filter summary to {DEFAULT_SUMMARY_OUTPUT_PATH}")


def read_raw_reviews(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Raw reviews file not found: {input_path}")
    dataframe = pd.read_csv(input_path, dtype={"id": str}).fillna("")
    missing_fields = [field for field in RAW_REVIEW_FIELDS if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"Missing required raw review fields: {missing_fields}")
    dataframe = dataframe[RAW_REVIEW_FIELDS].copy()
    dataframe["review"] = dataframe["review"].astype(str).str.strip()
    return dataframe


def prepare_reviews_for_llm(
    raw_reviews: pd.DataFrame,
    min_review_length: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    reviews_series = raw_reviews["review"].astype(str).str.strip()
    
    empty_mask = reviews_series == ""
    short_mask = reviews_series.str.len() < min_review_length
    
    valid_mask = (~empty_mask) & (~short_mask)
    valid_reviews = raw_reviews[valid_mask]
    
    duplicate_mask = valid_reviews["review"].astype(str).str.strip().duplicated(keep="first")
    prepared_reviews = valid_reviews[~duplicate_mask].copy()

    removal_counts = {
        "empty_reviews": int(empty_mask.sum()),
        "short_reviews": int((~empty_mask & short_mask).sum()),
        "duplicate_review_texts": int(duplicate_mask.sum()),
    }
    return prepared_reviews, removal_counts


def print_removal_counts(removal_counts: dict[str, int]) -> None:
    print(f"Removed empty reviews: {removal_counts['empty_reviews']}")
    print(f"Removed short reviews: {removal_counts['short_reviews']}")
    print(f"Removed duplicate review texts: {removal_counts['duplicate_review_texts']}")


def deterministic_pre_filter(review_text: str) -> dict[str, object] | None:
    text_lower = review_text.lower()
    discovery_keywords = [
        "discover",
        "recommend",
        "weekly",
        "mix",
        "dj",
        "shuffle",
        "radio",
        "finding new",
        "find new",
        "new music",
        "new artist",
        "repetitive",
        "repeat",
        "same song",
    ]
    has_discovery = any(kw in text_lower for kw in discovery_keywords)
    if not has_discovery:
        return {
            "relevant": False,
            "reason": "Deterministic pre-filter: no discovery keywords found.",
            "confidence": 1.0,
        }
    return None


def classify_relevance(
    raw_reviews: pd.DataFrame,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    records = raw_reviews.to_dict(orient="records")
    relevance_rows_by_id: dict[str, dict[str, object]] = {}
    
    llm_records: list[dict[str, object]] = []
    for record in records:
        review_id = str(record["id"])
        pre_filter_result = deterministic_pre_filter(record["review"])
        if pre_filter_result is not None:
            relevance_rows_by_id[review_id] = {
                "id": review_id,
                **pre_filter_result
            }
        else:
            llm_records.append(record)
            
    total_llm_reviews = len(llm_records)
    if total_llm_reviews > 0:
        progress = tqdm(
            range(0, total_llm_reviews, args.batch_size),
            desc="Filtering reviews via LLM",
            unit="batch",
        )
        for start in progress:
            batch = llm_records[start : start + args.batch_size]
            batch_results = classify_relevance_batch(
                reviews=_prompt_records(batch),
                model=args.model,
                max_retries=args.max_retries,
                retry_delay_seconds=args.retry_delay_seconds,
            )
            for res in batch_results:
                relevance_rows_by_id[str(res["id"])] = res
            progress.set_postfix(
                reviews=f"{min(start + args.batch_size, total_llm_reviews)}/{total_llm_reviews}"
            )
            
    return [relevance_rows_by_id[str(record["id"])] for record in records]


def classify_relevance_batch(
    reviews: list[dict[str, object]],
    model: str,
    max_retries: int,
    retry_delay_seconds: float,
) -> list[dict[str, object]]:
    remaining_reviews = reviews
    relevance_by_id: dict[str, dict[str, object]] = {}
    attempts = max(1, max_retries + 1)

    for attempt in range(attempts):
        batch_rows = _request_relevance_batch(
            reviews=remaining_reviews,
            model=model,
            max_retries=0,
            retry_delay_seconds=retry_delay_seconds,
        )
        for row in batch_rows:
            relevance_by_id[str(row["id"])] = row

        missing_reviews = _missing_reviews(remaining_reviews, relevance_by_id)
        if not missing_reviews:
            break

        missing_ids = [str(review["id"]) for review in missing_reviews]
        print(f"LLM returned no result for review ids: {', '.join(missing_ids)}")
        if attempt == attempts - 1:
            for review in missing_reviews:
                fallback_row = _no_model_response(str(review["id"]))
                relevance_by_id[str(review["id"])] = fallback_row
            break

        delay_seconds = retry_delay_seconds * (2**attempt)
        print(f"Retrying missing review ids only in {delay_seconds} seconds...")
        time.sleep(delay_seconds)
        remaining_reviews = missing_reviews

    return [relevance_by_id[str(review["id"])] for review in reviews]


def _request_relevance_batch(
    reviews: list[dict[str, object]],
    model: str,
    max_retries: int,
    retry_delay_seconds: float,
) -> list[dict[str, object]]:
    def request_and_parse() -> list[dict[str, object]]:
        content = generate_json_content(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_relevance_prompt(reviews),
            max_tokens=DEFAULT_MAX_TOKENS_RELEVANCE,
        )
        parsed = parse_json_response(content)
        return normalize_relevance_response(parsed, reviews)

    return _with_retries(request_and_parse, max_retries, retry_delay_seconds)


def build_relevance_prompt(reviews: list[dict[str, object]]) -> str:
    return (
        "Classify each review for assignment relevance. Return a JSON object with "
        'one key named "reviews". Its value must be an array with exactly one '
        "result per input. Each result must include the original id and these "
        f"fields: {RELEVANCE_FIELDS}.\n\n"
        f"Input reviews:\n{reviews}"
    )


def normalize_relevance_response(
    parsed: dict[str, Any],
    input_reviews: list[dict[str, object]],
) -> list[dict[str, object]]:
    items = parsed.get("reviews", parsed.get("results", parsed.get("items")))
    if not isinstance(items, list):
        raise AnalysisError('Gemini response must contain a "reviews" array.')

    by_id = {
        str(item.get("id")): item
        for item in items
        if isinstance(item, dict) and item.get("id") is not None
    }

    rows: list[dict[str, object]] = []
    for input_review in input_reviews:
        review_id = str(input_review["id"])
        item = by_id.get(review_id)
        if not isinstance(item, dict):
            continue
        rows.append({"id": review_id, **_normalize_relevance_item(item)})
    return rows


def filter_relevant_reviews(
    raw_reviews: pd.DataFrame,
    relevance_rows: list[dict[str, object]],
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    relevance_by_id = {str(row["id"]): row for row in relevance_rows}
    keep_ids = {
        review_id
        for review_id, row in relevance_by_id.items()
        if row["relevant"] is True and float(row["confidence"]) >= min_confidence
    }
    return raw_reviews[raw_reviews["id"].astype(str).isin(keep_ids)][RAW_REVIEW_FIELDS].copy()


def write_filtered_reviews(filtered_reviews: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_reviews.to_csv(
        output_path,
        columns=RAW_REVIEW_FIELDS,
        index=False,
        encoding="utf-8-sig",
    )


def build_rejected_reviews(
    raw_reviews: pd.DataFrame,
    relevance_rows: list[dict[str, object]],
) -> pd.DataFrame:
    raw_by_id = raw_reviews.set_index(raw_reviews["id"].astype(str))
    rows: list[dict[str, object]] = []
    for relevance_row in relevance_rows:
        if relevance_row["relevant"] is True:
            continue
        review_id = str(relevance_row["id"])
        if review_id not in raw_by_id.index:
            continue
        rows.append(
            {
                "id": review_id,
                "review": raw_by_id.loc[review_id, "review"],
                "reason": relevance_row["reason"],
                "confidence": relevance_row["confidence"],
            }
        )
    return pd.DataFrame(rows, columns=["id", "review", "reason", "confidence"])


def write_rejected_reviews(rejected_reviews: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_reviews.to_csv(
        output_path,
        columns=["id", "review", "reason", "confidence"],
        index=False,
        encoding="utf-8-sig",
    )


def build_filter_summary(
    total_reviews: int,
    reviews_sent_to_llm: int,
    filtered_reviews: pd.DataFrame,
    relevance_rows: list[dict[str, object]],
    model: str,
    processing_time_seconds: float,
) -> dict[str, object]:
    confidence_values = [float(row["confidence"]) for row in relevance_rows]
    irrelevant_reviews = sum(1 for row in relevance_rows if row["relevant"] is not True)
    return {
        "total_reviews": total_reviews,
        "reviews_sent_to_llm": reviews_sent_to_llm,
        "relevant_reviews": len(filtered_reviews),
        "irrelevant_reviews": irrelevant_reviews,
        "average_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        ),
        "model": model,
        "processing_time_seconds": round(processing_time_seconds, 2),
    }


def write_filter_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _prompt_records(batch: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": str(row["id"]),
            "review": _truncate_review_text(row.get("review"), DEFAULT_MAX_REVIEW_CHARACTERS),
        }
        for row in batch
    ]


def _truncate_review_text(review: object, max_characters: int) -> str:
    text = str(review or "").strip()
    if len(text) <= max_characters:
        return text
    return f"{text[: max_characters - 3].rstrip()}..."


def _normalize_relevance_item(item: dict[str, Any]) -> dict[str, object]:
    return {
        "relevant": _as_bool(item.get("relevant")),
        "reason": str(item.get("reason") or "").strip(),
        "confidence": clamp_confidence(item.get("confidence")),
    }


def _missing_reviews(
    input_reviews: list[dict[str, object]],
    relevance_by_id: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    return [
        review
        for review in input_reviews
        if str(review["id"]) not in relevance_by_id
    ]


def _no_model_response(review_id: str) -> dict[str, object]:
    return {
        "id": review_id,
        "relevant": False,
        "reason": "No model response",
        "confidence": 0.0,
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() == "true"
    return False


def _with_retries(
    operation: Callable[[], list[dict[str, object]]],
    max_retries: int,
    retry_delay_seconds: float,
) -> list[dict[str, object]]:
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            delay_seconds = retry_delay_seconds * (2**attempt)
            print(f"Attempt {attempt + 1}/{attempts} failed.")
            print(f"Reason: {exc}")
            print(f"Retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise AnalysisError("LLM relevance filtering failed without a captured exception.")


if __name__ == "__main__":
    main()

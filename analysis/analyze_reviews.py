from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from config import (
    FILTERED_REVIEWS_CSV,
    ANALYZED_REVIEWS_CSV,
    OPENROUTER_MODEL,
    DEFAULT_BATCH_SIZE_ANALYZE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
)

from dotenv import load_dotenv

from analysis.llm_client import AnalysisError, analyze_review_batch
from analysis.schema import ANALYSIS_FIELDS, empty_analysis
from reviews.models import RAW_REVIEW_FIELDS

load_dotenv()

DEFAULT_INPUT_PATH = FILTERED_REVIEWS_CSV
DEFAULT_OUTPUT_PATH = ANALYZED_REVIEWS_CSV
DEFAULT_MODEL = OPENROUTER_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze raw Spotify reviews with an LLM into structured fields."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE_ANALYZE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay-seconds", type=float, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write unknown/0-confidence rows for failed batches instead of stopping.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    raw_reviews = read_raw_reviews(args.input)
    analyzed_rows = analyze_reviews(raw_reviews, args)
    write_analyzed_reviews(analyzed_rows, args.output)
    print(f"Wrote {len(analyzed_rows)} analyzed reviews to {args.output}")


def read_raw_reviews(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Raw reviews file not found: {input_path}")
    dataframe = pd.read_csv(input_path, dtype={"id": str})
    missing_fields = [field for field in RAW_REVIEW_FIELDS if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"Missing required raw review fields: {missing_fields}")
    dataframe = dataframe[RAW_REVIEW_FIELDS].fillna("")
    dataframe["review"] = dataframe["review"].astype(str).str.strip()
    return dataframe[dataframe["review"] != ""].copy()


def analyze_reviews(
    raw_reviews: pd.DataFrame,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total_reviews = len(raw_reviews)
    records = raw_reviews.to_dict(orient="records")

    for start in range(0, total_reviews, args.batch_size):
        batch = records[start : start + args.batch_size]
        try:
            batch_analysis = analyze_review_batch(
                reviews=_prompt_records(batch),
                model=args.model,
                max_retries=args.max_retries,
                retry_delay_seconds=args.retry_delay_seconds,
            )
        except AnalysisError:
            if not args.continue_on_error:
                raise
            batch_analysis = _fallback_batch_analysis(batch)

        analysis_by_id = {item["id"]: item for item in batch_analysis}
        for raw_row in batch:
            analysis = analysis_by_id.get(str(raw_row["id"]), _fallback_analysis(raw_row))
            rows.append({**raw_row, **_without_id(analysis)})

        print(f"Analyzed {min(start + args.batch_size, total_reviews)}/{total_reviews}")

    return rows


def write_analyzed_reviews(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*RAW_REVIEW_FIELDS, *ANALYSIS_FIELDS]
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False, encoding="utf-8")


def _prompt_records(batch: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": str(row["id"]),
            "source": row["source"],
            "rating": row["rating"],
            "date": row["date"],
            "review": row["review"],
        }
        for row in batch
    ]


def _fallback_batch_analysis(batch: list[dict[str, object]]) -> list[dict[str, object]]:
    return [_fallback_analysis(row) for row in batch]


def _fallback_analysis(row: dict[str, object]) -> dict[str, object]:
    return {"id": str(row["id"]), **empty_analysis()}


def _without_id(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "id"}


if __name__ == "__main__":
    main()

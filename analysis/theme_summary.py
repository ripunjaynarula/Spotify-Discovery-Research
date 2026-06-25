from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from analysis.schema import ANALYSIS_FIELDS
from reviews.models import RAW_REVIEW_FIELDS


DEFAULT_INPUT_PATH = Path("data/analyzed_reviews.csv")
DEFAULT_OUTPUT_PATH = Path("output/theme_summary.md")

SUMMARY_SECTIONS = [
    ("Most common discovery problems", "pain_point"),
    ("Desired discovery experience", "desired_outcome"),
    ("Discovery surfaces", "discovery_surface"),
    ("Current behaviour", "current_behaviour"),
    ("Likely root causes", "root_cause"),
    ("User goals", "user_goal"),
    ("Primary user segments", "user_segment"),
    ("User emotions", "emotion"),
]

# Consolidated set of lowercase values to ignore
IGNORE_VALUES = {
    "",
    "unknown",
    "none",
    "n/a",
    "na",
    "not available",
    "not applicable",
    "not discovery related",
    "other",
    "misc",
    "no issue",
    "not relevant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown theme summary from analyzed Spotify reviews."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Exclude rows below this confidence threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1")

    analyzed_reviews = read_analyzed_reviews(args.input, args.min_confidence)
    markdown = build_theme_summary(analyzed_reviews, args.top_n)
    write_theme_summary(markdown, args.output)
    print(f"Wrote theme summary to {args.output}")


def read_analyzed_reviews(input_path: Path, min_confidence: float = 0.0) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Analyzed reviews file not found: {input_path}")

    dataframe = pd.read_csv(input_path, dtype={"id": str}).fillna("")
    required_fields = [*RAW_REVIEW_FIELDS, *ANALYSIS_FIELDS]
    missing_fields = [field for field in required_fields if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"Missing required analyzed review fields: {missing_fields}")

    dataframe["confidence"] = pd.to_numeric(dataframe["confidence"], errors="coerce").fillna(0)
    if min_confidence > 0:
        dataframe = dataframe[dataframe["confidence"] >= min_confidence].copy()
    return dataframe


def build_theme_summary(analyzed_reviews: pd.DataFrame, top_n: int) -> str:
    total_reviews = len(analyzed_reviews)
    lines = [
        "# Spotify Discovery Review Theme Summary",
        "",
        "## Method",
        "",
        (
            "This summary is generated only from `analyzed_reviews.csv` by counting "
            "the structured labels already extracted for each review. No new claims, "
            "interpretations, or external data are added."
        ),
        "",
        "## Dataset",
        "",
        f"- Reviews analyzed: {total_reviews}",
        f"- Average confidence: {_format_confidence(analyzed_reviews)}",
        "",
    ]

    for title, column in SUMMARY_SECTIONS:
        lines.extend(_section_lines(title, column, analyzed_reviews, top_n))

    return "\n".join(lines).rstrip() + "\n"


def write_theme_summary(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def clean_representative_review(review_text: str) -> str:
    """Cleans, formats, and truncates a representative review to a maximum of 120 characters, preserving complete words and escaping markdown."""
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', review_text)

    # 2. Remove repeated punctuation (e.g., ... or !!! or ??? to single characters)
    text = re.sub(r'([.,!?\-;:_])\1+', r'\1', text)

    # 3. Remove repeated whitespace (normalize whitespace)
    text = " ".join(text.split())

    # 4. Truncate to maximum of 120 characters preserving complete words
    if len(text) <= 120:
        cleaned_text = text
    else:
        truncated = text[:117]
        last_space = truncated.rfind(" ")
        if last_space != -1 and last_space > 50:
            cleaned_text = truncated[:last_space].strip() + "..."
        else:
            cleaned_text = truncated.strip() + "..."

    # 5. Escape markdown characters
    escaped_text = ""
    for char in cleaned_text:
        if char in ["|", "*", "_", "`", "[", "]", "(", ")", "#", "\\"]:
            escaped_text += "\\" + char
        else:
            escaped_text += char

    return escaped_text


def ranked_counts(dataframe: pd.DataFrame, column: str, top_n: int) -> list[tuple[str, int]]:
    values: list[str] = []
    for value in dataframe[column].tolist():
        normalized = _normalize_label(value)
        if normalized:
            values.append(normalized)
    counter = Counter(values)

    # Sorts descending by frequency and then ascending alphabetically by theme (case-insensitive)
    return sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0].lower())
    )[:top_n]


def _section_lines(
    title: str,
    column: str,
    dataframe: pd.DataFrame,
    top_n: int,
) -> list[str]:
    counts = ranked_counts(dataframe, column, top_n)
    lines = [f"## {title}", ""]
    if not counts:
        lines.extend(["No meaningful themes found after filtering.", ""])
        return lines

    lines.extend(["| Theme | Frequency | Share | Representative Review |", "|---|---:|---:|---|"])
    
    denominator = sum(
        1
        for value in dataframe[column]
        if _normalize_label(value)
    )

    for theme, frequency in counts:
        share = _format_share(frequency, denominator)
        
        # Get representative review directly from the dataset
        matching_rows = dataframe[
            dataframe[column].apply(_normalize_label) == theme
        ]
        
        if not matching_rows.empty:
            # Pick the review with the highest confidence
            best_row = matching_rows.sort_values(by="confidence", ascending=False).iloc[0]
            rep_review = clean_representative_review(str(best_row["review"]))
        else:
            rep_review = ""

        lines.append(f"| {_escape_markdown_table(theme)} | {frequency} | {share} | {rep_review} |")
        
    lines.append("")
    return lines


def _normalize_label(value: object) -> str:
    label = " ".join(str(value or "").split()).strip()

    if not label:
        return ""

    if label.lower() in IGNORE_VALUES:
        return ""

    return label


def _format_share(frequency: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(frequency / denominator) * 100:.1f}%"


def _format_confidence(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "0.00"
    return f"{dataframe['confidence'].mean():.2f}"


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from config import (
    RAW_REVIEWS_CSV,
    DEFAULT_LIMIT,
    GOOGLE_PLAY_COUNTRY,
    GOOGLE_PLAY_LANGUAGE,
)

from reviews.google_play import collect_google_play_reviews
from reviews.reddit import collect_reddit_reviews
from reviews.spotify_community import collect_spotify_community_reviews
from reviews.utils import write_raw_reviews_csv


DEFAULT_OUTPUT_PATH = RAW_REVIEWS_CSV
SUPPORTED_SOURCES = ("google_play", "reddit", "spotify_community")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Spotify discovery-related feedback into raw_reviews.csv."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SUPPORTED_SOURCES,
        default=["google_play"],
        help="Review sources to collect from.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max reviews per run.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path.",
    )
    parser.add_argument(
        "--spotify-community-url",
        default=None,
        help="Optional Spotify Community search URL.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    reviews = []

    if "google_play" in args.sources:
        reviews.extend(
            collect_google_play_reviews(
                limit=args.limit,
                country=GOOGLE_PLAY_COUNTRY,
                language=GOOGLE_PLAY_LANGUAGE,
            )
        )

    if "reddit" in args.sources:
        reviews.extend(collect_reddit_reviews(limit=args.limit))

    if "spotify_community" in args.sources:
        community_kwargs = {"limit": args.limit}
        if args.spotify_community_url:
            community_kwargs["search_url"] = args.spotify_community_url
        reviews.extend(collect_spotify_community_reviews(**community_kwargs))

    write_raw_reviews_csv(reviews, args.output)
    print(f"Wrote {len(reviews)} collected reviews to {args.output}")


if __name__ == "__main__":
    main()

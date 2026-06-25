from __future__ import annotations

import copy
import time
import urllib.parse
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup

from reviews.models import RawReview
from reviews.utils import clean_text, stable_id


DEFAULT_QUERIES = [
    "Discover Weekly",
    "Daily Mix",
    "AI DJ",
    "Smart Shuffle",
    "Spotify recommendations",
    "recommendation algorithm",
    "recommendation quality",
    "finding new music",
    "repetitive listening",
    "playlist recommendations",
    "recommended artists",
    "Spotify Radio",
]


def _extract_reddit_post(res: BeautifulSoup) -> str:
    """Extracts only the post title and self-text/snippet, discarding subreddit, author, flairs, flairs-class, etc."""
    element = copy.copy(res)

    # Decompose all metadata/interaction selectors
    unwanted_selectors = [
        ".search-author", ".search-subreddit", ".search-flair",
        ".search-comments", ".search-score", "time", ".search-time",
        ".search-info-header", ".search-result-meta", ".search-meta",
        ".flair", ".flair-rich", ".search-subreddit-link", ".author"
    ]
    for selector in unwanted_selectors:
        for tag in element.select(selector):
            tag.decompose()

    title_el = element.select_one("a.search-title")
    snippet_el = element.select_one(".search-result-text")

    title = ""
    snippet = ""

    if title_el:
        title = clean_text(title_el.get_text(" ", strip=True))
        title_el.decompose()

    if snippet_el:
        snippet = clean_text(snippet_el.get_text(" ", strip=True))
    else:
        snippet = clean_text(element.get_text(" ", strip=True))

    if title and snippet:
        if snippet.startswith(title):
            snippet = snippet[len(title):].strip()

    review_parts = []
    if title:
        review_parts.append(title)
    if snippet and snippet not in title:
        review_parts.append(snippet)

    return clean_text(" ".join(review_parts))


def collect_reddit_reviews(
    limit: int,
    subreddits: list[str] | None = None,
    queries: list[str] | None = None,
) -> list[RawReview]:
    """Collects Spotify discovery-related reviews from public Reddit search results.
    
    No API keys (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET) are required.
    """
    search_queries = queries or DEFAULT_QUERIES
    raw_reviews: list[RawReview] = []
    seen_texts: set[str] = set()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }

    with requests.Session() as session:
        session.headers.update(headers)
        for query in search_queries:
            if len(raw_reviews) >= limit:
                break

            quoted_query = urllib.parse.quote(query)
            url = f"https://old.reddit.com/search?q={quoted_query}&sort=new"

            try:
                # Reuse HTTP session
                response = session.get(url, timeout=20)
                if response.status_code == 429:
                    # Respect rate limits and retry after a short delay
                    time.sleep(2)
                    response = session.get(url, timeout=20)

                if response.status_code != 200:
                    print(f"Failed to fetch Reddit search results for query '{query}': {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                results = soup.select(".search-result")

                for res in results:
                    title_el = res.select_one("a.search-title")
                    if not title_el:
                        continue

                    url_path = title_el.get("href", "")
                    if url_path.startswith("/"):
                        post_url = f"https://www.reddit.com{url_path}"
                    else:
                        post_url = url_path

                    # Extract post ID
                    fullname = res.get("data-fullname") or ""
                    post_id = fullname.split("_")[-1] if "_" in fullname else ""
                    if not post_id and "comments/" in post_url:
                        try:
                            parts = post_url.split("comments/")
                            if len(parts) > 1:
                                post_id = parts[1].split("/")[0]
                        except Exception:
                            post_id = ""

                    # Extract clean review text excluding metadata/subreddit details
                    review_text = _extract_reddit_post(res)
                    if not review_text:
                        continue

                    # Deduplicate by text content
                    norm_text = " ".join(review_text.split()).strip().lower()
                    if norm_text in seen_texts:
                        continue
                    seen_texts.add(norm_text)

                    # Extract date
                    time_el = res.select_one("time")
                    date_str = ""
                    if time_el and time_el.has_attr("datetime"):
                        dt_val = time_el["datetime"]
                        if dt_val:
                            try:
                                if "T" in dt_val:
                                    date_str = dt_val.split("T")[0]
                                else:
                                    date_str = dt_val
                            except Exception:
                                date_str = ""

                    if not post_id:
                        post_id = stable_id("reddit", review_text, post_url)

                    raw_reviews.append(
                        RawReview(
                            id=post_id,
                            source="reddit",
                            review=review_text,
                            rating=None,
                            date=date_str,
                            url=post_url,
                        )
                    )

                    if len(raw_reviews) >= limit:
                        break

                # Polite delay between query requests
                time.sleep(1.0)

            except Exception as e:
                print(f"Error collecting Reddit reviews for query '{query}': {e}")

    return raw_reviews

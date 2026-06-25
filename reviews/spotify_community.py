from __future__ import annotations

import copy
import time
import urllib.parse
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

from reviews.models import RawReview
from reviews.utils import clean_text, stable_id


DEFAULT_COMMUNITY_SEARCH_URL_TEMPLATE = (
    "https://community.spotify.com/t5/forums/searchpage/tab/message?q="
)

DEFAULT_COMMUNITY_QUERIES = [
    "Discover Weekly",
    "Daily Mix",
    "AI DJ",
    "recommendations",
    "Smart Shuffle",
    "recommendation algorithm",
    "Radio",
    "playlists",
    "new music",
]


def _extract_community_post(candidate: BeautifulSoup) -> str:
    """Extracts only the original post title and body, excluding all metadata, comments, stats, and breadcrumbs."""
    element = copy.copy(candidate)

    # List of metadata/interaction selectors to decompose
    unwanted_selectors = [
        ".lia-message-author", ".lia-message-dates", ".lia-message-statistics",
        ".lia-message-actions", ".lia-message-views", ".lia-message-likes",
        ".lia-message-replies", ".lia-message-author-username", ".lia-message-author-avatar",
        ".lia-message-read-count", ".lia-message-kudos", ".lia-message-rating",
        ".lia-message-feedback", ".lia-message-breadcrumb", ".lia-breadcrumb",
        ".lia-message-meta", ".lia-message-metadata", ".lia-message-tags",
        ".lia-tags", "time", ".time", ".date", ".author", ".likes", ".views", ".replies",
        ".metadata", ".meta", ".breadcrumbs", ".lia-message-view-meta", ".lia-message-meta-items",
        ".lia-message-comments", ".lia-message-replies-container", ".replies-wrapper",
        ".lia-component-reply-button", ".lia-button-group", ".lia-menu-bar",
        ".lia-user-avatar", ".lia-user-rank", ".lia-user-online-status"
    ]
    for selector in unwanted_selectors:
        for tag in element.select(selector):
            tag.decompose()

    # Find title and body specifically
    title_el = element.select_one(".lia-message-subject, h3, h2, .lia-message-title, a.lia-link-navigation")
    body_el = element.select_one(".lia-message-body-content, .lia-search-match-snippet, .lia-message-body, .lia-message-body-text")

    title_text = ""
    body_text = ""

    if title_el:
        title_text = clean_text(title_el.get_text(" ", strip=True))
        title_el.decompose()

    if body_el:
        body_text = clean_text(body_el.get_text(" ", strip=True))
    else:
        body_text = clean_text(element.get_text(" ", strip=True))

    # De-duplicate title from body if duplicated
    if title_text and body_text:
        if body_text.startswith(title_text):
            body_text = body_text[len(title_text):].strip()

    review_parts = []
    if title_text:
        review_parts.append(title_text)
    if body_text and body_text not in title_text:
        review_parts.append(body_text)

    return clean_text(" ".join(review_parts))


def collect_spotify_community_reviews(
    limit: int,
    search_url: str | None = None,
) -> list[RawReview]:
    """Collects Spotify discovery-related feedback from Spotify Community search results.
    
    If search_url is provided, it scrapes that specific page.
    Otherwise, it queries all default discovery keywords.
    """
    raw_reviews: list[RawReview] = []
    seen_texts: set[str] = set()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }

    # If a specific search URL is provided, we just scrape that single page
    if search_url is not None:
        try:
            with requests.Session() as session:
                session.headers.update(headers)
                response = session.get(search_url, timeout=20)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                candidates = soup.select("article, .lia-message-view-wrapper, .lia-list-row")
                for candidate in candidates:
                    text = _extract_community_post(candidate)
                    if len(text) < 40:
                        continue
                    norm_text = " ".join(text.split()).strip().lower()
                    if norm_text in seen_texts:
                        continue
                    seen_texts.add(norm_text)

                    link = candidate.find("a", href=True)
                    url = urljoin(search_url, link["href"]) if link else search_url
                    raw_reviews.append(
                        RawReview(
                            id=stable_id("spotify_community", text, url),
                            source="spotify_community",
                            review=text,
                            rating=None,
                            date="",
                            url=url,
                        )
                    )
                    if len(raw_reviews) >= limit:
                        break
        except Exception as e:
            print(f"Error collecting Spotify Community reviews from search URL {search_url}: {e}")
        return raw_reviews

    # Otherwise, loop over all default search queries using a reused Session
    with requests.Session() as session:
        session.headers.update(headers)
        for query in DEFAULT_COMMUNITY_QUERIES:
            if len(raw_reviews) >= limit:
                break

            quoted_query = urllib.parse.quote(query)
            query_url = f"{DEFAULT_COMMUNITY_SEARCH_URL_TEMPLATE}{quoted_query}"

            try:
                response = session.get(query_url, timeout=20)
                if response.status_code != 200:
                    print(f"Failed to fetch Spotify Community search for query '{query}': {response.status_code}")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                candidates = soup.select("article, .lia-message-view-wrapper, .lia-list-row")

                for candidate in candidates:
                    text = _extract_community_post(candidate)
                    if len(text) < 40:
                        continue
                    norm_text = " ".join(text.split()).strip().lower()
                    if norm_text in seen_texts:
                        continue
                    seen_texts.add(norm_text)

                    link = candidate.find("a", href=True)
                    url = urljoin(query_url, link["href"]) if link else query_url
                    raw_reviews.append(
                        RawReview(
                            id=stable_id("spotify_community", text, url),
                            source="spotify_community",
                            review=text,
                            rating=None,
                            date="",
                            url=url,
                        )
                    )
                    if len(raw_reviews) >= limit:
                        break

                # Polite delay between requests
                time.sleep(1.0)

            except Exception as e:
                print(f"Error collecting Spotify Community reviews for query '{query}': {e}")

    return raw_reviews

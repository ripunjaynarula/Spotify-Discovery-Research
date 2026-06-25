# Spotify Discovery Research

AI-powered review analysis workflow for Spotify music discovery research.

This project collects user feedback, filters for assignment relevance, extracts structured product insights with an LLM, and generates a presentation-ready theme summary. The current LLM transport uses OpenRouter through plain HTTP requests.

---

## Architecture

```text
       Collectors (Google Play, Reddit, Spotify Community)
                           ↓
                    raw_reviews.csv
                           ↓
             Deterministic Pre-filtering
                           ↓
                  AI Relevance Filter
                           ↓
                  filtered_reviews.csv
                           ↓
                AI Insight Extraction
                           ↓
                analyzed_reviews.csv
                           ↓
                   Theme Summary
                           ↓
                  theme_summary.md
```

---

## Folder Structure

```text
spotify-discovery-research/
  analysis/     LLM filtering, insight extraction, and theme summary code
  reviews/      Review source collectors and raw review normalization
  data/         Generated CSV datasets
  output/       Generated markdown summaries and future charts
  prototype/    Future demo or prototype artifacts
  docs/         Architecture notes
```

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

---

## Environment Variables

Required for LLM filtering and analysis:

```text
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=deepseek/deepseek-chat
```

No API keys are required for Reddit or Spotify Community collection.

---

## Execution Order and Pipeline Explanation

The pipeline runs sequentially in these steps:

1. **Review Collection**: Fetches reviews from Android (Google Play Store), Reddit public search, and Spotify Community boards. Normalizes raw reviews into a single deduplicated dataset (`data/raw_reviews.csv`).
2. **Relevance Filtering**: Passes raw reviews through a deterministic pre-filter. Reviews containing no discovery-related keywords are immediately rejected to save API costs. The remaining reviews are analyzed by the LLM for deep relevance and filtered into `data/filtered_reviews.csv` (rejected reviews are logged in `data/rejected_reviews.csv`).
3. **Insight Extraction**: Extracts structured PM insights (e.g. pain points, discovery surfaces, root causes, user segments) from relevant reviews. Responses are strictly validated against controlled vocabularies and saved to `data/analyzed_reviews.csv`.
4. **Theme Summary**: Aggregates the structured insights by counting frequencies and calculating shares of meaningful labels. It extracts representative reviews directly from the data and writes a markdown report to `output/theme_summary.md`.

---

## Example Commands

### 1. Collect Reviews

```powershell
python -m reviews.collect --sources google_play reddit spotify_community --limit 100
```

### 2. Run Relevance Filtering

```powershell
python -m analysis.filter_reviews
```

### 3. Run Insight Extraction

```powershell
python -m analysis.analyze_reviews
```

### 4. Generate Theme Summary

```powershell
python -m analysis.theme_summary
```

---

## Supported Review Sources

1. **Google Play Store**: Fetches reviews using the `google-play-scraper` library.
2. **Reddit**: Collects public Reddit discussions directly from public search pages (`old.reddit.com`) using HTTP requests and BeautifulSoup. **No API credentials are required.**
3. **Spotify Community**: Collects discussions from search queries on the official community forum.

---

## Technical Quality Improvements

### 1. Improved Spotify Community Collector
- Extracts **only** the original post title and body from the forum markup.
- Completely decomposes and excludes forum meta-information: replies, comments, timestamps, author names, statistics (views, kudos, ratings, read counts), avatar classes, buttons, menus, and breadcrumbs.
- Normalizes all whitespaces and cleans HTML structure.

### 2. Improved Reddit Collector
- Extracts **only** the post title and self-text snippet.
- Excludes and decomposes subreddit links, flairs, usernames, vote/score numbers, comment counts, timestamps, and header metadata.
- Normalizes all whitespaces.

### 3. Clean Representative Reviews
In the theme summary step:
- The representative review is selected directly from the original review text in `analyzed_reviews.csv` (never uses metadata or parsed forum info).
- Undergoes a rigorous cleanup: strips HTML tags, removes duplicate punctuation (e.g., reduces repeated dots or exclamation marks to single ones), normalizes repeated whitespace, and preserves complete words.
- Truncates precisely to a maximum of 120 characters, appends an ellipsis on word boundaries, and escapes all markdown table/format special characters (`|`, `*`, `_`, `\`, etc.) to prevent table layout corruption.

### 4. Behavioural User Segmentation
The insight extraction prompt has been updated to prefer behavioural segments with the following priority order:
1. `Discover Weekly User`
2. `AI DJ User`
3. `Smart Shuffle User`
4. `Radio User`
5. `Playlist User`
6. `Artist Explorer`
7. `Casual Listener`
8. `Heavy Listener`
9. `Student`
10. `Working Professional`
- Classified as `Premium User` or `Free User` **only** if the subscription model is explicitly central to the user's feedback.

### 5. Controlled Vocabulary Enforcement
- LLM prompts strictly enforce that `root_cause`, `discovery_surface`, and `user_segment` must reside inside the controlled vocabularies.
- The pipeline programmatically validates labels case-insensitively and coerses them to exact vocabulary casing, coersing unrecognized labels or low-confidence extractions directly to `"unknown"`.

---

## Limitations

- **Rate Limits**: Scraping old.reddit.com and the Spotify Community depends on HTTP requests; aggressive calling may result in temporary HTTP 429 rate limiting.
- **Data Truncation**: Search page snippets on Reddit do not contain the full post body if it is exceptionally long.
- **Deterministic Pre-filtering**: It is keyword-based; there is a minor possibility that a relevant post that uses highly unusual synonyms could be skipped.

---

## Future Improvements

- **Parallel Processing**: Batching LLM requests concurrently to speed up extraction.
- **Incremental Collection**: Skipping already-scraped posts/reviews to save API cost.
- **Semantic Clustering**: Using text embeddings to group themes rather than exact text matching.

---

## Repository File Guide

Here is a guide to every changed file in this refactor and why it changed:

### 1. `reviews/utils.py`
- **Change**: Refactored `dedupe_reviews` to deduplicate by both `review.id` and normalized lowercase `review.review` text to prevent duplicates across queries and sources.

### 2. `reviews/reddit.py`
- **Change**: Replaced PRAW/Reddit API dependency with a public HTML scraper searching `old.reddit.com`. Decomposes flairs, author profiles, timestamps, comments count, and scores to keep only title + snippet content.

### 3. `reviews/spotify_community.py`
- **Change**: Upgraded the collector to loop through multiple queries, reuse sessions, and decompose replies, comments, kudos, read counts, user ranks, and avatars.

### 4. `analysis/schema.py`
- **Change**: Updated `ANALYSIS_FIELDS` and prompts. Added the behavioural user segment vocabulary. Created a shared `clamp_confidence` helper.

### 5. `analysis/llm_client.py`
- **Change**: Added connection pooling reuse (`requests.Session`) and validation/coercion against allowed lists. Imported the shared `clamp_confidence` helper to remove duplicates.

### 6. `analysis/analyze_reviews.py`
- **Change**: Changed default input path to `data/filtered_reviews.csv` and aligned column outputs with the updated `schema.py`.

### 7. `analysis/filter_reviews.py`
- **Change**: Added keyword-based deterministic pre-filtering and updated the system prompt. Removed local `_clamp_confidence` duplicate and imported the shared one.

### 8. `analysis/theme_summary.py`
- **Change**: Redesigned tables (removed Rank, added Representative Review), consolidated ignored values, implemented alphabetical sorting of tied themes, and added `clean_representative_review` to clean HTML, duplicate punctuation, normalize spacing, truncate at word boundaries, and escape markdown characters.

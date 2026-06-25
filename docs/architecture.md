# Project Architecture

## Goal

Build a modular review analysis engine that collects Spotify user feedback, analyzes discovery-related pain points with a provider-agnostic LLM layer, clusters recurring themes, and generates charts for a Product Management assignment.

## Repository Layout

```text
reviews/      Phase 1 source collectors and CSV normalization
analysis/     LLM extraction modules and future theme clustering modules
data/         Generated datasets such as raw_reviews.csv and analyzed_reviews.csv
output/       Generated summaries, charts, and reports
prototype/    Future lightweight demo or notebook artifacts
docs/         Architecture and operating documentation
```

## Data Contract

Phase 1 writes `data/raw_reviews.csv` with the exact fields:

```text
id,source,review,rating,date,url
```

Each collector returns normalized `RawReview` objects. The writer handles deduplication and CSV serialization.

The LLM relevance filter reads `data/raw_reviews.csv` and writes `data/filtered_reviews.csv` with the same fields. `raw_reviews.csv` remains the unfiltered source dataset.

## Phase Plan

1. **Collection**: collect Google Play, Reddit, and optionally Spotify Community feedback into `raw_reviews.csv`.
2. **Relevance filtering**: use the LLM provider to keep only assignment-relevant reviews in `filtered_reviews.csv`.
3. **AI extraction**: read filtered reviews and extract pain point, desired outcome, behaviour, emotion, segment, root cause, and confidence into `analyzed_reviews.csv`.
4. **Theme synthesis**: count structured labels from analyzed reviews and generate `theme_summary.md`.
5. **Charts**: generate automatic visualizations from analyzed data.

## Design Principles

- Keep source collection independent from analysis.
- Use lazy imports for optional external integrations.
- Normalize all sources into one stable CSV schema.
- Avoid framework lock-in and unnecessary infrastructure.
- Keep generated data out of git by default.
- Keep clustering separate from per-review extraction.
- Use only analyzed review data for Phase 3 summaries.

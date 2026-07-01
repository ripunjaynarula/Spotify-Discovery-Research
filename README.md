# Spotify Discovery Research Pipeline

AI-powered review analysis workflow for a Product Management case study on Spotify music discovery.

The project collects user feedback from multiple sources, filters for relevance, extracts structured PM insights with an LLM, and generates a presentation-ready theme summary. It ships both a command-line interface (CLI) and a polished Streamlit web application.

---

## Architecture

```text
       Collectors (Google Play · Reddit · Spotify Community)
                           ↓
                    raw_reviews.csv
                           ↓
             Deterministic Keyword Pre-filter
                           ↓
                  AI Relevance Classifier
                           ↓
                  filtered_reviews.csv
                           ↓
                AI Insight Extraction
                           ↓
                analyzed_reviews.csv
                           ↓
                    Theme Clustering
                           ↓
                  theme_summary.md
```

---

## Folder Structure

```
spotify-discovery-research/
  .streamlit/          Streamlit theme and server settings
  analysis/            Filter, analyze, and summarize modules
  reviews/             Source collector modules and data models
  data/                Generated CSV datasets and pipeline metadata
  output/              Generated markdown reports
  config.py            Centralized paths, secrets, and defaults
  streamlit_app.py     Multipage Streamlit web application
  requirements.txt     Python dependencies
```

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in your keys
```

---

## Configuration

All runtime settings are resolved centrally in `config.py`.

Configuration priority is:
1. Streamlit Secrets (Streamlit Community Cloud)
2. `.env` (local development)
3. OS environment variables
4. Built-in defaults

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes (LLM stages) | API key for OpenRouter |
| `OPENROUTER_MODEL` | No | Default: `openai/gpt-4o-mini` |
| `LLM_PROVIDER` | No | Default: `openrouter` |
| `GOOGLE_PLAY_COUNTRY` | No | Default: `us` |
| `GOOGLE_PLAY_LANGUAGE` | No | Default: `en` |

No API keys are required for Reddit or Spotify Community collection.

---

## Running Locally

### Web Application

```powershell
python -m streamlit run streamlit_app.py
```

### CLI (individual pipeline stages)

```powershell
# Collect
python -m reviews.collect --sources google_play reddit spotify_community --limit 100

# Filter
python -m analysis.filter_reviews

# Analyze
python -m analysis.analyze_reviews

# Summarize
python -m analysis.theme_summary
```

---

## Streamlit App Navigation

| Page | Description |
|---|---|
| **Dashboard** | Executive KPIs, opportunity statement, funnel chart, annotated pipeline diagram, and a **Run Complete Pipeline** button |
| **Collect Reviews** | Source selector, limit slider, per-source status table, donut chart |
| **Filter Reviews** | Batch size / confidence controls, progress bar, relevance donut |
| **Analyze Reviews** | Batch controls, Plotly bar / treemap / donut / histogram charts |
| **Theme Summary** | Confidence cutoff, markdown preview, download |
| **Outputs** | Report cards for every artifact with filename, size, timestamp, and download |

---

## Visualizations

| Chart | Type |
|---|---|
| Review source distribution | Donut |
| Relevant vs irrelevant | Donut |
| Pain points | Horizontal bar |
| Root causes | Horizontal bar |
| Discovery surfaces | Horizontal bar |
| User segments | Treemap |
| Emotion distribution | Donut |
| LLM confidence | Histogram |
| Pipeline stage volumes | Funnel |

---

## Reddit Fallback Caching Strategy

Reddit blocks automated HTTP requests from cloud-hosted servers (HTTP 403).

The application handles this gracefully:

1. **Live collection is attempted first.** If it succeeds the reviews are saved to `data/reddit_cache.csv` as a fresh snapshot.
2. **On HTTP 403 or any network failure**, the most recent `data/reddit_cache.csv` is loaded instead. The collection stage continues uninterrupted.
3. **A source status table** is displayed after collection showing each source's mode (Live / Cached), review count, and any error detail.
4. **If cached data does not exist** (first cloud deployment with no prior cache), Reddit is skipped gracefully and the pipeline continues with the remaining sources.
5. **The pipeline only fails** if every source returns zero reviews.

To pre-seed the cache before deploying to Streamlit Cloud, run the collection locally once:

```powershell
python -m reviews.collect --sources reddit --limit 100
# This writes data/raw_reviews.csv; copy the reddit rows to data/reddit_cache.csv
# Or run the full app locally and let it create reddit_cache.csv automatically.
```

---

## Execution UX

Pipeline stages never print raw CLI output directly on the page.  Instead:

- A **progress bar** advances through the stages.
- **`st.status` containers** announce the current stage (e.g. *Stage 2/4 — Filtering reviews…*).
- All stdout/stderr output is routed into a collapsed **"Execution Logs"** expander for debugging.
- Failures now preserve the original traceback and exception details, including HTTP/LLM/provider errors when available.
- Review text is truncated to 800 characters before LLM submission, and smaller default batch sizes reduce prompt pressure.
- **Retry notices** are surfaced inline: *"Retrying 3 review(s) because the AI returned an incomplete response."*
- Elapsed time is shown at each stage and on completion.

---

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub (`.streamlit/secrets.toml` and `data/*.csv` are git-ignored).
2. Open [share.streamlit.io](https://share.streamlit.io), connect your repo, and select `streamlit_app.py` as the entry point.
3. Under **Advanced Settings → Secrets**, add:

```toml
OPENROUTER_API_KEY = "sk-or-..."
OPENROUTER_MODEL   = "openai/gpt-4o-mini"
LLM_PROVIDER       = "openrouter"
```

4. Click **Deploy**.  
   Local development uses `.env`; Streamlit Community Cloud uses Secrets. No code changes are required between the two environments.

---

## Limitations

| Limitation | Mitigation |
|---|---|
| Reddit blocks cloud requests (HTTP 403) | Automatic fallback to `reddit_cache.csv` |
| OpenRouter rate limits / timeouts | Retry logic with exponential back-off; `continue_on_error` mode |
| Google Play scraping rate limits | Polite session reuse; limit slider |
| Spotify Community HTML changes | Selector-based scraper may need updating if community layout changes |
| LLM hallucinations | Controlled vocabulary validation coerces invalid labels to `"unknown"` |

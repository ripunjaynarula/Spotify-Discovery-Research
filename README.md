# Spotify Discovery Research Pipeline

AI-powered review analysis workflow for Spotify music discovery research.

This project collects user feedback, filters for assignment relevance, extracts structured product insights with an LLM, and generates a presentation-ready theme summary. The pipeline includes both a command-line interface (CLI) and a polished, minimal, and modern Streamlit web application.

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
  .streamlit/   Streamlit UI configuration and theme settings
  analysis/     LLM relevance filtering, insight extraction, and theme summary
  reviews/      Review source collectors and raw review normalization
  data/         Generated CSV datasets and pipeline metadata
  output/       Generated markdown summaries and reports
  config.py     Centralized paths and defaults settings
  streamlit_app.py Multipage Streamlit application
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
OPENROUTER_API_KEY=your_openrouter_key
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=deepseek/deepseek-chat
```

No API keys are required for Reddit or Spotify Community collection.

---

## Running Locally

### 1. Web Application

To run the interactive multipage web application locally:

```powershell
streamlit run streamlit_app.py
```

### 2. Command Line Interface (CLI)

The underlying analysis pipeline can also be run sequentially via CLI commands:

#### Collect Reviews
```powershell
python -m reviews.collect --sources google_play reddit spotify_community --limit 100
```

#### Run Relevance Filtering
```powershell
python -m analysis.filter_reviews
```

#### Run Insight Extraction
```powershell
python -m analysis.analyze_reviews
```

#### Generate Theme Summary
```powershell
python -m analysis.theme_summary
```

---

## Multipage App Navigation

The Streamlit web interface is partitioned into 6 distinct navigation views:

1. **Dashboard**: View the block-flow architecture of the pipeline annotated with dynamic record counts from the latest run, high-level metrics, and the **Executive Insights** engine showing:
   * **Top Pain Point**: Most frequent discovery-related issue.
   * **Top Root Cause**: Most frequent system cause.
   * **Critical Segment**: Most affected user segment.
   * **Primary Surface**: Most mentioned product surface.
   * **Opportunity Recommendation**: A dynamically formulated PM opportunity statement linking the top segment, surface, root cause, and pain point.
2. **Collect Reviews**: Run the collection modules (Google Play, Reddit, Spotify Community) interactively with configurable review limits. Features real-time log streaming and a preview of `raw_reviews.csv`.
3. **Filter Reviews**: Run deterministic keyword pre-filtering and AI relevance filtering. Displays metrics cards for relevance distribution and confidence, with a preview of `filtered_reviews.csv`.
4. **Analyze Reviews**: Run structured PM insight extraction (pain points, root causes, surfaces, segments, emotions, confidence). Displays interactive Plotly visualizations.
5. **Theme Summary**: Compile structured labels into a readable markdown report. Renders the final markdown document in the UI for review.
6. **Outputs**: A download center to download all generated datasets and reports in CSV or Markdown formats formatted as clean visual cards.

---

## Visualizations

The dashboard contains the following customized chart types:
- **Review Source Distribution**: Donut chart representing collection proportions.
- **Relevance Distribution**: Donut chart representing relevance filter ratios.
- **Processing Stage Funnel**: A Funnel chart tracking volume reduction from collection, pre-filtering, relevance filtering, and insight extraction.
- **Root Causes, Pain Points, Surfaces**: Sorted horizontal bar charts in Spotify-green accents.
- **User Segments**: A Plotly Treemap chart representing user segments.
- **Confidence Distribution**: Styled confidence score histogram.
- **Emotion Distribution**: Donut chart representing user sentiments.

---

## Deploying to Streamlit Community Cloud

This project is configured to deploy directly to Streamlit Community Cloud without modifications:

1. Push your repository to GitHub.
2. Navigate to [Streamlit Share](https://share.streamlit.io/) and select your repository, branch, and `streamlit_app.py` as the entry file.
3. Under **Advanced Settings**, add your environment variables in the **Secrets** text area using TOML format:
   ```toml
   OPENROUTER_API_KEY = "your_actual_openrouter_api_key_here"
   LLM_PROVIDER = "openrouter"
   OPENROUTER_MODEL = "deepseek/deepseek-chat"
   ```
4. Click **Deploy**. The environment variables will be resolved automatically by `config.py` from Streamlit Secrets.

---

## Technical Quality Details

- **Deterministic Pre-Filtering**: Before using the LLM relevance filter, raw text is checked for key discovery terms to skip calling OpenRouter for obviously irrelevant data.
- **Controlled Vocabulary Validation**: Extracted user segments, root causes, and discovery surfaces are programmatically validated case-insensitively. Low confidence extractions or invalid inputs are coerced to "unknown" to prevent AI hallucinations.
- **Connection Pooling**: All HTTP calls reuse connections using `requests.Session` in collectors and the LLM client.
- **Onboarding Empty-States**: Onboarding guides are displayed automatically if raw feedback datasets do not exist yet.
- **"Run Complete Pipeline"**: One-click sidebar orchestrator that executes the pipeline end-to-end (Collect → Filter → Analyze → Summary) and streams logging progress.

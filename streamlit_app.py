"""Spotify Discovery Research – Streamlit application.

This module wires together the existing analysis pipeline into a PM-facing
web application.  It does NOT rewrite or duplicate any business logic; it
only orchestrates the modules that already exist.
"""
from __future__ import annotations

import contextlib
import datetime
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

# ── pipeline imports ──────────────────────────────────────────────────────────
from analysis.analyze_reviews import analyze_reviews, write_analyzed_reviews
from analysis.filter_reviews import (
    build_rejected_reviews,
    classify_relevance,
    filter_relevant_reviews,
    prepare_reviews_for_llm,
    read_raw_reviews,
    write_filtered_reviews,
    write_rejected_reviews,
)
from analysis.theme_summary import (
    build_theme_summary,
    read_analyzed_reviews,
    write_theme_summary,
)
from reviews.google_play import collect_google_play_reviews
from reviews.reddit import collect_reddit_reviews
from reviews.spotify_community import collect_spotify_community_reviews
from reviews.utils import write_raw_reviews_csv

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Discovery Research",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# TYPES
# ─────────────────────────────────────────────────────────────────────────────
SourceStatus = Literal["success", "cached", "skipped", "failed"]


@dataclass
class CollectionResult:
    source: str
    status: SourceStatus
    reviews: list = field(default_factory=list)
    error: str = ""

    @property
    def count(self) -> int:
        return len(self.reviews)

    @property
    def mode(self) -> str:
        return "Cached" if self.status == "cached" else "Live"


# ─────────────────────────────────────────────────────────────────────────────
# LOG CAPTURE  (stdout → collapsed expander, retries → inline warnings)
# ─────────────────────────────────────────────────────────────────────────────
class _LogCapture:
    """Redirects stdout/stderr into a StringIO buffer and surfaces retry events."""

    def __init__(self, log_placeholder, notify_placeholder=None):
        self._log = log_placeholder
        self._notify = notify_placeholder
        self._buf = io.StringIO()

    def write(self, text: str) -> None:
        if not text:
            return
        self._buf.write(text)
        self._log.code(self._buf.getvalue(), language=None)
        if self._notify and text.strip():
            self._surface_retry_notice(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def _surface_retry_notice(self, text: str) -> None:
        if "LLM returned no result for review ids" in text:
            try:
                ids_raw = text.split("review ids:")[-1].strip()
                count = len([x for x in ids_raw.split(",") if x.strip()])
                msg = (
                    f"Retrying {count} review(s) because the AI returned "
                    "an incomplete response."
                )
            except Exception:
                msg = "Retrying reviews because the AI returned an incomplete response."
            self._notify.warning(msg)
        elif "Retrying missing review ids only in" in text:
            try:
                secs = text.split("only in")[-1].split("seconds")[0].strip()
                self._notify.info(f"Waiting {secs}s before retry…")
            except Exception:
                pass

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return self._buf.getvalue()


@contextlib.contextmanager
def capture_logs(log_placeholder, notify_placeholder=None):
    """Context manager: redirect stdout/stderr to the Streamlit code block."""
    capture = _LogCapture(log_placeholder, notify_placeholder)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = capture
    sys.stderr = capture
    try:
        yield capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ─────────────────────────────────────────────────────────────────────────────
# REDDIT FAULT-TOLERANT COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
def collect_reddit_fault_tolerant(limit: int) -> CollectionResult:
    """
    Attempt live Reddit collection.
    • HTTP 403 / network failure → fall back to reddit_cache.csv.
    • Cache missing           → return empty with status=skipped.
    """
    import requests

    try:
        reviews = collect_reddit_reviews(limit=limit)
        # Persist a fresh cache so future cloud runs have data
        if reviews:
            df = pd.DataFrame([r.__dict__ for r in reviews])
            df.to_csv(config.REDDIT_CACHE_CSV, index=False)
        return CollectionResult(source="reddit", status="success", reviews=reviews)

    except Exception as exc:
        error_str = str(exc)
        is_blocked = (
            "403" in error_str
            or "Forbidden" in error_str
            or "ConnectionError" in error_str
            or "timeout" in error_str.lower()
        )

        if config.REDDIT_CACHE_CSV.exists():
            try:
                from reviews.models import RawReview
                df = pd.read_csv(config.REDDIT_CACHE_CSV, dtype=str).fillna("")
                cached = [
                    RawReview(
                        id=row.get("id", ""),
                        source="reddit",
                        review=row.get("review", ""),
                        rating=None,
                        date=row.get("date", ""),
                        url=row.get("url", ""),
                    )
                    for _, row in df.iterrows()
                    if row.get("review", "").strip()
                ]
                fallback_error = (
                    "Live Reddit collection unavailable (HTTP 403). Using cached Reddit dataset."
                    if is_blocked
                    else f"Live Reddit collection failed: {error_str}"
                )
                return CollectionResult(
                    source="reddit",
                    status="cached",
                    reviews=cached,
                    error=fallback_error,
                )
            except Exception as cache_exc:
                return CollectionResult(
                    source="reddit",
                    status="failed",
                    error=f"Live failed ({error_str}); cache unreadable ({cache_exc})",
                )

        fallback_error = (
            "Live Reddit collection unavailable (HTTP 403). Using cached Reddit dataset."
            if is_blocked
            else error_str
        )
        return CollectionResult(
            source="reddit",
            status="skipped" if is_blocked else "failed",
            error=fallback_error,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def get_mode(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return "unknown"
    series = df[~df[col].astype(str).str.lower().isin(config.IGNORE_VALUES)][col]
    return str(series.mode().iloc[0]) if not series.empty else "unknown"


def clean_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    return counts[~counts[col].astype(str).str.lower().isin(config.IGNORE_VALUES)]


def bar_chart(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    data = clean_counts(df, col).sort_values("count", ascending=True)
    fig = px.bar(data, x="count", y=col, orientation="h",
                 title=title, template="plotly_dark")
    fig.update_traces(marker_color="#1DB954")
    fig.update_layout(
        xaxis_title="Count", yaxis_title=None,
        height=350, margin=dict(l=180, r=20, t=44, b=40),
    )
    return fig


def donut_chart(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    data = clean_counts(df, col)
    fig = px.pie(data, values="count", names=col, hole=0.5,
                 title=title, template="plotly_dark",
                 color_discrete_sequence=px.colors.sequential.Greens_r)
    fig.update_layout(margin=dict(t=44, b=40, l=40, r=40), height=350)
    return fig


def insight_card(label: str, value: str, accent: str = "#1DB954") -> str:
    return f"""
    <div style="background:#181818;padding:18px 20px;border-radius:8px;
                border-left:4px solid {accent};height:100%;">
      <p style="color:#b3b3b3;margin:0;font-size:.78em;text-transform:uppercase;
                letter-spacing:.06em;">{label}</p>
      <p style="color:#fff;margin:6px 0 0 0;font-size:1.05em;
                font-weight:500;line-height:1.35;">{value}</p>
    </div>"""


def onboarding_card(icon: str, heading: str, body: str) -> str:
    return f"""
    <div style="background:#181818;padding:24px;border-radius:10px;
                border:1px solid #282828;text-align:center;">
      <div style="font-size:2em;margin-bottom:10px;">{icon}</div>
      <h4 style="color:#1DB954;margin:0 0 8px 0;font-weight:500;">{heading}</h4>
      <p style="color:#b3b3b3;margin:0;font-size:.9em;line-height:1.55;">{body}</p>
    </div>"""


def elapsed(start: float) -> str:
    secs = int(time.perf_counter() - start)
    return f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"


def handle_runtime_error(
    exc: Exception,
    *,
    stage_status=None,
    progress=None,
    log_capture=None,
    message: str = "Operation failed",
) -> None:
    if log_capture is not None:
        log_capture.write("\n\n=== FULL EXCEPTION TRACEBACK ===\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=log_capture)
    if stage_status is not None:
        stage_status.error(f"{message}: {exc}")
    st.exception(exc)
    if progress is not None:
        progress.empty()


# ─────────────────────────────────────────────────────────────────────────────
# ARGS SHIMS  (replace argparse.Namespace for programmatic calls)
# ─────────────────────────────────────────────────────────────────────────────
class _FilterArgs:
    input = config.RAW_REVIEWS_CSV
    output = config.FILTERED_REVIEWS_CSV
    batch_size = config.DEFAULT_BATCH_SIZE_FILTER
    model = config.OPENROUTER_MODEL
    max_retries = config.DEFAULT_MAX_RETRIES
    retry_delay_seconds = config.DEFAULT_RETRY_DELAY_SECONDS
    min_review_length = config.DEFAULT_MIN_REVIEW_LENGTH
    min_confidence = 0.0


class _AnalyzeArgs:
    input = config.FILTERED_REVIEWS_CSV
    output = config.ANALYZED_REVIEWS_CSV
    batch_size = config.DEFAULT_BATCH_SIZE_ANALYZE
    model = config.OPENROUTER_MODEL
    max_retries = config.DEFAULT_MAX_RETRIES
    retry_delay_seconds = config.DEFAULT_RETRY_DELAY_SECONDS
    continue_on_error = True


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGES  (each returns a human-readable result string or raises)
# ─────────────────────────────────────────────────────────────────────────────
def run_stage_collect(
    sources: list[str],
    limit: int,
    community_url: str = "",
) -> tuple[list[CollectionResult], int]:
    results: list[CollectionResult] = []

    if "google_play" in sources:
        try:
            reviews = collect_google_play_reviews(
                limit=limit,
                country=config.GOOGLE_PLAY_COUNTRY,
                language=config.GOOGLE_PLAY_LANGUAGE,
            )
            results.append(CollectionResult("google_play", "success", reviews))
        except Exception as exc:
            results.append(CollectionResult("google_play", "failed", error=str(exc)))

    if "reddit" in sources:
        results.append(collect_reddit_fault_tolerant(limit))

    if "spotify_community" in sources:
        try:
            kwargs: dict = {"limit": limit}
            if community_url:
                kwargs["search_url"] = community_url
            reviews = collect_spotify_community_reviews(**kwargs)
            results.append(CollectionResult("spotify_community", "success", reviews))
        except Exception as exc:
            results.append(CollectionResult("spotify_community", "failed", error=str(exc)))

    all_reviews = []
    for r in results:
        all_reviews.extend(r.reviews)

    if not all_reviews:
        raise RuntimeError("All sources failed to return reviews.")

    write_raw_reviews_csv(all_reviews, config.RAW_REVIEWS_CSV)
    return results, len(all_reviews)


def run_stage_filter(args=None) -> dict:
    args = args or _FilterArgs()
    raw = read_raw_reviews(args.input)
    reviews_for_llm, removal_counts = prepare_reviews_for_llm(
        raw, min_review_length=args.min_review_length
    )
    relevance_rows = classify_relevance(reviews_for_llm, args)
    filtered = filter_relevant_reviews(reviews_for_llm, relevance_rows, args.min_confidence)
    rejected = build_rejected_reviews(reviews_for_llm, relevance_rows)
    write_filtered_reviews(filtered, args.output)
    write_rejected_reviews(rejected, config.REJECTED_REVIEWS_CSV)

    conf_vals = [float(r["confidence"]) for r in relevance_rows]
    irr = sum(1 for r in relevance_rows if r["relevant"] is not True)
    summary = {
        "total_reviews": len(raw),
        "reviews_sent_to_llm": len(reviews_for_llm),
        "relevant_reviews": len(filtered),
        "irrelevant_reviews": irr,
        "average_confidence": sum(conf_vals) / len(conf_vals) if conf_vals else 0.0,
        "model": config.OPENROUTER_MODEL,
    }
    config.FILTER_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stage_analyze(args=None) -> int:
    args = args or _AnalyzeArgs()
    df = pd.read_csv(args.input, dtype={"id": str})
    df = df[df["review"].astype(str).str.strip() != ""].copy()
    rows = analyze_reviews(df, args)
    write_analyzed_reviews(rows, args.output)
    return len(rows)


def run_stage_summary(min_conf: float = None, top_n: int = None) -> None:
    min_conf = min_conf if min_conf is not None else config.DEFAULT_MIN_CONFIDENCE_SUMMARY
    top_n = top_n or config.DEFAULT_TOP_N_THEMES
    analyzed = read_analyzed_reviews(config.ANALYZED_REVIEWS_CSV, min_conf)
    md = build_theme_summary(analyzed, top_n)
    write_theme_summary(md, config.THEME_SUMMARY_MD)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE STATUS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def render_source_status_table(results: list[CollectionResult]) -> None:
    rows = []
    for r in results:
        status_icon = {
            "success": "✅ Success",
            "cached": "⚠ Cached",
            "skipped": "⏭ Skipped",
            "failed": "❌ Failed",
        }.get(r.status, r.status)
        rows.append({
            "Source": r.source.replace("_", " ").title(),
            "Status": status_icon,
            "Reviews Collected": r.count,
            "Mode": r.mode,
            "Error": r.error or "—",
        })
    st.table(pd.DataFrame(rows))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
if not config.OPENROUTER_API_KEY:
    st.sidebar.warning(
        "**OPENROUTER_API_KEY** not set.  "
        "LLM stages (Filter, Analyze) will fail until you add it to your "
        "secrets or `.env` file."
    )

st.sidebar.title("Spotify Discovery")
st.sidebar.markdown("AI review analysis pipeline.")

navigation_page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Collect Reviews", "Filter Reviews",
     "Analyze Reviews", "Theme Summary", "Outputs"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: `{config.OPENROUTER_MODEL}`")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATASETS  (used by all pages)
# ─────────────────────────────────────────────────────────────────────────────
df_raw = load_csv(config.RAW_REVIEWS_CSV)
df_filt = load_csv(config.FILTERED_REVIEWS_CSV)
df_rej = load_csv(config.REJECTED_REVIEWS_CSV)
df_ana = load_csv(config.ANALYZED_REVIEWS_CSV)

pipeline_started = df_raw is not None and len(df_raw) > 0


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if navigation_page == "Dashboard":
    st.title("Spotify Music Discovery Analytics")
    st.markdown(
        "Executive product intelligence derived from Spotify recommendations feedback."
    )

    # ── Run Complete Pipeline button ──────────────────────────────────────────
    st.markdown("---")
    run_col, _ = st.columns([1, 3])
    run_clicked = run_col.button(
        "Run Complete Pipeline", width="stretch", type="primary"
    )

    if run_clicked:
        t0 = time.perf_counter()
        progress = st.progress(0.0, text="Starting…")
        stage_status = st.empty()
        notify = st.empty()
        logs_exp = st.expander("Execution Logs", expanded=False)

        with logs_exp:
            log_placeholder = st.empty()

        with capture_logs(log_placeholder, notify) as log_capture:
            try:
                # 1 / 4 – Collect
                stage_status.info("**Stage 1/4** — Collecting reviews from all sources…")
                progress.progress(0.05, text="Collecting…")
                col_results, total_collected = run_stage_collect(
                    ["google_play", "reddit", "spotify_community"],
                    config.DEFAULT_LIMIT,
                )
                progress.progress(0.25, text=f"Collected {total_collected} reviews ({elapsed(t0)})")
                stage_status.success(f"**Stage 1/4 complete** — {total_collected} reviews collected.")

                # 2 / 4 – Filter
                stage_status.info("**Stage 2/4** — Filtering reviews for relevance…")
                progress.progress(0.3, text="Filtering…")
                filter_summary = run_stage_filter()
                n_rel = filter_summary["relevant_reviews"]
                progress.progress(0.55, text=f"{n_rel} relevant reviews ({elapsed(t0)})")
                stage_status.success(f"**Stage 2/4 complete** — {n_rel} relevant reviews kept.")

                # 3 / 4 – Analyze
                stage_status.info("**Stage 3/4** — Extracting product insights with AI…")
                progress.progress(0.6, text="Analyzing…")
                n_analyzed = run_stage_analyze()
                progress.progress(0.85, text=f"{n_analyzed} reviews analyzed ({elapsed(t0)})")
                stage_status.success(f"**Stage 3/4 complete** — {n_analyzed} reviews analyzed.")

                # 4 / 4 – Summary
                stage_status.info("**Stage 4/4** — Generating theme summary report…")
                progress.progress(0.9, text="Summarizing…")
                run_stage_summary()
                progress.progress(1.0, text=f"Done in {elapsed(t0)}")
                stage_status.success(
                    f"**Pipeline complete** — finished in {elapsed(t0)}. "
                    "Refresh the page to see updated insights."
                )
                notify.empty()

            except Exception as exc:
                handle_runtime_error(
                    exc,
                    stage_status=stage_status,
                    progress=progress,
                    log_capture=log_capture,
                    message="Pipeline failed",
                )

        st.markdown("---")

    # ── Onboarding ────────────────────────────────────────────────────────────
    if not pipeline_started:
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            onboarding_card("📥", "Step 1 — Collect",
                            "Click 'Run Complete Pipeline' above, or navigate to "
                            "'Collect Reviews' to pull feedback from Google Play, "
                            "Reddit and Spotify Community."),
            unsafe_allow_html=True,
        )
        c2.markdown(
            onboarding_card("🔍", "Step 2 — Filter & Analyze",
                            "The AI relevance filter removes off-topic posts.  "
                            "The insight extractor then tags each review with pain "
                            "points, root causes, user segments and emotions."),
            unsafe_allow_html=True,
        )
        c3.markdown(
            onboarding_card("📊", "Step 3 — Explore",
                            "Once the pipeline has run, this dashboard auto-populates "
                            "with executive KPI cards, funnel charts, and "
                            "Plotly visualizations."),
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Executive insights ────────────────────────────────────────────────────
    if df_ana is not None and len(df_ana) > 0:
        st.subheader("Executive Product Insights")

        top_pp = get_mode(df_ana, "pain_point")
        top_rc = get_mode(df_ana, "root_cause")
        top_seg = get_mode(df_ana, "user_segment")
        top_surf = get_mode(df_ana, "discovery_surface")

        cols = st.columns(4)
        for col, lbl, val in zip(
            cols,
            ["Top Pain Point", "Top Root Cause", "Critical Segment", "Primary Surface"],
            [top_pp, top_rc, top_seg, top_surf],
        ):
            col.markdown(insight_card(lbl, val), unsafe_allow_html=True)

        # Opportunity statement
        if "unknown" not in (top_pp, top_rc, top_seg, top_surf):
            opp = (
                f"Address **{top_rc}** on the **{top_surf}** surface "
                f"to resolve **{top_pp}** for **{top_seg}** users — "
                "our highest-leverage product optimization opportunity."
            )
        else:
            opp = "Complete the full pipeline to generate an opportunity recommendation."

        st.markdown(
            f"""<div style="background:#181818;padding:18px 22px;border-radius:8px;
                border-top:3px solid #1DB954;margin:18px 0 28px 0;">
              <p style="color:#b3b3b3;margin:0;font-size:.78em;text-transform:uppercase;
                        letter-spacing:.06em;">Largest Product Opportunity</p>
              <p style="color:#fff;margin:8px 0 0 0;font-size:1.05em;line-height:1.5;">
                {opp}</p></div>""",
            unsafe_allow_html=True,
        )

    # ── Stage volume funnel ───────────────────────────────────────────────────
    raw_n = len(df_raw) if df_raw is not None else 0
    filt_n = len(df_filt) if df_filt is not None else 0
    ana_n = len(df_ana) if df_ana is not None else 0
    pre_n = raw_n

    if config.FILTER_SUMMARY_JSON.exists():
        try:
            pre_n = json.loads(
                config.FILTER_SUMMARY_JSON.read_text(encoding="utf-8")
            ).get("reviews_sent_to_llm", raw_n)
        except Exception:
            pass

    st.subheader("Review Processing Funnel")
    fig_funnel = go.Figure(go.Funnel(
        y=["Raw Collection", "Passed Pre-filter", "Relevance Filtered", "Analyzed"],
        x=[raw_n, pre_n, filt_n, ana_n],
        textinfo="value+percent initial",
        marker=dict(color=["#3d3d3d", "#2a2a2a", "#1DB954", "#1aa34a"]),
    ))
    fig_funnel.update_layout(
        template="plotly_dark", height=300,
        margin=dict(t=20, b=20, l=160, r=160),
    )
    st.plotly_chart(fig_funnel, width="stretch")

    # ── Annotated process diagram ─────────────────────────────────────────────
    st.subheader("Annotated Pipeline Diagram")
    st.graphviz_chart(f"""
    digraph G {{
        rankdir=LR;
        node [shape=box, style="filled,rounded", color="#1DB954",
              fillcolor="#1E1E1E", fontcolor="#FFFFFF", fontname="Helvetica"];
        edge [color="#1DB954"];
        A [label="Collection\\n{raw_n} reviews"];
        B [label="Relevance Filter\\n{filt_n} passed"];
        C [label="Insight Extraction\\n{ana_n} analyzed"];
        D [label="Theme Summary"];
        A -> B -> C -> D;
    }}
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – COLLECT REVIEWS
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Collect Reviews":
    st.title("Review Collection")
    st.markdown(
        "Pull user feedback from Google Play, Reddit, and Spotify Community.  "
        "If Reddit is blocked on Streamlit Cloud the most recent cached snapshot "
        "is used automatically."
    )

    sources = st.multiselect(
        "Sources to collect from",
        ["google_play", "reddit", "spotify_community"],
        default=["google_play", "reddit", "spotify_community"],
    )
    limit = st.slider("Max reviews per source", 10, 500, config.DEFAULT_LIMIT)
    community_url = st.text_input(
        "Custom Spotify Community URL (optional)",
        placeholder="https://community.spotify.com/t5/…",
    )

    if st.button("Run Collection", width="stretch", type="primary"):
        if not sources:
            st.error("Select at least one source.")
        else:
            t0 = time.perf_counter()
            progress = st.progress(0.0, text="Initialising…")
            stage_status = st.empty()
            logs_exp = st.expander("Execution Logs", expanded=False)
            with logs_exp:
                log_ph = st.empty()

            with capture_logs(log_ph) as log_capture:
                try:
                    stage_status.info("Connecting to review sources…")
                    progress.progress(0.1)
                    col_results, total = run_stage_collect(sources, limit, community_url)
                    progress.progress(1.0, text=f"Done — {total} reviews ({elapsed(t0)})")
                    stage_status.success(f"Collected **{total} reviews** in {elapsed(t0)}.")
                    render_source_status_table(col_results)

                    # Show Reddit cache notices inline
                    for r in col_results:
                        if r.source == "reddit" and r.status == "cached":
                            st.info(
                                "Live Reddit collection is temporarily unavailable because "
                                "Reddit blocked automated cloud requests.  "
                                f"Using the latest cached Reddit dataset ({r.count} reviews)."
                            )
                        elif r.source == "reddit" and r.status == "skipped":
                            st.warning(
                                "Reddit was unreachable and no cached data exists.  "
                                "Continuing with remaining sources."
                            )
                        elif r.status == "failed":
                            st.warning(
                                f"{r.source.replace('_',' ').title()} failed: {r.error}"
                            )
                    st.rerun()

                except RuntimeError as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Collection failed",
                    )
                except Exception as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Unexpected error",
                    )

    # Preview
    if df_raw is not None:
        st.subheader("Raw Reviews Preview")
        st.dataframe(df_raw, width="stretch")

        if "source" in df_raw.columns:
            st.subheader("Source Distribution")
            df_src = df_raw["source"].value_counts().reset_index()
            df_src.columns = ["source", "count"]
            fig = px.pie(
                df_src, values="count", names="source", hole=0.5,
                template="plotly_dark",
                color_discrete_sequence=["#1DB954", "#1aa34a", "#148a3a"],
            )
            fig.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No raw reviews found.  Run collection above to begin.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – FILTER REVIEWS
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Filter Reviews":
    st.title("Relevance Filtering")
    st.markdown(
        "Removes off-topic reviews using a keyword pre-filter followed by an "
        "LLM relevance classifier."
    )

    c1, c2 = st.columns(2)
    batch_size = c1.number_input("Batch size", 1, 200, config.DEFAULT_BATCH_SIZE_FILTER)
    min_len = c2.number_input("Min review length (chars)", 1, 500, config.DEFAULT_MIN_REVIEW_LENGTH)
    min_conf = st.slider("Min relevance confidence", 0.0, 1.0, 0.0, 0.05)

    if st.button("Run Relevance Filtering", width="stretch", type="primary"):
        if df_raw is None:
            st.error("Collect reviews first.")
        else:
            t0 = time.perf_counter()
            progress = st.progress(0.0, text="Loading…")
            stage_status = st.empty()
            notify = st.empty()
            logs_exp = st.expander("Execution Logs", expanded=False)
            with logs_exp:
                log_ph = st.empty()

            with capture_logs(log_ph, notify) as log_capture:
                try:
                    class _Args(_FilterArgs):
                        pass
                    _Args.batch_size = int(batch_size)
                    _Args.min_review_length = int(min_len)
                    _Args.min_confidence = float(min_conf)

                    stage_status.info("Stage 1/3 — Pre-filtering by keywords…")
                    progress.progress(0.2)

                    stage_status.info("Stage 2/3 — Running LLM relevance assessment…")
                    progress.progress(0.4)
                    summary = run_stage_filter(_Args())

                    stage_status.info("Stage 3/3 — Writing datasets…")
                    progress.progress(0.9)

                    n_rel = summary["relevant_reviews"]
                    progress.progress(1.0, text=f"Done ({elapsed(t0)})")
                    stage_status.success(
                        f"**Filtering complete** — {n_rel} relevant reviews retained "
                        f"out of {summary['total_reviews']} in {elapsed(t0)}."
                    )
                    notify.empty()
                    st.rerun()

                except Exception as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Filtering failed",
                    )

    if df_filt is not None:
        st.subheader("Filtered Reviews Preview")
        st.dataframe(df_filt, width="stretch")

        rej_n = len(df_rej) if df_rej is not None else 0
        filt_n = len(df_filt)
        fig = px.pie(
            pd.DataFrame({"label": ["Relevant", "Irrelevant"],
                          "n": [filt_n, rej_n]}),
            values="n", names="label", hole=0.5,
            title="Relevance Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#1DB954", "#3d3d3d"],
        )
        fig.update_layout(height=300, margin=dict(t=44, b=40, l=40, r=40))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No filtered reviews yet.  Run filtering above.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – ANALYZE REVIEWS
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Analyze Reviews":
    st.title("Insight Extraction")
    st.markdown(
        "Tags each relevant review with structured labels: pain point, root cause, "
        "discovery surface, user segment, emotion, and LLM confidence."
    )

    c1, c2 = st.columns(2)
    batch_size = c1.number_input("Batch size", 1, 50, config.DEFAULT_BATCH_SIZE_ANALYZE)
    cont_err = c2.checkbox("Continue on partial AI errors", value=True)

    if st.button("Run Insight Extraction", width="stretch", type="primary"):
        if df_filt is None:
            st.error("Run relevance filtering first.")
        else:
            t0 = time.perf_counter()
            progress = st.progress(0.0, text="Loading filtered data…")
            stage_status = st.empty()
            notify = st.empty()
            logs_exp = st.expander("Execution Logs", expanded=False)
            with logs_exp:
                log_ph = st.empty()

            with capture_logs(log_ph, notify) as log_capture:
                try:
                    class _Args(_AnalyzeArgs):
                        pass
                    _Args.batch_size = int(batch_size)
                    _Args.continue_on_error = cont_err

                    stage_status.info("Sending batches to AI for classification…")
                    progress.progress(0.2)
                    n = run_stage_analyze(_Args())
                    progress.progress(1.0, text=f"Done — {n} reviews ({elapsed(t0)})")
                    stage_status.success(
                        f"**Analysis complete** — {n} reviews classified in {elapsed(t0)}."
                    )
                    notify.empty()
                    st.rerun()

                except Exception as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Analysis failed",
                    )

    if df_ana is not None:
        st.subheader("Analyzed Reviews Preview")
        st.dataframe(df_ana, width="stretch")

        st.markdown("---")
        st.subheader("Product Feedback Visualizations")

        c1, c2 = st.columns(2)
        if "root_cause" in df_ana.columns:
            c1.plotly_chart(
                bar_chart(df_ana, "root_cause", "Root Causes"),
                width="stretch",
            )
        if "pain_point" in df_ana.columns:
            c2.plotly_chart(
                bar_chart(df_ana, "pain_point", "Pain Points"),
                width="stretch",
            )

        c3, c4 = st.columns(2)
        if "discovery_surface" in df_ana.columns:
            c3.plotly_chart(
                bar_chart(df_ana, "discovery_surface", "Discovery Surfaces"),
                width="stretch",
            )
        if "user_segment" in df_ana.columns:
            seg_df = clean_counts(df_ana, "user_segment")
            if not seg_df.empty:
                fig_tree = px.treemap(
                    seg_df, path=["user_segment"], values="count",
                    title="User Segments", template="plotly_dark",
                    color_discrete_sequence=["#1DB954", "#1aa34a", "#148a3a"],
                )
                fig_tree.update_layout(height=350, margin=dict(t=44, b=20, l=20, r=20))
                c4.plotly_chart(fig_tree, width="stretch")

        c5, c6 = st.columns(2)
        if "emotion" in df_ana.columns:
            c5.plotly_chart(
                donut_chart(df_ana, "emotion", "Emotions Expressed"),
                width="stretch",
            )
        if "confidence" in df_ana.columns:
            fig_hist = px.histogram(
                df_ana, x="confidence", nbins=15,
                title="LLM Confidence Distribution", template="plotly_dark",
            )
            fig_hist.update_traces(marker_color="#1DB954")
            fig_hist.update_layout(
                xaxis_title="Confidence", yaxis_title="Count", height=350,
            )
            c6.plotly_chart(fig_hist, width="stretch")
    else:
        st.info("No analysis results yet.  Run insight extraction above.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 – THEME SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Theme Summary":
    st.title("Theme Summary")
    st.markdown(
        "Clusters the structured labels into a human-readable executive report."
    )

    min_conf = st.slider(
        "Min confidence to include", 0.0, 1.0,
        config.DEFAULT_MIN_CONFIDENCE_SUMMARY, 0.05,
    )
    top_n = st.number_input("Top N themes per section", 1, 50, config.DEFAULT_TOP_N_THEMES)

    if st.button("Generate Report", width="stretch", type="primary"):
        if df_ana is None:
            st.error("Run insight extraction first.")
        else:
            t0 = time.perf_counter()
            progress = st.progress(0.0)
            stage_status = st.empty()
            logs_exp = st.expander("Execution Logs", expanded=False)
            with logs_exp:
                log_ph = st.empty()

            with capture_logs(log_ph) as log_capture:
                try:
                    stage_status.info("Clustering themes…")
                    progress.progress(0.4)
                    run_stage_summary(float(min_conf), int(top_n))
                    progress.progress(1.0, text=f"Done ({elapsed(t0)})")
                    stage_status.success(f"Report generated in {elapsed(t0)}.")
                    st.rerun()
                except Exception as exc:
                    handle_runtime_error(
                        exc,
                        stage_status=stage_status,
                        progress=progress,
                        log_capture=log_capture,
                        message="Report generation failed",
                    )

    if config.THEME_SUMMARY_MD.exists():
        st.subheader("Report Preview")
        try:
            md_text = config.THEME_SUMMARY_MD.read_text(encoding="utf-8")
            st.markdown(md_text)
            st.download_button(
                "Download theme_summary.md",
                md_text.encode("utf-8"),
                "theme_summary.md",
                "text/markdown",
            )
        except Exception as exc:
            st.error(f"Could not read report: {exc}")
    else:
        st.info("No report generated yet.  Click 'Generate Report' above.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 – OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
elif navigation_page == "Outputs":
    st.title("Outputs & Artifact Center")
    st.markdown("Download every pipeline artifact as a report card.")

    artifacts = [
        ("Raw Reviews", config.RAW_REVIEWS_CSV),
        ("Filtered Reviews", config.FILTERED_REVIEWS_CSV),
        ("Rejected Reviews", config.REJECTED_REVIEWS_CSV),
        ("Reddit Cache", config.REDDIT_CACHE_CSV),
        ("Filter Summary (JSON)", config.FILTER_SUMMARY_JSON),
        ("Analyzed Reviews", config.ANALYZED_REVIEWS_CSV),
        ("Theme Summary Report", config.THEME_SUMMARY_MD),
    ]

    left, right = st.columns(2)

    for idx, (label, path) in enumerate(artifacts):
        col = left if idx % 2 == 0 else right
        with col:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                ts = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime(
                    "%Y-%m-%d  %H:%M"
                )
                mime = (
                    "text/csv" if path.suffix == ".csv"
                    else "text/markdown" if path.suffix == ".md"
                    else "application/json"
                )
                st.markdown(
                    f"""<div style="background:#181818;padding:18px 20px;
                        border-radius:8px;border:1px solid #282828;
                        margin:12px 0 4px 0;">
                      <h4 style="color:#1DB954;margin:0 0 10px 0;font-weight:normal;">
                        {label}</h4>
                      <p style="margin:0;color:#b3b3b3;font-size:.82em;">
                        <code>{path.name}</code></p>
                      <p style="margin:3px 0;color:#b3b3b3;font-size:.82em;">
                        {size_kb:.1f} KB &nbsp;·&nbsp; {ts}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.download_button(
                    f"Download {path.name}",
                    path.read_bytes(),
                    path.name,
                    mime,
                    key=f"dl_{idx}",
                    width="stretch",
                )
            else:
                st.markdown(
                    f"""<div style="background:#181818;padding:18px 20px;
                        border-radius:8px;border:1px dashed #333;
                        margin:12px 0 18px 0;">
                      <h4 style="color:#555;margin:0 0 6px 0;font-weight:normal;">
                        {label}</h4>
                      <p style="margin:0;color:#555;font-size:.82em;">
                        Not generated yet.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

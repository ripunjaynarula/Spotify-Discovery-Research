from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from analysis.analyze_reviews import analyze_reviews, write_analyzed_reviews
from analysis.filter_reviews import (
    build_rejected_reviews,
    classify_relevance,
    deterministic_pre_filter,
    filter_relevant_reviews,
    prepare_reviews_for_llm,
    read_raw_reviews,
    write_filtered_reviews,
    write_rejected_reviews,
)
from analysis.theme_summary import build_theme_summary, read_analyzed_reviews, write_theme_summary
from reviews.google_play import collect_google_play_reviews
from reviews.reddit import collect_reddit_reviews
from reviews.spotify_community import collect_spotify_community_reviews
from reviews.utils import write_raw_reviews_csv

# Configure Streamlit page layout
st.set_page_config(
    page_title="Spotify Discovery Research PM Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom stdout redirector to capture python print statements in real-time
class StreamToStreamlit:
    def __init__(self, placeholder, warning_placeholder=None):
        self.placeholder = placeholder
        self.warning_placeholder = warning_placeholder
        self.buffer = io.StringIO()

    def write(self, text):
        self.buffer.write(text)
        self.placeholder.code(self.buffer.getvalue())
        
        # Intercept and show user-friendly notifications for retries
        if self.warning_placeholder and text.strip():
            if "LLM returned no result for review ids" in text:
                try:
                    ids_part = text.split("review ids:")[-1].strip()
                    id_count = len([x for x in ids_part.split(",") if x.strip()])
                    if id_count > 0:
                        self.warning_placeholder.warning(
                            f"Retrying {id_count} review(s) because the AI returned an incomplete response."
                        )
                except Exception:
                    self.warning_placeholder.warning(
                        "Retrying reviews because the AI returned an incomplete response."
                    )
            elif "Retrying missing review ids only in" in text:
                try:
                    delay = text.split("only in")[-1].split("seconds")[0].strip()
                    self.warning_placeholder.info(f"Retrying in {delay} seconds...")
                except Exception:
                    pass

    def flush(self):
        pass


@contextlib.contextmanager
def st_redirect_stdout(placeholder, warning_placeholder=None):
    old_stdout = sys.stdout
    sys.stdout = StreamToStreamlit(placeholder, warning_placeholder)
    try:
        yield
    finally:
        sys.stdout = old_stdout


# Check and display API key warnings
def check_api_key():
    if not config.OPENROUTER_API_KEY:
        st.sidebar.warning(
            "OPENROUTER_API_KEY is not configured. LLM analysis steps (Filter, "
            "Analyze) will fail. Please set it in secrets or environment."
        )
        return False
    return True


check_api_key()

# Sidebar Navigation
st.sidebar.title("Spotify Discovery Analytics")
st.sidebar.markdown("AI feedback analysis pipeline for Product Managers.")

navigation_page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Collect Reviews",
        "Filter Reviews",
        "Analyze Reviews",
        "Theme Summary",
        "Outputs",
    ],
)

st.sidebar.markdown("---")

# Global "Run Complete Pipeline" Orchestrator in Sidebar
st.sidebar.subheader("One-Click Pipeline")
if st.sidebar.button("Run Complete Pipeline", use_container_width=True):
    status_container = st.status("Initializing workflow...", expanded=True)
    warning_placeholder = st.empty()
    logs_expander = st.expander("Execution Logs", expanded=False)
    
    with logs_expander:
        logs_placeholder = st.empty()
        with st_redirect_stdout(logs_placeholder, warning_placeholder):
            try:
                # Stage 1: Collection
                status_container.update(label="Stage 1/4: Collecting raw reviews...", state="running")
                print("=========================================")
                print("STAGE 1: COLLECTING REVIEWS")
                print("=========================================")
                collected = []
                collected.extend(
                    collect_google_play_reviews(
                        limit=config.DEFAULT_LIMIT,
                        country=config.GOOGLE_PLAY_COUNTRY,
                        language=config.GOOGLE_PLAY_LANGUAGE,
                    )
                )
                collected.extend(collect_reddit_reviews(limit=config.DEFAULT_LIMIT))
                collected.extend(collect_spotify_community_reviews(limit=config.DEFAULT_LIMIT))
                write_raw_reviews_csv(collected, config.RAW_REVIEWS_CSV)
                print(f"Collected and saved {len(collected)} raw reviews.")

                # Stage 2: Relevance Filtering
                status_container.update(label="Stage 2/4: Filtering reviews for relevance...", state="running")
                print("\n=========================================")
                print("STAGE 2: RELEVANCE FILTERING")
                print("=========================================")
                raw_reviews = read_raw_reviews(config.RAW_REVIEWS_CSV)
                reviews_for_llm, removal_counts = prepare_reviews_for_llm(
                    raw_reviews,
                    min_review_length=config.DEFAULT_MIN_REVIEW_LENGTH,
                )
                
                class relevance_args:
                    input = config.RAW_REVIEWS_CSV
                    output = config.FILTERED_REVIEWS_CSV
                    batch_size = config.DEFAULT_BATCH_SIZE_FILTER
                    model = config.OPENROUTER_MODEL
                    max_retries = config.DEFAULT_MAX_RETRIES
                    retry_delay_seconds = config.DEFAULT_RETRY_DELAY_SECONDS
                    min_review_length = config.DEFAULT_MIN_REVIEW_LENGTH
                    min_confidence = 0.0

                relevance_rows = classify_relevance(reviews_for_llm, relevance_args())
                filtered_reviews = filter_relevant_reviews(
                    reviews_for_llm,
                    relevance_rows,
                    min_confidence=0.0,
                )
                rejected_reviews = build_rejected_reviews(reviews_for_llm, relevance_rows)
                
                write_filtered_reviews(filtered_reviews, config.FILTERED_REVIEWS_CSV)
                write_rejected_reviews(rejected_reviews, config.REJECTED_REVIEWS_CSV)
                
                confidence_vals = [float(row["confidence"]) for row in relevance_rows]
                irrelevant_count = sum(1 for row in relevance_rows if row["relevant"] is not True)
                summary_data = {
                    "total_reviews": len(raw_reviews),
                    "reviews_sent_to_llm": len(reviews_for_llm),
                    "relevant_reviews": len(filtered_reviews),
                    "irrelevant_reviews": irrelevant_count,
                    "average_confidence": sum(confidence_vals) / len(confidence_vals) if confidence_vals else 0.0,
                    "model": config.OPENROUTER_MODEL,
                    "processing_time_seconds": 1.0,
                }
                config.FILTER_SUMMARY_JSON.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
                print(f"Filtered out {irrelevant_count} irrelevant reviews. {len(filtered_reviews)} passed.")

                # Stage 3: Insight Extraction
                status_container.update(label="Stage 3/4: Extracting product insights...", state="running")
                print("\n=========================================")
                print("STAGE 3: INSIGHT EXTRACTION")
                print("=========================================")
                dataframe = pd.read_csv(config.FILTERED_REVIEWS_CSV, dtype={"id": str})
                dataframe = dataframe[dataframe["review"].astype(str).str.strip() != ""].copy()
                
                class analyze_args:
                    input = config.FILTERED_REVIEWS_CSV
                    output = config.ANALYZED_REVIEWS_CSV
                    batch_size = config.DEFAULT_BATCH_SIZE_ANALYZE
                    model = config.OPENROUTER_MODEL
                    max_retries = config.DEFAULT_MAX_RETRIES
                    retry_delay_seconds = config.DEFAULT_RETRY_DELAY_SECONDS
                    continue_on_error = True

                analyzed_rows = analyze_reviews(dataframe, analyze_args())
                write_analyzed_reviews(analyzed_rows, config.ANALYZED_REVIEWS_CSV)
                print(f"Extracted insights from {len(analyzed_rows)} reviews.")

                # Stage 4: Theme Summary
                status_container.update(label="Stage 4/4: Generating cluster summaries...", state="running")
                print("\n=========================================")
                print("STAGE 4: THEME SUMMARY GENERATION")
                print("=========================================")
                analyzed_reviews = read_analyzed_reviews(config.ANALYZED_REVIEWS_CSV, config.DEFAULT_MIN_CONFIDENCE_SUMMARY)
                markdown = build_theme_summary(analyzed_reviews, config.DEFAULT_TOP_N_THEMES)
                write_theme_summary(markdown, config.THEME_SUMMARY_MD)
                
                status_container.update(label="Workflow completed successfully!", state="complete", expanded=False)
                st.sidebar.success("Pipeline executed successfully!")
                st.rerun()
            except Exception as exc:
                status_container.update(label="Workflow failed", state="error")
                st.sidebar.error(f"Pipeline failed: {exc}")

st.sidebar.markdown("---")
st.sidebar.caption(f"Default Model: {config.OPENROUTER_MODEL}")


# Helper: Load dataset details safely
def load_csv_safely(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


df_raw = load_csv_safely(config.RAW_REVIEWS_CSV)
df_filt = load_csv_safely(config.FILTERED_REVIEWS_CSV)
df_ana = load_csv_safely(config.ANALYZED_REVIEWS_CSV)

pipeline_has_run = df_raw is not None and len(df_raw) > 0


# Onboarding Empty State Panel
def display_onboarding_instructions():
    st.markdown("""
    <div style="background-color: #181818; padding: 30px; border-radius: 10px; border: 1px solid #282828; margin-top:20px;">
        <h3 style="color: #1DB954; margin-top: 0; font-weight: normal;">Pipeline Initialization Onboarding</h3>
        <p style="color: #b3b3b3; line-height: 1.6;">
            Welcome to the Spotify Discovery Research analytics portal. No datasets have been generated yet.
        </p>
        <p style="color: #b3b3b3; line-height: 1.6;">
            To initialize the dashboard and view customer feedback insights:
        </p>
        <ol style="color: #b3b3b3; line-height: 1.8;">
            <li>Ensure you have set your <b>OPENROUTER_API_KEY</b>.</li>
            <li>Click the green <b>"Run Complete Pipeline"</b> button on the sidebar to execute the collection, filtering, analysis, and report generation in one click.</li>
            <li>Alternatively, navigate through the steps sequentially using the sidebar links to run components individually.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 1: DASHBOARD
# ==========================================
if navigation_page == "Dashboard":
    st.title("Spotify Music Discovery Analytics")
    st.markdown(
        "Executive product intelligence and telemetry derived from Spotify recommendations feedback."
    )

    if not pipeline_has_run:
        display_onboarding_instructions()
    else:
        # Dynamic Executive Insights calculation
        top_pain_point = "unknown"
        top_root_cause = "unknown"
        top_segment = "unknown"
        top_surface = "unknown"
        opportunity_statement = "Run full insights analysis to formulate recommendation opportunities."

        if df_ana is not None and len(df_ana) > 0:
            def get_mode_value(df, column):
                valid_series = df[~df[column].astype(str).str.lower().isin(config.IGNORE_VALUES)][column]
                return str(valid_series.mode().iloc[0]) if not valid_series.empty else "unknown"

            top_pain_point = get_mode_value(df_ana, "pain_point")
            top_root_cause = get_mode_value(df_ana, "root_cause")
            top_segment = get_mode_value(df_ana, "user_segment")
            top_surface = get_mode_value(df_ana, "discovery_surface")

            if top_pain_point != "unknown" and top_root_cause != "unknown" and top_segment != "unknown":
                opportunity_statement = (
                    f"Address **{top_root_cause}** on the **{top_surface}** surface "
                    f"to resolve **{top_pain_point}** for **{top_segment}** users. "
                    f"This represents our highest leverage product optimization vector."
                )

        # 1. Executive Insights Cards Grid
        st.subheader("Executive Product Insights")
        
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 25px;">
            <div style="background-color: #181818; padding: 20px; border-radius: 8px; border-left: 4px solid #1DB954;">
                <p style="color: #b3b3b3; margin: 0; font-size: 0.85em; text-transform: uppercase;">Top Pain Point</p>
                <h4 style="color: #FFFFFF; margin: 5px 0 0 0; font-size: 1.15em; font-weight: 500;">{top_pain_point}</h4>
            </div>
            <div style="background-color: #181818; padding: 20px; border-radius: 8px; border-left: 4px solid #1DB954;">
                <p style="color: #b3b3b3; margin: 0; font-size: 0.85em; text-transform: uppercase;">Top Root Cause</p>
                <h4 style="color: #FFFFFF; margin: 5px 0 0 0; font-size: 1.15em; font-weight: 500;">{top_root_cause}</h4>
            </div>
            <div style="background-color: #181818; padding: 20px; border-radius: 8px; border-left: 4px solid #1DB954;">
                <p style="color: #b3b3b3; margin: 0; font-size: 0.85em; text-transform: uppercase;">Critical Segment</p>
                <h4 style="color: #FFFFFF; margin: 5px 0 0 0; font-size: 1.15em; font-weight: 500;">{top_segment}</h4>
            </div>
            <div style="background-color: #181818; padding: 20px; border-radius: 8px; border-left: 4px solid #1DB954;">
                <p style="color: #b3b3b3; margin: 0; font-size: 0.85em; text-transform: uppercase;">Primary Surface</p>
                <h4 style="color: #FFFFFF; margin: 5px 0 0 0; font-size: 1.15em; font-weight: 500;">{top_surface}</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color: #181818; padding: 20px; border-radius: 8px; border-top: 3px solid #1DB954; margin-bottom: 30px;">
            <p style="color: #b3b3b3; margin: 0; font-size: 0.9em; text-transform: uppercase;">Largest Product Opportunity</p>
            <p style="color: #FFFFFF; margin: 8px 0 0 0; font-size: 1.1em; line-height: 1.5;">{opportunity_statement}</p>
        </div>
        """, unsafe_allow_html=True)

        # 2. Dynamic Stage Counts calculations
        raw_count = len(df_raw) if df_raw is not None else 0
        filtered_count = len(df_filt) if df_filt is not None else 0
        analyzed_count = len(df_ana) if df_ana is not None else 0
        pre_filtered_count = raw_count
        
        if config.FILTER_SUMMARY_JSON.exists():
            try:
                summary = json.loads(config.FILTER_SUMMARY_JSON.read_text(encoding="utf-8"))
                pre_filtered_count = summary.get("reviews_sent_to_llm", raw_count)
            except Exception:
                pass

        st.subheader("Pipeline Stage Volumes")
        
        # Funnel Stage Flow Diagram
        fig_funnel = go.Figure(go.Funnel(
            y=["Raw Collection", "Passed Pre-filter", "Relevance Filtered", "Analyzed Insights"],
            x=[raw_count, pre_filtered_count, filtered_count, analyzed_count],
            textinfo="value+percent initial",
            marker=dict(color=["#282828", "#121212", "#1DB954", "#1aa34a"])
        ))
        fig_funnel.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20, l=150, r=150))
        st.plotly_chart(fig_funnel, use_container_width=True)

        # Workflow Flowchart Annotated with Latest Run Counts
        st.subheader("Annotated Process Diagram")
        dot = f"""
        digraph G {{
            rankdir=LR;
            node [shape=box, style="filled,rounded", color="#1DB954", fillcolor="#1E1E1E", fontcolor="#FFFFFF", fontname="Helvetica", size="0.5"];
            edge [color="#1DB954"];
            
            Coll [label="Raw Collection\\n({raw_count} reviews)"];
            RF [label="Relevance Filter\\n({filtered_count} passed)"];
            IE [label="Insight Extraction\\n({analyzed_count} analyzed)"];
            TS [label="Theme Summary\\n(theme_summary.md)"];
            
            Coll -> RF;
            RF -> IE;
            IE -> TS;
        }}
        """
        st.graphviz_chart(dot)


# ==========================================
# PAGE 2: COLLECT REVIEWS
# ==========================================
elif navigation_page == "Collect Reviews":
    st.title("Review Collection")
    st.markdown("Download feedback strings directly from Spotify social/store listings.")

    sources = st.multiselect(
        "Sources",
        options=["google_play", "reddit", "spotify_community"],
        default=["google_play"],
    )

    limit = st.slider("Limit per Source", min_value=10, max_value=500, value=config.DEFAULT_LIMIT)
    spotify_community_url = st.text_input(
        "Spotify Community Board URL (Optional)",
        placeholder="https://community.spotify.com/...",
    )

    if st.button("Run Collection", use_container_width=True):
        if not sources:
            st.error("Select at least one review source.")
        else:
            status_container = st.status("Initializing collectors...", expanded=True)
            logs_expander = st.expander("Execution Logs", expanded=False)
            
            with logs_expander:
                logs_placeholder = st.empty()
                with st_redirect_stdout(logs_placeholder):
                    try:
                        collected_reviews = []
                        progress_bar = st.progress(0.0)
                        
                        if "google_play" in sources:
                            status_container.update(label="Stage 1: Downloading Google Play Store feedback...", state="running")
                            progress_bar.progress(0.2)
                            collected_reviews.extend(
                                collect_google_play_reviews(
                                    limit=limit,
                                    country=config.GOOGLE_PLAY_COUNTRY,
                                    language=config.GOOGLE_PLAY_LANGUAGE,
                                )
                            )
                            
                        if "reddit" in sources:
                            status_container.update(label="Stage 2: Scanning public Reddit recommendation threads...", state="running")
                            progress_bar.progress(0.5)
                            collected_reviews.extend(collect_reddit_reviews(limit=limit))
                            
                        if "spotify_community" in sources:
                            status_container.update(label="Stage 3: Scraping Spotify Community forum topics...", state="running")
                            progress_bar.progress(0.8)
                            community_kwargs = {"limit": limit}
                            if spotify_community_url:
                                community_kwargs["search_url"] = spotify_community_url
                            collected_reviews.extend(collect_spotify_community_reviews(**community_kwargs))

                        progress_bar.progress(0.9)
                        status_container.update(label="Stage 4: Deduplicating and packaging strings...", state="running")
                        write_raw_reviews_csv(collected_reviews, config.RAW_REVIEWS_CSV)
                        
                        progress_bar.progress(1.0)
                        status_container.update(label=f"Completed! Downloaded {len(collected_reviews)} reviews.", state="complete", expanded=False)
                        st.success("Review collection completed successfully!")
                        st.rerun()
                    except Exception as e:
                        status_container.update(label="Collection failed", state="error")
                        st.error(f"Error executing collection: {e}")

    # Preview and download
    if df_raw is not None:
        st.subheader("Raw Feedback Dataset Preview")
        st.dataframe(df_raw, use_container_width=True)
        
        # Donut Chart of Source Proportions
        st.subheader("Feedback Source Breakdown")
        df_source = df_raw["source"].value_counts().reset_index()
        df_source.columns = ["source", "count"]
        fig_donut = px.pie(
            df_source, values="count", names="source", hole=0.5,
            template="plotly_dark", color_discrete_sequence=["#1DB954", "#282828", "#1aa34a"]
        )
        fig_donut.update_layout(margin=dict(t=40, b=40, l=40, r=40), height=300)
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No raw reviews collected yet. Click 'Run Collection' to begin.")


# ==========================================
# PAGE 3: FILTER REVIEWS
# ==========================================
elif navigation_page == "Filter Reviews":
    st.title("Relevance Filtering")
    st.markdown("Isolate music discovery and playlist recommendation topics.")

    col1, col2 = st.columns(2)
    batch_size = col1.number_input("Filter Batch Size", min_value=1, value=config.DEFAULT_BATCH_SIZE_FILTER)
    min_review_length = col2.number_input("Minimum Character Length", min_value=1, value=config.DEFAULT_MIN_REVIEW_LENGTH)
    min_confidence = st.slider("Min Relevance Confidence", min_value=0.0, max_value=1.0, value=0.0)

    if st.button("Run Relevance Filtering", use_container_width=True):
        if df_raw is None:
            st.error("Collect raw reviews first before filtering.")
        else:
            status_container = st.status("Preprocessing raw reviews...", expanded=True)
            warning_placeholder = st.empty()
            logs_expander = st.expander("Execution Logs", expanded=False)
            
            with logs_expander:
                logs_placeholder = st.empty()
                with st_redirect_stdout(logs_placeholder, warning_placeholder):
                    try:
                        progress_bar = st.progress(0.0)
                        
                        class Args:
                            input = config.RAW_REVIEWS_CSV
                            output = config.FILTERED_REVIEWS_CSV
                            batch_size = int(batch_size)
                            model = config.OPENROUTER_MODEL
                            max_retries = config.DEFAULT_MAX_RETRIES
                            retry_delay_seconds = config.DEFAULT_RETRY_DELAY_SECONDS
                            min_review_length = int(min_review_length)
                            min_confidence = float(min_confidence)

                        run_args = Args()
                        raw_reviews = read_raw_reviews(run_args.input)
                        progress_bar.progress(0.2)
                        
                        status_container.update(label="Stage 1/3: Applying deterministic keyword filters...", state="running")
                        reviews_for_llm, removal_counts = prepare_reviews_for_llm(
                            raw_reviews,
                            min_review_length=run_args.min_review_length,
                        )
                        progress_bar.progress(0.4)
                        
                        status_container.update(label="Stage 2/3: Executing LLM relevance assessment batches...", state="running")
                        relevance_rows = classify_relevance(reviews_for_llm, run_args)
                        progress_bar.progress(0.8)
                        
                        status_container.update(label="Stage 3/3: Exporting output filtered/rejected sets...", state="running")
                        filtered_reviews = filter_relevant_reviews(
                            reviews_for_llm,
                            relevance_rows,
                            min_confidence=run_args.min_confidence,
                        )
                        rejected_reviews = build_rejected_reviews(reviews_for_llm, relevance_rows)
                        
                        write_filtered_reviews(filtered_reviews, run_args.output)
                        write_rejected_reviews(rejected_reviews, config.REJECTED_REVIEWS_CSV)
                        
                        confidence_vals = [float(row["confidence"]) for row in relevance_rows]
                        irrelevant_count = sum(1 for row in relevance_rows if row["relevant"] is not True)
                        summary_data = {
                            "total_reviews": len(raw_reviews),
                            "reviews_sent_to_llm": len(reviews_for_llm),
                            "relevant_reviews": len(filtered_reviews),
                            "irrelevant_reviews": irrelevant_count,
                            "average_confidence": sum(confidence_vals) / len(confidence_vals) if confidence_vals else 0.0,
                            "model": config.OPENROUTER_MODEL,
                            "processing_time_seconds": 1.0,
                        }
                        config.FILTER_SUMMARY_JSON.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

                        progress_bar.progress(1.0)
                        status_container.update(label=f"Done! Extracted {len(filtered_reviews)} relevant posts.", state="complete", expanded=False)
                        st.success("Relevance filtering finished!")
                        st.rerun()
                    except Exception as e:
                        status_container.update(label="Relevance filtering failed", state="error")
                        st.error(f"Error filtering reviews: {e}")

    # Preview and metrics
    if df_filt is not None:
        st.subheader("Filtered Dataset Preview")
        st.dataframe(df_filt, use_container_width=True)
        
        # Relevance vs Irrelevant Donut Chart
        df_rej = load_csv_safely(config.REJECTED_REVIEWS_CSV)
        rej_count = len(df_rej) if df_rej is not None else 0
        filt_count = len(df_filt)
        
        st.subheader("Feedback Relevance Ratio")
        df_ratio = pd.DataFrame({
            "Classification": ["Relevant", "Irrelevant"],
            "Count": [filt_count, rej_count]
        })
        fig_ratio = px.pie(
            df_ratio, values="Count", names="Classification", hole=0.5,
            template="plotly_dark", color_discrete_sequence=["#1DB954", "#282828"]
        )
        fig_ratio.update_layout(margin=dict(t=40, b=40, l=40, r=40), height=300)
        st.plotly_chart(fig_ratio, use_container_width=True)
    else:
        st.info("No filtered reviews dataset found. Run Relevance Filtering to build.")


# ==========================================
# PAGE 4: ANALYZE REVIEWS
# ==========================================
elif navigation_page == "Analyze Reviews":
    st.title("Insight Extraction")
    st.markdown("Extract structured feedback classification and sentiment telemetry.")

    col1, col2 = st.columns(2)
    batch_size = col1.number_input("Analysis Batch Size", min_value=1, value=config.DEFAULT_BATCH_SIZE_ANALYZE)
    continue_on_error = col2.checkbox("Ignore Analysis Errors", value=True)

    if st.button("Run Insight Extraction", use_container_width=True):
        if df_filt is None:
            st.error("Run Relevance Filtering before running insight extraction.")
        else:
            status_container = st.status("Loading filtered data...", expanded=True)
            warning_placeholder = st.empty()
            logs_expander = st.expander("Execution Logs", expanded=False)
            
            with logs_expander:
                logs_placeholder = st.empty()
                with st_redirect_stdout(logs_placeholder, warning_placeholder):
                    try:
                        progress_bar = st.progress(0.0)
                        
                        class Args:
                            input = config.FILTERED_REVIEWS_CSV
                            output = config.ANALYZED_REVIEWS_CSV
                            batch_size = int(batch_size)
                            model = config.OPENROUTER_MODEL
                            max_retries = config.DEFAULT_MAX_RETRIES
                            retry_delay_seconds = config.DEFAULT_RETRY_DELAY_SECONDS
                            continue_on_error = continue_on_error

                        run_args = Args()
                        dataframe = pd.read_csv(run_args.input, dtype={"id": str})
                        dataframe = dataframe[dataframe["review"].astype(str).str.strip() != ""].copy()
                        progress_bar.progress(0.2)
                        
                        status_container.update(label="Executing LLM classification batches...", state="running")
                        analyzed_rows = analyze_reviews(dataframe, run_args)
                        progress_bar.progress(0.8)
                        
                        status_container.update(label="Writing structured analysis CSV...", state="running")
                        write_analyzed_reviews(analyzed_rows, run_args.output)
                        
                        progress_bar.progress(1.0)
                        status_container.update(label="Analysis completed!", state="complete", expanded=False)
                        st.success("Insight extraction finished!")
                        st.rerun()
                    except Exception as e:
                        status_container.update(label="Analysis failed", state="error")
                        st.error(f"Error during extraction: {e}")

    # Preview and Charts
    if df_ana is not None:
        st.subheader("Analyzed Insights Preview")
        st.dataframe(df_ana, use_container_width=True)

        st.markdown("---")
        st.subheader("Product Feedback Visualizations")

        # Custom plotting functions with ignores applied
        def get_clean_counts(df, col):
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            return counts[~counts[col].astype(str).str.lower().isin(config.IGNORE_VALUES)]

        def plot_bar(df, col, title):
            clean_df = get_clean_counts(df, col)
            clean_df = clean_df.sort_values(by="count", ascending=True)
            fig = px.bar(clean_df, x="count", y=col, orientation="h", title=title, template="plotly_dark")
            fig.update_traces(marker_color="#1DB954")
            fig.update_layout(xaxis_title="Count", yaxis_title=None, height=350, margin=dict(l=150, r=20, t=40, b=40))
            return fig

        def plot_donut(df, col, title):
            clean_df = get_clean_counts(df, col)
            fig = px.pie(clean_df, values="count", names=col, hole=0.5, title=title, template="plotly_dark",
                         color_discrete_sequence=px.colors.sequential.Greens_r)
            fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), height=350)
            return fig

        col1, col2 = st.columns(2)
        
        if "root_cause" in df_ana.columns:
            col1.plotly_chart(plot_bar(df_ana, "root_cause", "Root Causes of Discovery Issues"), use_container_width=True)
            
        if "pain_point" in df_ana.columns:
            col2.plotly_chart(plot_bar(df_ana, "pain_point", "Common Discovery Pain Points"), use_container_width=True)

        col3, col4 = st.columns(2)
        
        if "discovery_surface" in df_ana.columns:
            col3.plotly_chart(plot_bar(df_ana, "discovery_surface", "Discovery Surfaces Mentioned"), use_container_width=True)
            
        if "user_segment" in df_ana.columns:
            st_counts = get_clean_counts(df_ana, "user_segment")
            fig_tree = px.treemap(
                st_counts, path=["user_segment"], values="count",
                title="User Segments Distribution", template="plotly_dark",
                color_discrete_sequence=["#1DB954", "#282828", "#1aa34a"]
            )
            fig_tree.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
            col4.plotly_chart(fig_tree, use_container_width=True)

        col5, col6 = st.columns(2)
        
        if "emotion" in df_ana.columns:
            col5.plotly_chart(plot_donut(df_ana, "emotion", "Emotions Expressed"), use_container_width=True)

        if "confidence" in df_ana.columns:
            fig_hist = px.histogram(df_ana, x="confidence", nbins=15, title="LLM Confidence Distribution", template="plotly_dark")
            fig_hist.update_traces(marker_color="#1DB954")
            fig_hist.update_layout(xaxis_title="Confidence", yaxis_title="Count", height=350)
            col6.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No analyzed insights dataset found. Run Insight Extraction to generate.")


# ==========================================
# PAGE 5: THEME SUMMARY
# ==========================================
elif navigation_page == "Theme Summary":
    st.title("Theme Summary Generation")
    st.markdown("Consolidate extracted structured feedback into an executive markdown report.")

    min_confidence = st.slider("Report Confidence Cutoff", min_value=0.0, max_value=1.0, value=config.DEFAULT_MIN_CONFIDENCE_SUMMARY)
    top_n = st.number_input("Themes Top Bounds", min_value=1, value=config.DEFAULT_TOP_N_THEMES)

    if st.button("Generate Summary Report", use_container_width=True):
        if df_ana is None:
            st.error("Run Insight Extraction before building summary.")
        else:
            status_container = st.status("Reading analyzed datasets...", expanded=True)
            logs_expander = st.expander("Execution Logs", expanded=False)
            
            with logs_expander:
                logs_placeholder = st.empty()
                with st_redirect_stdout(logs_placeholder):
                    try:
                        progress_bar = st.progress(0.0)
                        analyzed_reviews = read_analyzed_reviews(config.ANALYZED_REVIEWS_CSV, min_confidence)
                        progress_bar.progress(0.4)
                        
                        status_container.update(label="Clustering and parsing report themes...", state="running")
                        markdown = build_theme_summary(analyzed_reviews, top_n)
                        progress_bar.progress(0.8)
                        
                        status_container.update(label="Writing final theme summary report...", state="running")
                        write_theme_summary(markdown, config.THEME_SUMMARY_MD)
                        
                        progress_bar.progress(1.0)
                        status_container.update(label="Complete!", state="complete", expanded=False)
                        st.success("Summary report generated successfully!")
                        st.rerun()
                    except Exception as e:
                        status_container.update(label="Generation failed", state="error")
                        st.error(f"Error compiling report: {e}")

    # Preview and download
    if config.THEME_SUMMARY_MD.exists():
        st.subheader("Markdown Report Preview")
        try:
            report_text = config.THEME_SUMMARY_MD.read_text(encoding="utf-8")
            st.markdown(report_text)
            st.download_button(
                label="Download theme_summary.md",
                data=report_text.encode("utf-8"),
                file_name="theme_summary.md",
                mime="text/markdown",
            )
        except Exception as e:
            st.error(f"Error reading report: {e}")
    else:
        st.info("No theme_summary.md has been generated yet. Click 'Generate Summary Report' to compile.")


# ==========================================
# PAGE 6: OUTPUTS
# ==========================================
elif navigation_page == "Outputs":
    st.title("Outputs & Artifact Center")
    st.markdown("Download raw datasets and summary analysis files.")

    files_list = [
        ("Raw Reviews", config.RAW_REVIEWS_CSV),
        ("Filtered Reviews", config.FILTERED_REVIEWS_CSV),
        ("Rejected Reviews", config.REJECTED_REVIEWS_CSV),
        ("Relevance Filter Summary", config.FILTER_SUMMARY_JSON),
        ("Analyzed Reviews Insights", config.ANALYZED_REVIEWS_CSV),
        ("Theme Summary Markdown Report", config.THEME_SUMMARY_MD),
    ]

    col1, col2 = st.columns(2)
    
    for idx, (label, path) in enumerate(files_list):
        target_col = col1 if idx % 2 == 0 else col2
        
        with target_col:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                mtime = path.stat().st_mtime
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                mime_type = "text/csv" if path.suffix == ".csv" else "text/markdown" if path.suffix == ".md" else "application/json"
                
                st.markdown(f"""
                <div style="background-color: #181818; padding: 20px; border-radius: 8px; border: 1px solid #282828; margin-top: 15px; margin-bottom: 5px;">
                    <h4 style="color: #1DB954; margin: 0 0 10px 0; font-weight: normal;">{label}</h4>
                    <p style="margin: 0; color: #b3b3b3; font-size: 0.85em;">Filename: <code>{path.name}</code></p>
                    <p style="margin: 3px 0; color: #b3b3b3; font-size: 0.85em;">Size: {size_kb:.2f} KB</p>
                    <p style="margin: 3px 0; color: #b3b3b3; font-size: 0.85em;">Generated: {mtime_str}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.download_button(
                    label=f"Download {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime_type,
                    key=f"dl_{path.name}_{idx}",
                    use_container_width=True
                )
            else:
                st.markdown(f"""
                <div style="background-color: #181818; padding: 20px; border-radius: 8px; border: 1px dashed #282828; margin-top: 15px; margin-bottom: 15px;">
                    <h4 style="color: #b3b3b3; margin: 0 0 10px 0; font-weight: normal;">{label}</h4>
                    <p style="margin: 0; color: #7f7f7f; font-size: 0.85em;">File not generated yet.</p>
                </div>
                """, unsafe_allow_html=True)

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


WORKSPACE_DIR = Path(__file__).parent.resolve()
DATA_DIR = WORKSPACE_DIR / "data"
OUTPUT_DIR = WORKSPACE_DIR / "output"

# Load .env if present for local development.
load_dotenv(dotenv_path=WORKSPACE_DIR / ".env", override=False)
_DOTENV_VALUES = dotenv_values(dotenv_path=WORKSPACE_DIR / ".env")


def get_env_var(name: str, default: str = "") -> str:
    """
    Priority:
    1. Streamlit Secrets (Cloud)
    2. .env file
    3. OS Environment Variables
    4. Default value
    """

    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError

        try:
            value = st.secrets.get(name)
            if value is not None:
                return str(value)
        except StreamlitSecretNotFoundError:
            pass

    except Exception:
        pass

    dotenv_value = _DOTENV_VALUES.get(name)
    if dotenv_value not in (None, ""):
        return str(dotenv_value)

    value = os.getenv(name)
    if value:
        return value

    return default


# Directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_REVIEWS_CSV = DATA_DIR / "raw_reviews.csv"
FILTERED_REVIEWS_CSV = DATA_DIR / "filtered_reviews.csv"
REJECTED_REVIEWS_CSV = DATA_DIR / "rejected_reviews.csv"
FILTER_SUMMARY_JSON = DATA_DIR / "filter_summary.json"
ANALYZED_REVIEWS_CSV = DATA_DIR / "analyzed_reviews.csv"
THEME_SUMMARY_MD = OUTPUT_DIR / "theme_summary.md"
REDDIT_CACHE_CSV = DATA_DIR / "reddit_cache.csv"

# LLM
OPENROUTER_API_KEY = get_env_var("OPENROUTER_API_KEY")
LLM_PROVIDER = get_env_var("LLM_PROVIDER", "openrouter")
OPENROUTER_MODEL = get_env_var("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# Defaults
DEFAULT_LIMIT = 100
DEFAULT_BATCH_SIZE_FILTER = 10
DEFAULT_BATCH_SIZE_ANALYZE = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_MIN_CONFIDENCE_SUMMARY = 0.7
DEFAULT_TOP_N_THEMES = 10
DEFAULT_MIN_REVIEW_LENGTH = 15

# Google Play
GOOGLE_PLAY_COUNTRY = get_env_var("GOOGLE_PLAY_COUNTRY", "us")
GOOGLE_PLAY_LANGUAGE = get_env_var("GOOGLE_PLAY_LANGUAGE", "en")

# Ignored Values for reporting
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
from __future__ import annotations

import json
import traceback
import time
from collections.abc import Callable
from json import JSONDecodeError
from typing import Any

import requests

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    LLM_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS_RELEVANCE,
    DEFAULT_MAX_TOKENS_ANALYSIS,
)

from analysis.schema import (
    ANALYSIS_FIELDS,
    SYSTEM_PROMPT,
    build_batch_prompt,
    empty_analysis,
    ROOT_CAUSES,
    DISCOVERY_SURFACES,
    USER_SEGMENTS,
    clamp_confidence,
)

_session: requests.Session | None = None

def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PROVIDER = "openrouter"


class AnalysisError(RuntimeError):
    pass


class LLMRequestError(AnalysisError):
    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def analyze_review_batch(
    reviews: list[dict[str, object]],
    model: str,
    max_retries: int,
    retry_delay_seconds: float,
) -> list[dict[str, object]]:
    def request_and_parse() -> list[dict[str, object]]:
        content = generate_json_content(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_batch_prompt(reviews),
            max_tokens=DEFAULT_MAX_TOKENS_ANALYSIS,
        )
        parsed = parse_json_response(content)
        return normalize_batch_response(parsed, reviews)

    return _with_retries(request_and_parse, max_retries, retry_delay_seconds)


def generate_json_content(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
) -> str:
    selected_model = model or OPENROUTER_MODEL or DEFAULT_MODEL
    selected_max_tokens = max_tokens or DEFAULT_MAX_TOKENS_ANALYSIS
    session = _get_session()
    response = session.post(
        OPENROUTER_URL,
        headers=_build_headers(),
        json={
            "model": selected_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0,
            "max_tokens": selected_max_tokens,
            "response_format": {
                "type": "json_object",
            },
        },
        timeout=60,
    )
    _raise_for_response(response)
    content = _extract_message_content(response)
    if not content:
        raise AnalysisError("LLM provider returned an empty response.")
    return content


def parse_json_response(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except JSONDecodeError:
        parsed = json.loads(_extract_json_object(cleaned))

    if isinstance(parsed, list):
        return {"reviews": parsed}
    if not isinstance(parsed, dict):
        raise AnalysisError("LLM response was not a JSON object or array.")
    return parsed


def normalize_batch_response(
    parsed: dict[str, Any],
    input_reviews: list[dict[str, object]],
) -> list[dict[str, object]]:
    items = parsed.get("reviews", parsed.get("results", parsed.get("items")))
    if not isinstance(items, list):
        raise AnalysisError('LLM response must contain a "reviews" array.')

    by_id = {
        str(item.get("id")): item
        for item in items
        if isinstance(item, dict) and item.get("id") is not None
    }

    normalized: list[dict[str, object]] = []
    for input_review in input_reviews:
        review_id = str(input_review["id"])
        item = by_id.get(review_id, {})
        normalized.append({"id": review_id, **_normalize_analysis_item(item)})
    return normalized


def _normalize_analysis_item(item: dict[str, Any]) -> dict[str, object]:
    root_cause_map = {rc.lower(): rc for rc in ROOT_CAUSES}
    discovery_surface_map = {ds.lower(): ds for ds in DISCOVERY_SURFACES}
    user_segment_map = {us.lower(): us for us in USER_SEGMENTS}

    normalized = empty_analysis()
    for field in ANALYSIS_FIELDS:
        value = item.get(field, normalized[field])
        if field == "confidence":
            normalized[field] = clamp_confidence(value)
            continue

        val_str = str(value or "unknown").strip()
        val_lower = val_str.lower()

        if field == "root_cause":
            if val_lower in root_cause_map:
                normalized[field] = root_cause_map[val_lower]
            else:
                normalized[field] = "unknown"
        elif field == "discovery_surface":
            if val_lower in discovery_surface_map:
                normalized[field] = discovery_surface_map[val_lower]
            else:
                normalized[field] = "unknown"
        elif field == "user_segment":
            if val_lower in user_segment_map:
                normalized[field] = user_segment_map[val_lower]
            else:
                normalized[field] = "unknown"
        else:
            normalized[field] = val_str if val_str else "unknown"
    return normalized


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AnalysisError("Could not recover a JSON object from LLM response.")
    return content[start : end + 1]


def _with_retries(
    operation: Callable[[], list[dict[str, object]]],
    max_retries: int,
    retry_delay_seconds: float,
) -> list[dict[str, object]]:
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            print(
                f"[LLM attempt {attempt + 1}/{attempts} failed] "
                f"{type(exc).__name__}: {exc}\n"
                + traceback.format_exc()
            )
            if attempt == attempts - 1:
                break
            retry_after_seconds = getattr(last_error, "retry_after_seconds", None)
            delay_seconds = (
                retry_after_seconds
                if retry_after_seconds is not None
                else retry_delay_seconds * (2**attempt)
            )
            print(f"Retrying in {delay_seconds:.1f}s…")
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise AnalysisError("LLM analysis failed after all attempts without a captured error.")


def _build_headers() -> dict[str, str]:
    provider = (LLM_PROVIDER or DEFAULT_PROVIDER).casefold()
    if provider != DEFAULT_PROVIDER:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: '{provider}'. Only 'openrouter' is supported.")

    api_key = OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file (local) or Streamlit Secrets (cloud)."
        )

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _raise_for_response(response: requests.Response) -> None:
    if response.status_code < 400:
        return

    retry_after_seconds = _parse_retry_after(response.headers.get("Retry-After"))
    response_text = response.text.strip()
    request_url = getattr(response.request, "url", "") or "<unknown-url>"
    if response.status_code == 401:
        raise LLMRequestError(
            f"OpenRouter authentication failed with 401. URL: {request_url}. Response body: {response_text}",
            retry_after_seconds=retry_after_seconds,
        )
    if response.status_code == 402:
        raise LLMRequestError(
            "OpenRouter request exceeded the available token budget. "
            "Try reducing the batch size or increasing available OpenRouter credits. "
            f"URL: {request_url}. Response body: {response_text}",
            retry_after_seconds=retry_after_seconds,
        )
    if response.status_code == 429:
        raise LLMRequestError(
            f"OpenRouter rate limit exceeded with 429. URL: {request_url}. Response body: {response_text}",
            retry_after_seconds=retry_after_seconds,
        )
    if response.status_code == 500:
        raise LLMRequestError(
            f"OpenRouter server error 500. URL: {request_url}. Response body: {response_text}",
            retry_after_seconds=retry_after_seconds,
        )
    if response.status_code == 503:
        raise LLMRequestError(
            f"OpenRouter service unavailable with 503. URL: {request_url}. Response body: {response_text}",
            retry_after_seconds=retry_after_seconds,
        )
    raise LLMRequestError(
        f"OpenRouter request failed with status {response.status_code}. URL: {request_url}. Response body: {response_text}",
        retry_after_seconds=retry_after_seconds,
    )


def _extract_message_content(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError as exc:
        raise AnalysisError(
            f"OpenRouter returned non-JSON response. URL: {getattr(response.request, 'url', '<unknown-url>')}. Response body: {response.text.strip()}"
        ) from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AnalysisError("OpenRouter response did not include message content.") from exc
    return str(content or "")


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None

from __future__ import annotations


ANALYSIS_FIELDS = [
    "pain_point",
    "desired_outcome",
    "discovery_surface",
    "current_behaviour",
    "root_cause",
    "user_goal",
    "user_segment",
    "emotion",
    "confidence",
]

# Controlled Vocabularies
ROOT_CAUSES = [
    "Poor personalization",
    "Low recommendation diversity",
    "Repetitive recommendations",
    "Cold start",
    "Weak contextual understanding",
    "Limited user control",
    "Discovery hidden in UI",
    "Poor recommendation trust",
]

DISCOVERY_SURFACES = [
    "Discover Weekly",
    "Daily Mix",
    "AI DJ",
    "Radio",
    "Smart Shuffle",
    "Search",
    "Home Feed",
    "Playlist",
]

USER_SEGMENTS = [
    "Discover Weekly User",
    "AI DJ User",
    "Smart Shuffle User",
    "Radio User",
    "Playlist User",
    "Artist Explorer",
    "Casual Listener",
    "Heavy Listener",
    "Student",
    "Working Professional",
    "Free User",
    "Premium User",
]


SYSTEM_PROMPT = f"""
You classify Spotify user feedback for a Product Management research project.

Return JSON only. Do not include markdown, explanations, or prose.

For each input review, infer these fields using only the evidence in the review:
- pain_point: the specific discovery-related user problem, or "unknown"
- desired_outcome: what the user wants instead, or "unknown"
- discovery_surface: must be exactly one of the allowed discovery surfaces listed below, or "unknown"
- current_behaviour: observed/implied way the user currently listens or behaves, or "unknown"
- root_cause: must be exactly one of the allowed root causes listed below, or "unknown"
- user_goal: what the user is trying to achieve, or "unknown"
- user_segment: must be exactly one of the allowed user segments listed below, or "unknown"
- emotion: primary emotion expressed by the user (e.g. frustration, positive, annoyance), or "unknown"
- confidence: number from 0.0 to 1.0 reflecting evidence strength

Allowed Root Causes:
{", ".join(f"'{rc}'" for rc in ROOT_CAUSES)}

Allowed Discovery Surfaces:
{", ".join(f"'{ds}'" for ds in DISCOVERY_SURFACES)}

Allowed User Segments:
{", ".join(f"'{us}'" for us in USER_SEGMENTS)}

CRITICAL EXTRACTION CONSTRAINTS:
1. ALL labels for root_cause, discovery_surface, and user_segment MUST come ONLY from the allowed lists above.
2. NEVER invent, hallucinate, or construct root causes, discovery surfaces, or user segments. If the review content does not fit any of the allowed values, or you are uncertain, you MUST return "unknown".
3. Return "unknown" for any text fields (pain_point, desired_outcome, current_behaviour, user_goal, emotion) if they are missing or if you are uncertain.
4. For user_segment, select the most specific behavioural segment using this priority order:
   1. Discover Weekly User
   2. AI DJ User
   3. Smart Shuffle User
   4. Radio User
   5. Playlist User
   6. Artist Explorer
   7. Casual Listener
   8. Heavy Listener
   9. Student
   10. Working Professional
   Only classify as Premium User or Free User if subscription tier (free vs premium) is explicitly central to the review. Otherwise, select from the behavioural/demographic segments above, or return "unknown".
""".strip()


def build_batch_prompt(reviews: list[dict[str, object]]) -> str:
    return (
        "Classify each review. Return a JSON object with one key named "
        '"reviews". Its value must be an array with exactly one result per input. '
        "Each result must include the original id and the schema fields.\n\n"
        f"Input reviews:\n{reviews}"
    )


def empty_analysis() -> dict[str, object]:
    return {
        "pain_point": "unknown",
        "desired_outcome": "unknown",
        "discovery_surface": "unknown",
        "current_behaviour": "unknown",
        "root_cause": "unknown",
        "user_goal": "unknown",
        "user_segment": "unknown",
        "emotion": "unknown",
        "confidence": 0.0,
    }


def clamp_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))

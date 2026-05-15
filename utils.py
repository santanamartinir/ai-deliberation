"""
utils.py

This file contains small helper functions that are useful in several parts
of the project, such as:
- parsing JSON from model outputs
- cleaning payloads
- creating safe file names
- building recent conversation context
"""

import json
import re

"""
def parse_json_response(text):
    
    Try to read JSON from the model output.

    Sometimes the model returns perfect JSON.
    Sometimes it returns extra text around the JSON.
    This function tries both cases.
    
    text = text.strip()

    # First try: parse the full text directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second try: extract the JSON block from the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from model output:\n{text}")
"""

def parse_json_response(text):
    """
    Robust JSON parser for model outputs.

    1. Try direct parsing
    2. Try extracting JSON block
    3. If everything fails, return fallback instead of crashing
    """
    text = text.strip()

    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback (IMPORTANT: no crash)
    return {
        "error": "invalid_json",
        "raw_output": text
    }

def clean_payload(payload, fallback_stance):
    """
    Make sure the model output has valid values.

    If something is missing or invalid:
    - use a fallback stance
    - use default confidence
    """
    message = str(payload.get("message", "")).strip()
    stance = str(payload.get("stance", fallback_stance)).strip().lower()

    try:
        confidence = int(payload.get("confidence", 50))
    except Exception:
        confidence = 50

    if stance not in ["support", "mixed", "oppose"]:
        stance = fallback_stance

    confidence = max(0, min(100, confidence))

    return {
        "message": message,
        "stance": stance,
        "confidence": confidence,
    }


def slugify(text, max_len=50):
    """
    Turn a text into a filename-friendly slug.

    Example:
        "Should AI be used?" -> "should-ai-be-used"
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]


def build_recent_context(conversation_log, limit=6):
    """
    Build a short text block from the latest conversation messages.

    This is what agents can see from the recent history.
    """
    if not conversation_log:
        return "No previous messages."

    recent = conversation_log[-limit:]
    lines = []

    for item in recent:
        lines.append(f'{item["speaker"]}: {item["message"]}')

    return "\n".join(lines)
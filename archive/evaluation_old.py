"""
evaluation.py

This script reads one JSON experiment log created by deliberation_framework.py
and computes a few simple automatic metrics.

Why this is useful:
- You do not need to manually read every conversation
- You can compare experiments more objectively
- It gives you first basic indicators for:
    - repetition
    - vocabulary diversity
    - confidence
    - stance changes
    - agreement trend

Example usage:
    python evaluation.py --input .\logs\your_experiment_file.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# ============================================================
# 1. TERMINAL ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Read the input JSON file path from the terminal.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to experiment JSON log",
    )
    return parser.parse_args()


# ============================================================
# 2. BASIC TEXT HELPERS
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Split text into simple lowercase word tokens.
    This is a very basic tokenizer, but enough for simple metrics.
    """
    return re.findall(r"\b\w+\b", text.lower())


# ============================================================
# 3. METRICS
# ============================================================

def repeated_opening_ratio(messages: list[str], n_words: int = 3) -> float:
    """
    Measure how often messages start in the same way.

    Example:
    if many messages begin with "while i appreciate",
    the ratio will be higher.

    This is a simple indicator of repetitive style.
    """
    openings = []

    for msg in messages:
        tokens = tokenize(msg)
        if tokens:
            openings.append(" ".join(tokens[:n_words]))

    if not openings:
        return 0.0

    counts = Counter(openings)

    # Count how many opening patterns appear more than once
    repeated = sum(c for c in counts.values() if c > 1)

    return round(repeated / len(openings), 4)


def type_token_ratio(messages: list[str]) -> float:
    """
    A simple measure of vocabulary diversity.

    Formula:
        number of unique words / total number of words

    Higher usually means more lexical variety.
    """
    tokens = []

    for msg in messages:
        tokens.extend(tokenize(msg))

    if not tokens:
        return 0.0

    return round(len(set(tokens)) / len(tokens), 4)


def average_confidence(conversation_log: list[dict]) -> float:
    """
    Average the confidence values produced by the agents.
    """
    values = [int(x["confidence"]) for x in conversation_log if "confidence" in x]

    if not values:
        return 0.0

    return round(sum(values) / len(values), 2)


def stance_shift_count(conversation_log: list[dict]) -> int:
    """
    Count how many times agents change stance across the conversation.

    Example:
        support -> mixed = one shift
        mixed -> oppose = another shift
    """
    last_stance_by_speaker = {}
    shifts = 0

    for item in conversation_log:
        speaker = item["speaker"]
        stance = item["stance"]

        if speaker in last_stance_by_speaker and last_stance_by_speaker[speaker] != stance:
            shifts += 1

        last_stance_by_speaker[speaker] = stance

    return shifts


def final_round_metrics(round_summaries: list[dict]) -> dict:
    """
    Extract final agreement/disagreement values from the last round.
    """
    if not round_summaries:
        return {
            "final_agreement_pct": 0.0,
            "final_disagreement_pct": 0.0,
        }

    last = round_summaries[-1]
    return {
        "final_agreement_pct": last.get("agreement_pct", 0.0),
        "final_disagreement_pct": last.get("disagreement_pct", 0.0),
    }


def slope_over_rounds(values: list[float]) -> float:
    """
    Compute a very simple linear slope over rounds.

    Interpretation:
    - positive slope in agreement = agreement tends to increase
    - negative slope in disagreement = disagreement tends to decrease

    This is a simple trend indicator, not a complex statistical model.
    """
    if len(values) < 2:
        return 0.0

    x = list(range(len(values)))
    x_mean = sum(x) / len(x)
    y_mean = sum(values) / len(values)

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if denominator == 0:
        return 0.0

    return round(numerator / denominator, 4)


# ============================================================
# 4. MAIN PROGRAM
# ============================================================

def main() -> None:
    """
    Main function:
    - load one experiment JSON file
    - compute automatic metrics
    - save results to a new JSON file
    """
    args = parse_args()
    path = Path(args.input)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversation_log = data.get("conversation_log", [])
    round_summaries = data.get("round_summaries", [])

    messages = [x["message"] for x in conversation_log if "message" in x]
    agreement_series = [x.get("agreement_pct", 0.0) for x in round_summaries]
    disagreement_series = [x.get("disagreement_pct", 0.0) for x in round_summaries]

    metrics = {
        "config": data.get("config", {}),
        "num_messages": len(messages),
        "type_token_ratio": type_token_ratio(messages),
        "repeated_opening_ratio": repeated_opening_ratio(messages),
        "average_confidence": average_confidence(conversation_log),
        "stance_shift_count": stance_shift_count(conversation_log),
        "agreement_trend_slope": slope_over_rounds(agreement_series),
        "disagreement_trend_slope": slope_over_rounds(disagreement_series),
    }

    metrics.update(final_round_metrics(round_summaries))

    # Save evaluation results next to the original file
    out_path = path.with_name(path.stem + "_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[DONE] Evaluation finished.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
"""
evaluation.py

This script reads one JSON experiment log and computes a few simple
automatic metrics, so you do not need to manually inspect every output.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to experiment JSON log")
    return parser.parse_args()


def tokenize(text):
    """
    Split text into simple lowercase word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def repeated_opening_ratio(messages, n_words=3):
    """
    Measure how often messages start in the same way.
    """
    openings = []

    for msg in messages:
        tokens = tokenize(msg)
        if tokens:
            openings.append(" ".join(tokens[:n_words]))

    if not openings:
        return 0.0

    counts = Counter(openings)
    repeated = sum(c for c in counts.values() if c > 1)

    return round(repeated / len(openings), 4)


def type_token_ratio(messages):
    """
    Measure vocabulary diversity:
    unique words / total words
    """
    tokens = []

    for msg in messages:
        tokens.extend(tokenize(msg))

    if not tokens:
        return 0.0

    return round(len(set(tokens)) / len(tokens), 4)


def average_confidence(conversation_log):
    """
    Average confidence of all messages.
    """
    values = [int(x["confidence"]) for x in conversation_log if "confidence" in x]

    if not values:
        return 0.0

    return round(sum(values) / len(values), 2)


def stance_shift_count(conversation_log):
    """
    Count how often agents change stance during the discussion.
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


def final_round_metrics(round_summaries):
    """
    Get final agreement / disagreement values from the last round.
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


def slope_over_rounds(values):
    """
    Compute a simple linear trend over rounds.

    Positive agreement slope means agreement tends to increase.
    Negative disagreement slope means disagreement tends to decrease.
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


def main():
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

    out_path = path.with_name(path.stem + "_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[DONE] Evaluation finished.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
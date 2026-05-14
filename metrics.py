"""
metrics.py

This file contains functions for computing agreement and disagreement
between agents based on their stance values.
"""

from itertools import combinations


def stance_to_score(stance):
    """
    Convert stance labels into numbers.

    oppose = 0.0
    mixed = 0.5
    support = 1.0
    """
    mapping = {
        "oppose": 0.0,
        "mixed": 0.5,
        "support": 1.0,
    }
    return mapping.get(stance, 0.5)


def compute_agreement(agent_results):
    """
    Compute agreement and disagreement percentages for one round.

    If all agents are close in stance, agreement is high.
    If they are far apart, disagreement is high.
    """
    scores = [stance_to_score(item["stance"]) for item in agent_results]

    if len(scores) < 2:
        return 100.0, 0.0

    distances = []
    for a, b in combinations(scores, 2):
        distances.append(abs(a - b))

    avg_distance = sum(distances) / len(distances)

    agreement = round((1.0 - avg_distance) * 100.0, 2)
    disagreement = round(100.0 - agreement, 2)

    return agreement, disagreement


def dominant_stance(agent_results):
    """
    Find the most common stance in one round.
    """
    counts = {
        "support": 0,
        "mixed": 0,
        "oppose": 0,
    }

    for item in agent_results:
        counts[item["stance"]] += 1

    return max(counts, key=counts.get)
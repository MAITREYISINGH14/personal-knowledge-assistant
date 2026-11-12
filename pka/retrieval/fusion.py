"""Combining and shaping ranked lists."""
from collections import defaultdict


def rrf(rankings, k: int = 60):
    """Reciprocal rank fusion: each list votes 1/(k + rank) for its items.
    rankings: lists of ids, best first. Returns (id, fused score) pairs, best first."""
    score = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            score[item] += 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def diversify(items, key, k: int, max_per_group: int):
    """Keep ranking order but allow at most max_per_group items per group (e.g. per
    document version), so one long document can't fill every slot. Falls back to the
    skipped items only if there aren't enough others."""
    chosen, skipped, used = [], [], defaultdict(int)
    for item in items:
        g = key(item)
        if used[g] < max_per_group:
            chosen.append(item)
            used[g] += 1
        else:
            skipped.append(item)
        if len(chosen) == k:
            return chosen
    return (chosen + skipped)[:k]

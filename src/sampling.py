"""
Sampling helpers for reproducible classification and validation cohorts.
"""

from __future__ import annotations

import random
from collections import defaultdict


def sample_provisions(
    provisions: list[dict],
    limit: int | None = None,
    *,
    mode: str = "random",
    seed: int = 42,
) -> list[dict]:
    """
    Return a reproducible subset of provisions.

    mode='random' avoids the severe ordering bias introduced by taking the
    first N rows from an extraction-ordered corpus.
    """
    if limit is None or limit >= len(provisions):
        return list(provisions)
    if limit <= 0:
        return []
    if mode == "head":
        return list(provisions[:limit])
    if mode != "random":
        raise ValueError(f"Unknown sampling mode: {mode}")

    rng = random.Random(seed)
    return rng.sample(list(provisions), limit)


def stratified_sample_by_agreement(
    provisions: list[dict],
    per_agreement: int,
    *,
    seed: int = 42,
    agreements: tuple[str, ...] = ("RCEP", "AHKFTA", "AANZFTA"),
) -> list[dict]:
    """
    Draw an equal-sized sample per agreement.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for provision in provisions:
        buckets[provision.get("agreement", "")].append(provision)

    sample: list[dict] = []
    for agreement in agreements:
        bucket = buckets.get(agreement, [])
        if not bucket:
            continue
        take = min(per_agreement, len(bucket))
        sample.extend(rng.sample(bucket, take))

    return sample


def stratified_sample_by_agreement_and_category(
    provisions: list[dict],
    total_n: int,
    *,
    seed: int = 42,
    agreements: tuple[str, ...] = ("RCEP", "AHKFTA", "AANZFTA"),
) -> list[dict]:
    """
    Balance a sample across agreements first, then spread draws across the
    available predicted/assigned categories within each agreement.
    """
    rng = random.Random(seed)
    per_agreement = total_n // len(agreements)
    remainder = total_n % len(agreements)
    chosen: list[dict] = []

    for index, agreement in enumerate(agreements):
        bucket = [p for p in provisions if p.get("agreement") == agreement]
        if not bucket:
            continue

        target = per_agreement + (1 if index < remainder else 0)
        by_category: dict[str, list[dict]] = defaultdict(list)
        for provision in bucket:
            by_category[provision.get("category", "Other")].append(provision)

        categories = list(by_category)
        rng.shuffle(categories)

        picked_ids: set[str] = set()
        while len(picked_ids) < min(target, len(bucket)):
            progressed = False
            for category in categories:
                candidates = [
                    p for p in by_category[category]
                    if p["id"] not in picked_ids
                ]
                if not candidates:
                    continue
                picked_ids.add(rng.choice(candidates)["id"])
                progressed = True
                if len(picked_ids) >= target:
                    break
            if not progressed:
                break

        if len(picked_ids) < target:
            leftovers = [p for p in bucket if p["id"] not in picked_ids]
            rng.shuffle(leftovers)
            picked_ids.update(p["id"] for p in leftovers[: target - len(picked_ids)])

        chosen.extend([p for p in bucket if p["id"] in picked_ids])

    return chosen

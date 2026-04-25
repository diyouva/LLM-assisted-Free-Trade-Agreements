"""
Step 5 – Analysis & Model Comparison
---------------------------------------
Pure-Python analysis on already-classified JSON files. No API calls.

Produces:
  - Category × Agreement matrix (counts per FTA per category)
  - Model-vs-Model agreement scores (Cohen's kappa, raw agreement %)
  - Strategy-vs-Strategy agreement within a model
  - Per-agreement category distribution (for convergence/fragmentation analysis)

Usage:
    python -m src.analysis
"""

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AGREEMENTS, POLICY_CATEGORIES, RESULT_DIR


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load(name: str) -> list[dict]:
    path = RESULT_DIR / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _id_set(run: list[dict]) -> frozenset[str]:
    return frozenset(p["id"] for p in run)


def _build_run_catalog() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for path in sorted(RESULT_DIR.glob("classified_*.json")):
        run = _load(path.name)
        if run:
            runs[path.stem.replace("classified_", "")] = run
    return runs


def _cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa — agreement adjusted for chance. 1=perfect, 0=chance."""
    assert len(labels_a) == len(labels_b), "Length mismatch"
    n = len(labels_a)
    if n == 0:
        return float("nan")

    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    expected = sum(
        (count_a[c] / n) * (count_b[c] / n)
        for c in set(count_a) | set(count_b)
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


# ── 1. Category × Agreement matrix ────────────────────────────────────────────

def category_matrix(classified: list[dict]) -> dict:
    """Returns {category: {agreement: count}}."""
    matrix = {c: {a: 0 for a in AGREEMENTS} for c in POLICY_CATEGORIES}
    for p in classified:
        c, a = p.get("category", "Other"), p.get("agreement", "")
        if c in matrix and a in matrix[c]:
            matrix[c][a] += 1
    return matrix


# ── 2. Model/strategy agreement ───────────────────────────────────────────────

def compare_two_runs(run_a: list[dict], run_b: list[dict]) -> dict:
    """
    Given two classified outputs on the *same* provisions, compute:
      - raw_agreement: % of provisions with identical category
      - kappa: Cohen's kappa
      - confusion: list of (id, cat_a, cat_b) where they disagreed
    """
    by_id_a = {p["id"]: p["category"] for p in run_a}
    by_id_b = {p["id"]: p["category"] for p in run_b}
    common = sorted(set(by_id_a) & set(by_id_b))
    if not common:
        return {"n": 0, "raw_agreement": None, "kappa": None, "disagreements": []}

    la = [by_id_a[i] for i in common]
    lb = [by_id_b[i] for i in common]

    raw = sum(1 for a, b in zip(la, lb) if a == b) / len(common)
    kappa = _cohens_kappa(la, lb)
    disagreements = [
        {"id": i, "cat_a": by_id_a[i], "cat_b": by_id_b[i]}
        for i in common if by_id_a[i] != by_id_b[i]
    ]
    return {
        "n":             len(common),
        "raw_agreement": round(raw, 4),
        "kappa":         round(kappa, 4),
        "disagreements": disagreements,
    }


# ── 3. Convergence / fragmentation signal ─────────────────────────────────────

def convergence_signal(matrix: dict) -> dict:
    """
    For each category, compute the normalised share per agreement and
    measure dispersion (entropy + coefficient of variation).

    Low dispersion  → agreements allocate similar attention = convergent
    High dispersion → one agreement dominates                = fragmented
    """
    import math
    out = {}
    for cat, per_ag in matrix.items():
        total = sum(per_ag.values())
        if total == 0:
            continue
        shares = [v / total for v in per_ag.values()]
        # entropy (max = log(n), higher = more even)
        entropy = -sum(p * math.log(p) for p in shares if p > 0)
        # coefficient of variation
        mean = sum(shares) / len(shares)
        var  = sum((p - mean) ** 2 for p in shares) / len(shares)
        cv   = (var ** 0.5) / mean if mean else 0
        out[cat] = {
            "total_provisions": total,
            "shares":           {a: round(s, 3) for a, s in zip(per_ag, shares)},
            "entropy":          round(entropy, 3),
            "max_entropy":      round(math.log(len(per_ag)), 3),
            "coef_variation":   round(cv, 3),
            "signal":           "convergent" if cv < 0.35 else "fragmented",
        }
    return out


# ── Main: produce an analysis bundle ──────────────────────────────────────────

def run_all():
    runs = _build_run_catalog()
    cohort_ids: dict[frozenset[str], list[str]] = defaultdict(list)
    for label, run in runs.items():
        cohort_ids[_id_set(run)].append(label)

    bundle = {
        "run_summary":       {k: {"n": len(v)} for k, v in runs.items()},
        "category_matrix":   {k: category_matrix(v) for k, v in runs.items()},
        "convergence":       {k: convergence_signal(category_matrix(v)) for k, v in runs.items()},
        "pairwise_agreement": {},
        "cohorts": {
            f"cohort_{idx+1}": {
                "n": len(group_ids),
                "runs": sorted(labels),
            }
            for idx, (group_ids, labels) in enumerate(cohort_ids.items())
        },
    }

    # pairwise kappa/raw agreement
    for a, b in combinations(runs, 2):
        if _id_set(runs[a]) != _id_set(runs[b]):
            continue
        key = f"{a} vs {b}"
        cmp = compare_two_runs(runs[a], runs[b])
        bundle["pairwise_agreement"][key] = {
            "n":             cmp["n"],
            "raw_agreement": cmp["raw_agreement"],
            "kappa":         cmp["kappa"],
            "n_disagree":    len(cmp["disagreements"]),
        }

    # save full results (including disagreement lists) separately
    disagreements = {
        f"{a} vs {b}": compare_two_runs(runs[a], runs[b])["disagreements"]
        for a, b in combinations(runs, 2)
        if _id_set(runs[a]) == _id_set(runs[b])
    }

    out_path = RESULT_DIR / "analysis_bundle.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    disagree_path = RESULT_DIR / "analysis_disagreements.json"
    with open(disagree_path, "w", encoding="utf-8") as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Analysis complete")
    print(f"   Bundle:        {out_path}")
    print(f"   Disagreements: {disagree_path}")

    # pretty print
    print("\n── Pairwise agreement ──")
    for k, v in bundle["pairwise_agreement"].items():
        print(f"  {k:40s}  n={v['n']:<4}  raw={v['raw_agreement']:.3f}  κ={v['kappa']:.3f}")

    print("\n── Convergence signals (preferred stratified run) ──")
    preferred_run = next(
        (
            name for name in (
                "qwen_few_shot_stratified",
                "qwen_cot_stratified",
                "qwen_few_shot",
            )
            if name in bundle["convergence"]
        ),
        None,
    )
    sig = bundle["convergence"].get(preferred_run or "", {})
    for cat, info in sorted(sig.items(), key=lambda x: -x[1]["total_provisions"]):
        print(f"  {cat:40s}  n={info['total_provisions']:<4}  CV={info['coef_variation']:.2f}  → {info['signal']}")

    return bundle


if __name__ == "__main__":
    run_all()

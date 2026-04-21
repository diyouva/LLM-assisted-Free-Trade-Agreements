"""
Step 6 – Manual Validation
-----------------------------
Samples 50 provisions for manual ground-truth labelling, then computes
accuracy / precision / recall / F1 against each model-strategy run.

Workflow:
  1. python -m src.validation --sample    # creates validation_set.csv
  2. Open validation_set.csv, fill "gold_category" column manually
  3. python -m src.validation --evaluate  # computes metrics vs each run
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import POLICY_CATEGORIES, RAW_DIR, RESULT_DIR

VALIDATION_CSV = RESULT_DIR / "validation_set.csv"


# ── 1. Build sample ─────────────────────────────────────────────────────────

def build_sample(n: int = 50, seed: int = 1) -> None:
    """Stratified random sample across agreements — write to CSV for labelling."""
    random.seed(seed)
    provisions = json.load(open(RAW_DIR / "all_provisions.json", encoding="utf-8"))

    per_ag = n // 3                     # ~17 per agreement
    sample: list[dict] = []
    for ag in ("RCEP", "AHKFTA", "AANZFTA"):
        pool = [p for p in provisions if p["agreement"] == ag]
        sample.extend(random.sample(pool, min(per_ag, len(pool))))

    # fill to exactly n by sampling remainder from RCEP
    while len(sample) < n:
        pool = [p for p in provisions if p["agreement"] == "RCEP" and p not in sample]
        sample.append(random.choice(pool))

    with open(VALIDATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "agreement", "article", "text_preview",
                         "gold_category", "notes"])
        for p in sample:
            writer.writerow([
                p["id"], p["agreement"], p.get("article", ""),
                p["text"][:250].replace("\n", " "),
                "",    # gold_category — to be filled manually
                "",    # notes
            ])
    print(f"✅ Validation set written: {VALIDATION_CSV}")
    print(f"   {len(sample)} provisions — fill 'gold_category' column then rerun --evaluate")
    print(f"\nValid categories:")
    for c in POLICY_CATEGORIES:
        print(f"   - {c}")


# ── 2. Evaluate against each classification run ─────────────────────────────

def _load_gold() -> dict[str, str]:
    if not VALIDATION_CSV.exists():
        raise FileNotFoundError("Run --sample first, then fill gold_category column")
    gold: dict[str, str] = {}
    with open(VALIDATION_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = (row.get("gold_category") or "").strip()
            if cat:
                gold[row["id"]] = cat
    return gold


def _metrics(pairs: list[tuple[str, str]]) -> dict:
    """Return accuracy + per-class precision/recall/F1 + macro F1."""
    if not pairs:
        return {"accuracy": None, "n": 0}
    correct = sum(1 for g, p in pairs if g == p)
    acc = correct / len(pairs)

    classes = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    per_class = {}
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0
        rec  = tp / (tp + fn) if tp + fn else 0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
        per_class[c] = {"precision": round(prec, 3),
                        "recall":    round(rec, 3),
                        "f1":        round(f1, 3),
                        "support":   sum(1 for g, _ in pairs if g == c)}
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0

    return {
        "n":         len(pairs),
        "accuracy":  round(acc, 3),
        "macro_f1":  round(macro_f1, 3),
        "per_class": per_class,
    }


def evaluate() -> None:
    gold = _load_gold()
    if not gold:
        print("⚠️  No gold labels found in validation_set.csv — fill it first.")
        return
    print(f"Evaluating against {len(gold)} gold-labelled provisions\n")

    report = {}
    for fname in sorted(RESULT_DIR.glob("classified_*.json")):
        run = json.load(open(fname, encoding="utf-8"))
        pairs = [(gold[p["id"]], p["category"]) for p in run if p["id"] in gold]
        if not pairs:
            continue
        m = _metrics(pairs)
        report[fname.stem] = m
        print(f"  {fname.stem:40s}  n={m['n']:<3}  acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")

    out_path = RESULT_DIR / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Report saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample",   action="store_true",
                   help="Build validation_set.csv for manual labelling")
    g.add_argument("--evaluate", action="store_true",
                   help="Compute metrics against classified runs")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    if args.sample:
        build_sample(n=args.n)
    else:
        evaluate()

"""
Step 6 – Manual Validation
-----------------------------
Samples 50 provisions for manual ground-truth labelling, then computes
accuracy / precision / recall / F1 against each model-strategy run.

Workflow:
  1. python -m src.validation --make-stratified-sample
  2. python -m src.validation --sample    # creates validation_set.csv
  3. Open validation_set.csv, fill "gold_category" column manually
  4. python -m src.validation --export-validation-provisions
  5. python -m src.validation --evaluate  # computes metrics vs each run
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import POLICY_CATEGORIES, RAW_DIR, RESULT_DIR
from src.sampling import stratified_sample_by_agreement, stratified_sample_by_agreement_and_category

VALIDATION_CSV = RESULT_DIR / "validation_set.csv"
VALIDATION_XLSX = RESULT_DIR / "validation_checked.xlsx"
STRATIFIED_SAMPLE_JSON = RAW_DIR / "stratified_sample.json"
VALIDATION_PROVISIONS_JSON = RAW_DIR / "validation_provisions.json"


def build_stratified_sample(
    per_agreement: int = 100,
    seed: int = 42,
    output_path: Path = STRATIFIED_SAMPLE_JSON,
) -> list[dict]:
    """
    Build the agreement-balanced sample used for cross-agreement comparison.
    """
    provisions = json.load(open(RAW_DIR / "all_provisions.json", encoding="utf-8"))
    sample = stratified_sample_by_agreement(provisions, per_agreement, seed=seed)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    print(f"✅ Stratified sample written: {output_path}")
    print(f"   {len(sample)} provisions total ({per_agreement} target per agreement, seed={seed})")
    return sample


# ── 1. Build sample ─────────────────────────────────────────────────────────

def build_sample(n: int = 50, seed: int = 1, source: str | None = None) -> None:
    """
    Build a validation cohort.

    If a classified source is provided, sample across agreement and predicted
    category to reduce skew in the labelled gold set. Otherwise fall back to
    agreement-balanced sampling from the raw corpus.
    """
    if source:
        source_path = RESULT_DIR / source
        provisions = json.load(open(source_path, encoding="utf-8"))
        sample = stratified_sample_by_agreement_and_category(provisions, n, seed=seed)
    else:
        provisions = json.load(open(RAW_DIR / "all_provisions.json", encoding="utf-8"))
        per_agreement = max(1, n // 3)
        sample = stratified_sample_by_agreement(provisions, per_agreement, seed=seed)
    if len(sample) > n:
        sample = sample[:n]
    elif len(sample) < n:
        rng = random.Random(seed)
        chosen_ids = {p["id"] for p in sample}
        leftovers = [p for p in provisions if p["id"] not in chosen_ids]
        rng.shuffle(leftovers)
        sample.extend(leftovers[: n - len(sample)])

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


def export_validation_provisions(
    output_path: Path = VALIDATION_PROVISIONS_JSON,
) -> list[dict]:
    """
    Export the currently labelled validation CSV rows back into raw JSON so the
    classification CLI can run against the exact same cohort.
    """
    rows, source_path = _load_validation_rows(require_gold=False)

    all_provisions = json.load(open(RAW_DIR / "all_provisions.json", encoding="utf-8"))
    by_id = {provision["id"]: provision for provision in all_provisions}

    missing = [row["id"] for row in rows if row["id"] not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"{len(missing)} validation ids were not found in all_provisions.json. "
            f"Examples: {preview}"
        )

    selected = [by_id[row["id"]] for row in rows]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(f"✅ Validation provisions written: {output_path}")
    print(f"   {len(selected)} provisions exported from {source_path.name}")
    return selected


# ── 2. Evaluate against each classification run ─────────────────────────────

def _load_gold() -> dict[str, str]:
    rows, source_path = _load_validation_rows(require_gold=True)
    gold: dict[str, str] = {}
    for row in rows:
        cat = str(row.get("gold_category") or "").strip()
        if cat:
            gold[row["id"]] = cat
    if not gold:
        raise ValueError(f"No gold labels found in {source_path.name}")
    return gold


def _load_validation_rows(
    require_gold: bool,
) -> tuple[list[dict[str, str]], Path]:
    """
    Prefer the checked Excel workbook when present, because that is the final
    hand-labelled artifact in this repository. Fall back to the CSV draft.
    """
    if VALIDATION_XLSX.exists():
        df = pd.read_excel(VALIDATION_XLSX)
        rows = _normalise_validation_rows(df.to_dict(orient="records"))
        return rows, VALIDATION_XLSX

    if VALIDATION_CSV.exists():
        rows = _normalise_validation_rows(
            list(csv.DictReader(open(VALIDATION_CSV, encoding="utf-8")))
        )
        return rows, VALIDATION_CSV

    missing_msg = "validation_checked.xlsx or validation_set.csv not found."
    if require_gold:
        raise FileNotFoundError(f"{missing_msg} Run --sample first, then fill gold labels.")
    raise FileNotFoundError(f"{missing_msg} Run --sample first.")


def _normalise_validation_rows(rows: list[dict]) -> list[dict[str, str]]:
    normalised: list[dict[str, str]] = []
    for row in rows:
        clean_row: dict[str, str] = {}
        for key, value in row.items():
            if pd.isna(value):
                clean_row[str(key)] = ""
            else:
                clean_row[str(key)] = str(value)
        normalised.append(clean_row)
    return normalised


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
    try:
        rows, source_path = _load_validation_rows(require_gold=True)
        gold = {}
        for row in rows:
            cat = str(row.get("gold_category") or "").strip()
            if cat:
                gold[row["id"]] = cat
    except ValueError as exc:
        print(f"⚠️  {exc}")
        return
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")
        return
    print(f"Using gold labels from: {source_path}")
    print(f"Evaluating against {len(gold)} gold-labelled provisions\n")

    report = {}
    gold_ids = set(gold)
    for fname in sorted(RESULT_DIR.glob("classified_*.json")):
        run = json.load(open(fname, encoding="utf-8"))
        run_ids = {p["id"] for p in run}
        if run_ids != gold_ids:
            overlap = len(run_ids & gold_ids)
            if overlap:
                print(
                    f"  {fname.stem:40s}  skipped  "
                    f"(overlap {overlap}/{len(gold_ids)}; cohort mismatch)"
                )
            continue
        pairs = [(gold[p["id"]], p["category"]) for p in run if p["id"] in gold]
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
    g.add_argument("--make-stratified-sample", action="store_true",
                   help="Build data/raw/stratified_sample.json")
    g.add_argument("--sample",   action="store_true",
                   help="Build validation_set.csv for manual labelling")
    g.add_argument("--export-validation-provisions", action="store_true",
                   help="Build data/raw/validation_provisions.json from validation_set.csv")
    g.add_argument("--evaluate", action="store_true",
                   help="Compute metrics against classified runs")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--per-agreement", type=int, default=100,
                        help="Rows per agreement for --make-stratified-sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset generation")
    parser.add_argument("--source", default=None,
                        help="Optional classified source in data/results/ for agreement+category stratified sampling")
    args = parser.parse_args()
    if args.make_stratified_sample:
        build_stratified_sample(per_agreement=args.per_agreement, seed=args.seed)
    elif args.sample:
        build_sample(n=args.n, seed=args.seed, source=args.source)
    elif args.export_validation_provisions:
        export_validation_provisions()
    else:
        evaluate()

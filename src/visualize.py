"""
Visualization module — produces all figures for the final report.

Usage:
    python -m src.visualize                 # all figures
    python -m src.visualize --fig kappa     # one figure
"""

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AGREEMENTS, POLICY_CATEGORIES, RESULT_DIR
from src.analysis import (
    category_matrix, compare_two_runs, convergence_signal,
)

sns.set_theme(style="whitegrid", context="talk")
FIG_DIR = RESULT_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_all_runs() -> dict[str, list[dict]]:
    """Return {run_label: classified_data} for every classified_*.json."""
    runs = {}
    for p in sorted(RESULT_DIR.glob("classified_*.json")):
        label = p.stem.replace("classified_", "")
        runs[label] = json.load(open(p, encoding="utf-8"))
    return runs


# ── Figure 1: Corpus overview ─────────────────────────────────────────────────

def fig_corpus_overview(provisions_path: Path):
    data = json.load(open(provisions_path, encoding="utf-8"))
    counts = Counter(p["agreement"] for p in data)
    doc_counts = Counter(p.get("doc_type", "") for p in data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # left: per agreement
    agreements = [a for a in AGREEMENTS if a in counts]
    vals = [counts[a] for a in agreements]
    axes[0].bar(agreements, vals, color=sns.color_palette("Set2", len(agreements)))
    axes[0].set_title("Provisions per agreement")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 30, f"{v:,}", ha="center", fontsize=12)

    # right: per doc type
    doc_counts_top = dict(doc_counts.most_common(8))
    axes[1].barh(list(doc_counts_top)[::-1], list(doc_counts_top.values())[::-1],
                 color=sns.color_palette("Set2"))
    axes[1].set_title("Provisions per document type")
    axes[1].set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_corpus_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ fig_corpus_overview.png")


# ── Figure 2: Category distribution heatmap ────────────────────────────────────

def fig_category_heatmap(runs: dict):
    rows = []
    for run_name, run_data in runs.items():
        counter = Counter(p["category"] for p in run_data)
        for cat, count in counter.items():
            rows.append({"run": run_name, "category": cat,
                         "pct": 100 * count / len(run_data)})
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="category", columns="run",
                           values="pct", fill_value=0)
    # sort categories by total
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(12, 2 + 1.6 * len(runs)), 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu",
                cbar_kws={"label": "% of provisions"}, ax=ax)
    ax.set_title("Category distribution (%) per model–strategy")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_category_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ fig_category_heatmap.png")


# ── Figure 3: Cohen's κ matrix ─────────────────────────────────────────────────

def fig_kappa_matrix(runs: dict):
    km = pd.DataFrame(index=list(runs), columns=list(runs), dtype=float)
    for a, b in combinations(runs, 2):
        cmp = compare_two_runs(runs[a], runs[b])
        if cmp["kappa"] is not None:
            km.loc[a, b] = km.loc[b, a] = cmp["kappa"]
    for r in runs:
        km.loc[r, r] = 1.0

    fig, ax = plt.subplots(figsize=(max(8, 1 + 1.2 * len(runs)),
                                    max(6, 0.8 + 1.0 * len(runs))))
    sns.heatmap(km.astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-0.2, vmax=1.0,
                cbar_kws={"label": "Cohen's κ"}, ax=ax)
    ax.set_title("Inter-run classification agreement (Cohen's κ)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_kappa_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ fig_kappa_matrix.png")


# ── Figure 4: Category × Agreement (stratified) ───────────────────────────────

def fig_category_x_agreement(runs: dict):
    # pick best available stratified run
    key = next((k for k in ("qwen_few_shot_stratified",
                             "qwen_cot_stratified",
                             "llama_few_shot_stratified") if k in runs),
               "qwen_few_shot")
    if key not in runs:
        print("  (no stratified run yet — skipping fig_category_x_agreement)")
        return
    data = runs[key]
    cross = pd.DataFrame(
        [(p["agreement"], p["category"]) for p in data],
        columns=["agreement", "category"]
    )
    matrix = pd.crosstab(cross["category"], cross["agreement"])
    if matrix.shape[1] < 2:
        print(f"  (run '{key}' has only one agreement — skipping)")
        return
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Provision counts: category × agreement\n(source: {key})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_category_x_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ fig_category_x_agreement.png")


# ── Figure 5: Strategy effect within a model ──────────────────────────────────

def fig_strategy_effect(runs: dict):
    """How much do categories shift zero → few → CoT for the same model?"""
    for model_prefix in ("llama", "qwen"):
        r0 = runs.get(f"{model_prefix}_zero_shot")
        r1 = runs.get(f"{model_prefix}_few_shot")
        r2 = runs.get(f"{model_prefix}_cot")
        if not (r0 and r1):
            continue
        rows = []
        for label, r in (("zero", r0), ("few", r1), ("cot", r2)):
            if not r:
                continue
            c = Counter(p["category"] for p in r)
            for cat in POLICY_CATEGORIES:
                rows.append({"strategy": label, "category": cat,
                             "pct": 100 * c[cat] / len(r)})
        df = pd.DataFrame(rows)
        pivot = df.pivot_table(index="category", columns="strategy",
                               values="pct", fill_value=0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(10, 7))
        pivot.plot(kind="barh", ax=ax, width=0.8)
        ax.set_title(f"Prompt-strategy effect on category distribution ({model_prefix.upper()})")
        ax.set_xlabel("% of provisions")
        ax.invert_yaxis()
        ax.legend(title="Strategy", loc="lower right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"fig_strategy_effect_{model_prefix}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ fig_strategy_effect_{model_prefix}.png")


# ── Figure 6: Convergence / fragmentation ─────────────────────────────────────

def fig_convergence(runs: dict):
    key = next((k for k in ("qwen_few_shot_stratified",
                             "qwen_cot_stratified") if k in runs),
               "qwen_few_shot")
    if key not in runs:
        return
    data = runs[key]
    # skip if only one agreement present
    if len({p.get("agreement") for p in data}) < 2:
        print("  (single-agreement run — convergence fig skipped)")
        return

    mat = category_matrix(data)
    sig = convergence_signal(mat)
    rows = [
        {"category": c, "total": v["total_provisions"],
         "entropy_ratio": v["entropy"] / v["max_entropy"] if v["max_entropy"] else 0,
         "signal": v["signal"]}
        for c, v in sig.items()
    ]
    df = pd.DataFrame(rows).sort_values("total", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if s == "convergent" else "#e67e22" for s in df["signal"]]
    ax.barh(df["category"], df["entropy_ratio"], color=colors)
    ax.axvline(0.95, ls="--", color="gray", label="≈ perfect convergence")
    ax.set_xlabel("Entropy ratio (1.0 = equal attention across all agreements)")
    ax.set_title(f"Convergence signal per category\n(green = converging, orange = fragmented)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ fig_convergence.png")


# ── Figure 7: Validation accuracy ─────────────────────────────────────────────

def fig_validation_accuracy():
    path = RESULT_DIR / "validation_report.json"
    if not path.exists():
        print("  (validation_report.json not present — skipping)")
        return
    rep = json.load(open(path))
    df = pd.DataFrame([
        {"run": k, "accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
        for k, v in rep.items()
    ]).sort_values("macro_f1")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.barh([i - 0.2 for i in x], df["accuracy"], height=0.4, label="Accuracy")
    ax.barh([i + 0.2 for i in x], df["macro_f1"], height=0.4, label="Macro F1")
    ax.set_yticks(list(x))
    ax.set_yticklabels(df["run"])
    ax.set_xlim(0, 1)
    ax.set_title("Validation accuracy vs gold-labelled sample")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_validation_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ fig_validation_accuracy.png")


# ── Main ───────────────────────────────────────────────────────────────────────

FIGURES = {
    "corpus":      lambda runs: fig_corpus_overview(
        Path(RESULT_DIR).parent / "raw" / "all_provisions.json"),
    "heatmap":     fig_category_heatmap,
    "kappa":       fig_kappa_matrix,
    "xagreement":  fig_category_x_agreement,
    "strategy":    fig_strategy_effect,
    "convergence": fig_convergence,
    "validation":  lambda runs: fig_validation_accuracy(),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", choices=list(FIGURES) + ["all"], default="all")
    args = parser.parse_args()
    runs = _load_all_runs()
    print(f"Loaded {len(runs)} classified runs")
    targets = list(FIGURES) if args.fig == "all" else [args.fig]
    for name in targets:
        FIGURES[name](runs)


if __name__ == "__main__":
    main()

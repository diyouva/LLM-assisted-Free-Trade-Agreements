"""
Step 4 – Cross-Agreement Comparison
-------------------------------------
For each policy category, retrieves the most relevant provisions from each
agreement and uses an LLM to produce a structured comparative analysis.

Usage:
    python -m src.comparison --category "Rules of Origin" --model claude
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    AGREEMENTS,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    POLICY_CATEGORIES,
    RESULT_DIR,
)
from src.classification import call_claude, call_qwen, call_llama, call_openai
from src.embedding import retrieve_similar

MODEL_CALLERS = {
    # Free models
    "llama":  call_llama,
    "qwen": call_qwen,
    # Paid models (optional)
    "claude": call_claude,
    "openai": call_openai,
}

# ── Comparison prompt ──────────────────────────────────────────────────────────

def _build_comparison_prompt(
    category: str,
    provisions_by_agreement: dict[str, list[dict]],
) -> str:
    """Build a structured prompt to compare provisions across agreements."""
    sections = []
    for agreement, provs in provisions_by_agreement.items():
        texts = "\n\n".join(
            f"  [{i+1}] {p['article']} — {p['text'][:600]}"
            for i, p in enumerate(provs)
        )
        sections.append(f"### {agreement}\n{texts}")

    provisions_block = "\n\n".join(sections)

    return f"""You are a senior trade policy analyst comparing provisions across ASEAN free trade agreements.

Policy Category: **{category}**

Below are relevant provisions from three agreements:

{provisions_block}

Please provide a structured comparative analysis covering:
1. **Key similarities** – What do all three agreements have in common on this topic?
2. **Key differences** – What are the most significant differences in policy design, thresholds, or obligations?
3. **Flexibility and rigidity** – Which agreement appears most flexible or most prescriptive?
4. **Convergence or fragmentation** – Do these agreements appear to be converging toward a common approach, or diverging?
5. **Policy implications** – What would this mean for a trade negotiator or compliance officer?

Be specific and cite the agreements by name. Keep your analysis concise (under 400 words)."""


# ── Main comparison function ───────────────────────────────────────────────────

def compare_category(
    category: str,
    model: str = "claude",
    n_provisions: int = 3,
) -> dict:
    """
    Compare provisions from all three agreements for a given policy category.

    Args:
        category:      One of the POLICY_CATEGORIES.
        model:         'claude' or 'openai'.
        n_provisions:  Number of provisions to retrieve per agreement.

    Returns:
        Dict with keys: category, model, provisions_used, analysis.
    """
    caller = MODEL_CALLERS[model]
    provisions_by_agreement: dict[str, list[dict]] = {}

    print(f"\n  Retrieving provisions for: {category}")
    for agreement in AGREEMENTS:
        similar = retrieve_similar(
            query=f"{category} provisions obligations requirements",
            agreement_filter=agreement,
            n_results=n_provisions,
        )
        provisions_by_agreement[agreement] = similar
        print(f"    {agreement}: {len(similar)} provisions retrieved")

    # Check we have something to compare
    total_provs = sum(len(v) for v in provisions_by_agreement.values())
    if total_provs == 0:
        return {
            "category": category, "model": model,
            "provisions_used": {}, "analysis": "No provisions found.",
        }

    prompt   = _build_comparison_prompt(category, provisions_by_agreement)
    analysis = caller(prompt)

    result = {
        "category":           category,
        "model":              model,
        "provisions_used":    {
            ag: [{"id": p["id"], "article": p["article"], "text": p["text"][:300]}
                 for p in provs]
            for ag, provs in provisions_by_agreement.items()
        },
        "analysis":           analysis,
    }
    return result


def run_full_comparison(
    model: str = "claude",
    categories: list[str] | None = None,
    delay: float = 1.0,
) -> list[dict]:
    """
    Run comparison across all (or specified) policy categories.

    Returns:
        List of comparison result dicts.
    """
    cats = categories or POLICY_CATEGORIES
    all_results = []

    print(f"\n{'='*60}")
    print(f"Full cross-agreement comparison  |  model={model}")
    print(f"Categories: {len(cats)}")
    print("="*60)

    for cat in cats:
        print(f"\n[{cats.index(cat)+1}/{len(cats)}] {cat}")
        result = compare_category(cat, model=model)
        all_results.append(result)

        # Save incrementally
        out_path = RESULT_DIR / f"comparison_{model}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        time.sleep(delay)

    print(f"\n✅ Full comparison complete: {len(all_results)} categories")
    print(f"   Saved to: {out_path}")
    return all_results


def build_comparison_matrix(classified_path: Path) -> dict:
    """
    Build a category × agreement matrix showing provision counts.
    Reads from a classification output JSON.

    Returns:
        Dict: {category: {agreement: count}}
    """
    with open(classified_path, encoding="utf-8") as f:
        classified = json.load(f)

    matrix: dict[str, dict[str, int]] = {
        cat: {ag: 0 for ag in AGREEMENTS} for cat in POLICY_CATEGORIES
    }

    for prov in classified:
        cat = prov.get("category", "Other")
        ag  = prov.get("agreement", "")
        if cat in matrix and ag in matrix[cat]:
            matrix[cat][ag] += 1

    out_path = RESULT_DIR / "category_matrix.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=2)

    print(f"✅ Category matrix saved to: {out_path}")
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-agreement FTA comparison")
    parser.add_argument("--model",    default="llama",
                        choices=["llama", "qwen", "claude", "openai"])
    parser.add_argument("--category", default=None,
                        help="Single category to compare (default: all)")
    args = parser.parse_args()

    if args.category:
        result = compare_category(args.category, model=args.model)
        print("\n" + "="*60)
        print(result["analysis"])
    else:
        run_full_comparison(model=args.model)

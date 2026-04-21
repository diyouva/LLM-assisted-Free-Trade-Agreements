"""
Step 3b – Attribute Extraction
--------------------------------
Extract structured fields (thresholds, conditions, obligations) from
classified provisions — directly answering RQ2 in the proposal:
  "how do comparable provisions differ across agreements in terms of
   observable policy design features such as rules of origin criteria,
   legal flexibility, and structure of commitments?"

Targets per category:
  - Rules of Origin      → RVC %, de minimis %, CTC rule, wholly-obtained flag
  - Tariff Commitments   → phase-out years, % reduction, HS code scope
  - Trade in Services    → MFN flag, NT flag, sector coverage
  - Dispute Settlement   → consultation days, panel days, appeal flag

Output: data/results/attributes_{model}.json
        one row per provision that has at least one extracted attribute.

Usage:
    python -m src.attribute_extraction --model qwen --category "Rules of Origin"
    python -m src.attribute_extraction --model qwen  # all categories
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULT_DIR
from src.classification import MODEL_CALLERS


# ── Regex-based quick extractors (deterministic, run first) ────────────────────

RX_PERCENT       = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|per\s*cent|percent)", re.IGNORECASE)
RX_DE_MINIMIS    = re.compile(r"de\s*minimis[^.]{0,120}?(\d{1,3}(?:\.\d+)?)\s*(?:%|per\s*cent)", re.IGNORECASE)
RX_RVC           = re.compile(r"(?:regional\s*value\s*content|RVC)[^.]{0,80}?(\d{1,3})\s*(?:%|per\s*cent)", re.IGNORECASE)
RX_CTC           = re.compile(r"\b(CTH|CTSH|CC|change\s*in\s*tariff\s*heading|change\s*in\s*tariff\s*sub-?heading|change\s*in\s*chapter)\b", re.IGNORECASE)
RX_WHOLLY_OBT    = re.compile(r"wholly\s*obtained|wholly\s*produced", re.IGNORECASE)
RX_PHASE_OUT     = re.compile(r"(\d{1,2})\s*(?:year|annual\s*instalment|tranche)", re.IGNORECASE)
RX_DAYS          = re.compile(r"(\d{1,4})\s*(?:calendar\s*)?days?", re.IGNORECASE)
RX_MFN           = re.compile(r"most[- ]favoured[- ]nation|MFN\b", re.IGNORECASE)
RX_NATIONAL_TRT  = re.compile(r"national\s*treatment|NT\b", re.IGNORECASE)


def regex_attributes(text: str, category: str) -> dict:
    """Fast regex sweep — deterministic attribute flags & numeric values."""
    out: dict = {}

    if category == "Rules of Origin":
        m = RX_DE_MINIMIS.search(text)
        if m: out["de_minimis_pct"] = float(m.group(1))
        m = RX_RVC.search(text)
        if m: out["rvc_pct"] = int(m.group(1))
        m = RX_CTC.search(text)
        if m: out["ctc_rule"] = m.group(1).upper().replace(" ", "_")
        if RX_WHOLLY_OBT.search(text):
            out["wholly_obtained_clause"] = True

    elif category == "Tariff Commitments":
        m = RX_PHASE_OUT.search(text)
        if m: out["phase_out_years"] = int(m.group(1))
        pcts = RX_PERCENT.findall(text)
        if pcts:
            out["percentages_mentioned"] = [float(p) for p in pcts[:5]]

    elif category == "Trade in Services":
        if RX_MFN.search(text):          out["mfn_clause"] = True
        if RX_NATIONAL_TRT.search(text): out["national_treatment_clause"] = True

    elif category == "Dispute Settlement":
        days = RX_DAYS.findall(text)
        if days: out["days_referenced"] = [int(d) for d in days[:5]]

    return out


# ── LLM prompt for richer attribute extraction ─────────────────────────────────

def _build_prompt(category: str, text: str) -> str:
    schema = {
        "Rules of Origin": (
            '{"rvc_pct": int|null, "de_minimis_pct": float|null, '
            '"ctc_rule": "CTH|CTSH|CC|null", "wholly_obtained_clause": bool, '
            '"cumulation_type": "bilateral|diagonal|full|null", "notes": str}'
        ),
        "Tariff Commitments": (
            '{"phase_out_years": int|null, "max_reduction_pct": float|null, '
            '"hs_scope": "all|selected|null", "staging_category": str|null, "notes": str}'
        ),
        "Trade in Services": (
            '{"mfn_clause": bool, "national_treatment_clause": bool, '
            '"market_access_clause": bool, "mode_of_supply": [1,2,3,4]|null, "notes": str}'
        ),
        "Dispute Settlement": (
            '{"consultation_days": int|null, "panel_formation_days": int|null, '
            '"appeal_mechanism": bool, "notes": str}'
        ),
        "Non-Tariff Measures": (
            '{"measure_type": "SPS|TBT|licensing|quota|null", '
            '"transparency_requirement": bool, "notes": str}'
        ),
        "Investment": (
            '{"expropriation_clause": bool, "isds_clause": bool, '
            '"performance_requirements_restricted": bool, "notes": str}'
        ),
    }.get(category, '{"notes": str}')

    return f"""Extract structured attributes from this FTA provision.
Category: {category}

Return a **single JSON object** matching this schema (use null for missing values):
{schema}

Provision text:
\"\"\"{text}\"\"\"

Respond with ONLY the JSON object, no prose, no code fences."""


def _parse_json(raw: str) -> dict | None:
    """Robust JSON parser — strips markdown fences, <think> blocks, surrounding text."""
    if not raw or raw == "Error":
        return None
    # remove <think>...</think> sections (Qwen)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # remove ```json ... ``` fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.replace("```", "")
    # find first {...} block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ── Main extraction loop ───────────────────────────────────────────────────────

# Categories we have rich schemas for
ATTRIBUTE_CATEGORIES = [
    "Rules of Origin",
    "Tariff Commitments",
    "Trade in Services",
    "Dispute Settlement",
    "Non-Tariff Measures",
    "Investment",
]


def extract_attributes(
    classified_path: Path,
    model: str = "qwen",
    categories: list[str] | None = None,
    limit_per_category: int | None = None,
    delay: float = 8.0,
) -> list[dict]:
    """
    Run attribute extraction on already-classified provisions.

    Args:
        classified_path:    Path to classified_*.json output.
        model:              LLM to use ('llama' or 'qwen').
        categories:         Which categories to process.
        limit_per_category: Cap per category (None = all).
        delay:              Seconds between LLM calls.

    Returns:
        List of result dicts with regex + LLM attributes merged.
    """
    with open(classified_path, encoding="utf-8") as f:
        classified = json.load(f)

    cats = categories or ATTRIBUTE_CATEGORIES
    caller = MODEL_CALLERS[model]

    results: list[dict] = []
    print(f"\nAttribute extraction | source={classified_path.name} | model={model}")

    for cat in cats:
        bucket = [p for p in classified if p.get("category") == cat]
        if limit_per_category:
            bucket = bucket[:limit_per_category]
        if not bucket:
            print(f"  [{cat}] 0 provisions — skipping")
            continue

        print(f"\n  [{cat}] {len(bucket)} provisions …")
        for i, prov in enumerate(bucket, 1):
            if i % 10 == 0 or i == 1:
                print(f"    {i}/{len(bucket)} …", flush=True)

            text = prov["text"]
            attrs_rx  = regex_attributes(text, cat)
            attrs_llm = _parse_json(caller(_build_prompt(cat, text)))

            # merge: regex is ground truth for deterministic fields,
            # LLM fills in semantic flags regex can't capture
            merged = {**(attrs_llm or {}), **attrs_rx}

            results.append({
                "id":        prov["id"],
                "agreement": prov["agreement"],
                "article":   prov.get("article", ""),
                "category":  cat,
                "attributes_regex": attrs_rx,
                "attributes_llm":   attrs_llm,
                "attributes":       merged,
                "text":      text[:400],
            })

            time.sleep(delay)

    # save
    src_stem = classified_path.stem          # e.g. classified_qwen_few_shot
    model_tag = src_stem.replace("classified_", "")
    out_path = RESULT_DIR / f"attributes_{model_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Attribute extraction complete: {len(results)} rows")
    print(f"   Saved to: {out_path}")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   default="classified_qwen_few_shot.json",
                        help="Source classified file in data/results/")
    parser.add_argument("--model",    default="qwen", choices=["llama", "qwen"])
    parser.add_argument("--category", default=None,
                        help="Single category (default: all supported)")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Max provisions per category")
    parser.add_argument("--delay",    type=float, default=8.0)
    args = parser.parse_args()

    src = RESULT_DIR / args.source
    cats = [args.category] if args.category else None
    extract_attributes(src, model=args.model, categories=cats,
                       limit_per_category=args.limit, delay=args.delay)

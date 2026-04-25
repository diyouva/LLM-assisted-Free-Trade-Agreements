"""
Master pipeline runner
-----------------------
Runs all steps in sequence. Pass --step to run individual steps.

Steps:
  1  extract      – PDF extraction & clause segmentation
  stratified_sample – Build agreement-balanced comparison sample
  2  embed        – Generate embeddings & build vector DB
  3  classify     – LLM classification (default: Claude, zero-shot, 200 provisions)
  4  compare      – Cross-agreement comparison (default: Claude, all categories)
  validation_sample – Build validation_set.csv for manual labelling
  validation_export – Export validation_provisions.json from validation_set.csv
  all             – Run steps 1→4 in order

Example:
    python run_pipeline.py --step all --model qwen
    python run_pipeline.py --step classify --model llama --strategy few_shot --limit 100
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def step_extract():
    print("\n" + "█"*60)
    print("  STEP 1: PDF Extraction & Clause Segmentation")
    print("█"*60)
    from src.extraction import run_extraction
    return run_extraction()


def step_embed():
    print("\n" + "█"*60)
    print("  STEP 2: Embeddings & Vector Database")
    print("█"*60)
    from src.embedding import build_vector_store, load_provisions
    provisions = load_provisions()
    build_vector_store(provisions)
    return provisions


def step_stratified_sample(per_agreement: int, seed: int):
    print("\n" + "█"*60)
    print("  STEP 1B: Build Stratified Sample")
    print("█"*60)
    from src.validation import build_stratified_sample
    return build_stratified_sample(per_agreement=per_agreement, seed=seed)


def step_classify(model: str, strategy: str, limit: int | None, sample_mode: str, seed: int):
    print("\n" + "█"*60)
    print(f"  STEP 3: LLM Classification  [{model} / {strategy}]")
    print("█"*60)
    from config import RAW_DIR
    from src.classification import classify_provisions
    with open(RAW_DIR / "all_provisions.json", encoding="utf-8") as f:
        provisions = json.load(f)
    return classify_provisions(
        provisions,
        model=model,
        strategy=strategy,
        limit=limit,
        sample_mode=sample_mode,
        seed=seed,
    )


def step_compare(model: str):
    print("\n" + "█"*60)
    print(f"  STEP 4: Cross-Agreement Comparison  [{model}]")
    print("█"*60)
    from src.comparison import run_full_comparison
    return run_full_comparison(model=model)


def step_validation_sample(n: int, seed: int, source: str | None):
    print("\n" + "█"*60)
    print("  STEP 4B: Build Validation Sample")
    print("█"*60)
    from src.validation import build_sample
    return build_sample(n=n, seed=seed, source=source)


def step_validation_export():
    print("\n" + "█"*60)
    print("  STEP 4C: Export Validation Provisions")
    print("█"*60)
    from src.validation import export_validation_provisions
    return export_validation_provisions()


def main():
    parser = argparse.ArgumentParser(description="FTA LLM Pipeline Runner")
    parser.add_argument(
        "--step",
        default="all",
        choices=[
            "1", "extract",
            "stratified_sample",
            "2", "embed",
            "3", "classify",
            "4", "compare",
            "validation_sample",
            "validation_export",
            "all",
        ],
    )
    parser.add_argument("--model",    default="llama",
                        choices=["llama", "qwen", "claude", "openai"])
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "few_shot", "cot"])
    parser.add_argument("--limit",    type=int, default=200,
                        help="Max provisions for classification step (default: 200)")
    parser.add_argument("--sample-mode", default="random",
                        choices=["random", "head"],
                        help="How to choose provisions when --limit is set")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed used for sampling")
    parser.add_argument("--per-agreement", type=int, default=100,
                        help="Rows per agreement for stratified sample generation")
    parser.add_argument("--validation-n", type=int, default=50,
                        help="Rows in validation sample generation")
    parser.add_argument("--validation-source", default=None,
                        help="Optional classified file in data/results/ used to balance validation sampling")
    args = parser.parse_args()

    step = args.step

    if step in ("1", "extract", "all"):
        step_extract()

    if step == "stratified_sample":
        step_stratified_sample(args.per_agreement, args.seed)

    if step in ("2", "embed", "all"):
        step_embed()

    if step in ("3", "classify", "all"):
        step_classify(args.model, args.strategy, args.limit, args.sample_mode, args.seed)

    if step in ("4", "compare", "all"):
        step_compare(args.model)

    if step == "validation_sample":
        step_validation_sample(args.validation_n, args.seed, args.validation_source)

    if step == "validation_export":
        step_validation_export()

    print("\n\n" + "="*60)
    print("  Pipeline complete!")
    print("="*60)
    print("\nOutputs:")
    print(f"  data/raw/          – extracted provisions JSON")
    print(f"  data/chromadb/     – vector database")
    print(f"  data/results/      – classification & comparison JSON")
    print(f"\nNext: open notebooks/analysis.ipynb for visualisation & reporting")


if __name__ == "__main__":
    main()

"""
Master pipeline runner
-----------------------
Runs all steps in sequence. Pass --step to run individual steps.

Steps:
  1  extract      – PDF extraction & clause segmentation
  2  embed        – Generate embeddings & build vector DB
  3  classify     – LLM classification (default: Claude, zero-shot, 200 provisions)
  4  compare      – Cross-agreement comparison (default: Claude, all categories)
  all             – Run steps 1→4 in order

Example:
    python run_pipeline.py --step all --model claude
    python run_pipeline.py --step classify --model openai --strategy few_shot --limit 100
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


def step_classify(model: str, strategy: str, limit: int | None):
    print("\n" + "█"*60)
    print(f"  STEP 3: LLM Classification  [{model} / {strategy}]")
    print("█"*60)
    from config import RAW_DIR
    from src.classification import classify_provisions
    with open(RAW_DIR / "all_provisions.json", encoding="utf-8") as f:
        provisions = json.load(f)
    return classify_provisions(provisions, model=model, strategy=strategy, limit=limit)


def step_compare(model: str):
    print("\n" + "█"*60)
    print(f"  STEP 4: Cross-Agreement Comparison  [{model}]")
    print("█"*60)
    from src.comparison import run_full_comparison
    return run_full_comparison(model=model)


def main():
    parser = argparse.ArgumentParser(description="FTA LLM Pipeline Runner")
    parser.add_argument(
        "--step",
        default="all",
        choices=["1", "extract", "2", "embed", "3", "classify", "4", "compare", "all"],
    )
    parser.add_argument("--model",    default="llama",
                        choices=["llama", "qwen", "claude", "openai"])
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "few_shot", "cot"])
    parser.add_argument("--limit",    type=int, default=200,
                        help="Max provisions for classification step (default: 200)")
    args = parser.parse_args()

    step = args.step

    if step in ("1", "extract", "all"):
        step_extract()

    if step in ("2", "embed", "all"):
        step_embed()

    if step in ("3", "classify", "all"):
        step_classify(args.model, args.strategy, args.limit)

    if step in ("4", "compare", "all"):
        step_compare(args.model)

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

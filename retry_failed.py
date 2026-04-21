"""
Retry only the failed provisions (raw_response == 'Error') from a previous run.
Saves results back to the same file.

Usage:
    python retry_failed.py --file classified_llama_few_shot.json --model llama --strategy few_shot
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULT_DIR
from src.classification import MODEL_CALLERS, PROMPT_BUILDERS, _parse_category


def retry_failed(filename: str, model: str, strategy: str, delay: float = 20.0):
    path = RESULT_DIR / filename
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    failed = [i for i, d in enumerate(data) if d.get("raw_response") == "Error"]
    print(f"Found {len(failed)} failed provisions in {filename}. Retrying…")
    print(f"Model: {model}  |  Strategy: {strategy}  |  Delay: {delay}s between calls\n")

    caller         = MODEL_CALLERS[model]
    prompt_builder = PROMPT_BUILDERS[strategy]

    for count, idx in enumerate(failed, 1):
        prov = data[idx]
        print(f"  [{count}/{len(failed)}] id={prov['id']} …", end=" ", flush=True)

        prompt   = prompt_builder(prov["text"])
        raw_resp = caller(prompt)
        category = _parse_category(raw_resp)

        data[idx]["raw_response"] = raw_resp
        data[idx]["category"]     = category
        print(f"→ {category}")

        # Save after every provision so progress isn't lost
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        time.sleep(delay)

    still_failed = sum(1 for d in data if d.get("raw_response") == "Error")
    print(f"\n✅ Done. Remaining errors: {still_failed}/{len(data)}")
    print(f"   Saved to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",     required=True, help="e.g. classified_llama_few_shot.json")
    parser.add_argument("--model",    default="llama",  choices=["llama", "qwen", "claude", "openai"])
    parser.add_argument("--strategy", default="few_shot", choices=["zero_shot", "few_shot", "cot"])
    parser.add_argument("--delay",    type=float, default=20.0, help="Seconds between calls (default 20)")
    args = parser.parse_args()

    retry_failed(args.file, args.model, args.strategy, args.delay)

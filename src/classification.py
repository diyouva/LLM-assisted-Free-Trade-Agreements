"""
Step 3 – LLM Classification Pipeline
--------------------------------------
Classifies each provision into a policy category using three prompt strategies:
  1. Zero-shot
  2. Few-shot
  3. Chain-of-thought (CoT)

FREE models supported (both via Groq — same API key):
  - llama  → LLaMA 3.3 70B  (large model)
  - gemma  → Gemma 2 9B     (small model — size comparison)

Paid models (optional):
  - claude  → Claude claude-sonnet-4-5 (Anthropic)
  - openai  → GPT-4o (OpenAI)

Usage:
    python -m src.classification --model llama --strategy zero_shot --limit 200
    python -m src.classification --model gemma --strategy zero_shot --limit 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import groq as groq_lib

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    FEW_SHOT_EXAMPLES,
    QWEN_MODEL,
    GROQ_API_KEY,
    LLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    POLICY_CATEGORIES,
    RAW_DIR,
    RESULT_DIR,
)
from src.sampling import sample_provisions

# ── Clients ────────────────────────────────────────────────────────────────────

def _groq_client():
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "Get a free key at: https://console.groq.com"
        )
    return groq_lib.Groq(api_key=GROQ_API_KEY)


def _claude_client():
    import anthropic
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _openai_client():
    import openai
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    return openai.OpenAI(api_key=OPENAI_API_KEY)


# ── Prompt builders ────────────────────────────────────────────────────────────

CATEGORIES_STR = "\n".join(f"- {c}" for c in POLICY_CATEGORIES)


def _zero_shot_prompt(provision_text: str) -> str:
    return f"""You are a trade policy analyst. Classify the following Free Trade Agreement provision into exactly one of the categories below.

Categories:
{CATEGORIES_STR}

Provision:
\"\"\"{provision_text}\"\"\"

Respond with ONLY the category name, nothing else."""


def _few_shot_prompt(provision_text: str) -> str:
    examples = "\n\n".join(
        f"Provision: \"{ex['text']}\"\nCategory: {ex['category']}"
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"""You are a trade policy analyst. Classify FTA provisions into one of these categories:
{CATEGORIES_STR}

Here are examples of correct classifications:

{examples}

Now classify this provision:
Provision: \"{provision_text}\"
Category:"""


def _cot_prompt(provision_text: str) -> str:
    return f"""You are a trade policy analyst classifying Free Trade Agreement provisions.

Available categories:
{CATEGORIES_STR}

Provision to classify:
\"\"\"{provision_text}\"\"\"

Think step by step:
1. Identify the main legal subject (tariffs, origin rules, services, investment, etc.)
2. Identify key obligations or rights described
3. Match to the most appropriate category

After your reasoning, end with:
CATEGORY: <category name>"""


PROMPT_BUILDERS = {
    "zero_shot": _zero_shot_prompt,
    "few_shot":  _few_shot_prompt,
    "cot":       _cot_prompt,
}


# ── LLM call helpers ────────────────────────────────────────────────────────────

def _parse_category(response_text: str) -> str:
    """Extract the category name from model output."""
    import re as _re
    # Strip Qwen-style <think>...</think> reasoning blocks
    cleaned = _re.sub(r"<think>.*?</think>", "", response_text,
                      flags=_re.DOTALL | _re.IGNORECASE).strip()
    # If response was truncated mid-think (no </think>), take text after the <think>
    if not cleaned and "<think>" in response_text.lower():
        cleaned = response_text.split("</think>")[-1].strip() if "</think>" in response_text \
                  else response_text  # last resort — use raw
    if not cleaned:
        cleaned = response_text

    # For CoT responses, look for CATEGORY: tag
    if "CATEGORY:" in cleaned:
        parts = cleaned.split("CATEGORY:")
        candidate = parts[-1].strip().splitlines()[0].strip().rstrip(".")
    else:
        lines = [l for l in cleaned.splitlines() if l.strip()]
        candidate = lines[-1].strip().rstrip(".") if lines else ""

    # Match against known categories (case-insensitive)
    for cat in POLICY_CATEGORIES:
        if cat.lower() in candidate.lower():
            return cat
    # Also search the full cleaned text (fallback)
    for cat in POLICY_CATEGORIES:
        if cat.lower() in cleaned.lower():
            return cat
    return "Other"


def call_llama(prompt: str, max_retries: int = 8) -> str:
    """Call LLaMA 3.3 70B via Groq (free).

    Groq free tier limits for llama-3.3-70b-versatile:
      - 30 req/min, ~6,000 tokens/min
    Few-shot & CoT prompts are ~1,200-1,500 tokens, so we may only fit
    4-5 requests per minute — handled via retry with proper cooldown.
    """
    import re as _re
    client = _groq_client()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LLAMA_MODEL,
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except groq_lib.RateLimitError as e:
            # Extract the suggested wait time from the error message.
            # Groq uses formats like "22.5s" or "3m22.5s"; convert both to seconds.
            err_str = str(e)
            print(f"  [Groq] Rate limit raw: {err_str[:120]}", flush=True)
            m_min = _re.search(r"try again in\s+(\d+)m([\d\.]+)s", err_str, _re.IGNORECASE)
            m_sec = _re.search(r"try again in\s+([\d\.]+)s", err_str, _re.IGNORECASE)
            if m_min:
                wait = int(m_min.group(1)) * 60 + float(m_min.group(2)) + 2
            elif m_sec:
                wait = float(m_sec.group(1)) + 1
            else:
                wait = 15 * (attempt + 1)   # 15s, 30s, 45s … if no hint
            print(f"  [Groq] Rate limit — waiting {wait:.0f}s …", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"  [Groq] Error: {e}")
            return "Error"
    return "Error"


def call_qwen(prompt: str, max_retries: int = 8, max_tokens: int = 2048) -> str:
    """Call Qwen 3 32B via Groq (free) — same API key as LLaMA.

    Qwen 3 is a "thinking" model — it emits <think>...</think> before the answer.
    Need ≥ 1,500 tokens for classification, 2,048+ for comparison prompts.
    """
    import re as _re
    client = _groq_client()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=QWEN_MODEL,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except groq_lib.RateLimitError as e:
            err_str = str(e)
            match = _re.search(r"try again in\s+([\d\.]+)s", err_str, _re.IGNORECASE)
            wait = float(match.group(1)) + 1 if match else (15 * (attempt + 1))
            print(f"  [Gemma] Rate limit — waiting {wait:.0f}s …")
            time.sleep(wait)
        except Exception as e:
            print(f"  [Gemma] Error: {e}")
            return "Error"
    return "Error"


def call_claude(prompt: str, max_retries: int = 3) -> str:
    import anthropic
    client = _claude_client()
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"  [Claude] Rate limit — waiting {wait}s …")
            time.sleep(wait)
        except Exception as e:
            print(f"  [Claude] Error: {e}")
            return "Error"
    return "Error"


def call_openai(prompt: str, max_retries: int = 3) -> str:
    import openai
    client = _openai_client()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait = 2 ** attempt
            print(f"  [OpenAI] Rate limit — waiting {wait}s …")
            time.sleep(wait)
        except Exception as e:
            print(f"  [OpenAI] Error: {e}")
            return "Error"
    return "Error"


MODEL_CALLERS = {
    # Free models — both via Groq, same API key
    "llama":  call_llama,   # LLaMA 3.3 70B — large model
    "qwen":  call_qwen,   # Qwen 3 32B    — cross-architecture comparison (Alibaba)
    # Paid models (optional)
    "claude": call_claude,
    "openai": call_openai,
}


# ── Main classification loop ──────────────────────────────────────────────────

def classify_provisions(
    provisions: list[dict],
    model: str = "claude",
    strategy: str = "zero_shot",
    limit: int | None = None,
    delay: float = 0.3,
    out_suffix: str = "",
    sample_mode: str = "random",
    seed: int = 42,
) -> list[dict]:
    """
    Classify provisions using the specified model and prompt strategy.

    Args:
        provisions: List of provision dicts (from extraction.py).
        model:      'claude' or 'openai'.
        strategy:   'zero_shot', 'few_shot', or 'cot'.
        limit:      Cap on number of provisions to classify (None = all).
        delay:      Seconds to sleep between API calls.

    Returns:
        List of provision dicts with added 'category' and 'raw_response' fields.
    """
    if model not in MODEL_CALLERS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(MODEL_CALLERS)}")
    if strategy not in PROMPT_BUILDERS:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(PROMPT_BUILDERS)}")

    caller         = MODEL_CALLERS[model]
    prompt_builder = PROMPT_BUILDERS[strategy]

    subset = sample_provisions(provisions, limit, mode=sample_mode, seed=seed)
    results = []

    # Groq free tier: ~6,000 tokens/min for LLaMA 3.3 70B
    #   zero_shot ~300 tokens  → 2s delay (20 req/min safe)
    #   few_shot  ~1,400 tokens → 12s delay (~5 req/min)
    #   cot       ~800 tokens  →  8s delay (~7 req/min)
    if model == "llama" and delay < 2:
        token_delay = {"zero_shot": 2, "few_shot": 12, "cot": 15}
        delay = token_delay.get(strategy, 5)

    # Qwen 3 32B — mid-size, similar token budget to LLaMA but separate daily quota
    if model == "qwen" and delay < 2:
        token_delay = {"zero_shot": 2, "few_shot": 10, "cot": 10}
        delay = token_delay.get(strategy, 5)

    print(
        f"\nClassifying {len(subset):,} provisions  |  model={model}  "
        f"strategy={strategy}  sampling={sample_mode}  seed={seed}"
    )
    for i, prov in enumerate(subset, 1):
        if i % 10 == 0 or i == 1:
            print(f"  [{i}/{len(subset)}] …", flush=True)

        prompt   = prompt_builder(prov["text"])
        raw_resp = caller(prompt)
        category = _parse_category(raw_resp)

        result = {**prov, "category": category, "raw_response": raw_resp,
                  "model": model, "strategy": strategy}
        results.append(result)
        time.sleep(delay)

    # Save
    suffix_tag = f"_{out_suffix}" if out_suffix else ""
    out_name = f"classified_{model}_{strategy}{suffix_tag}.json"
    out_path = RESULT_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Classification complete: {len(results):,} provisions")
    print(f"   Saved to: {out_path}")
    return results


def compare_strategies(
    provisions: list[dict],
    model: str = "claude",
    sample_size: int = 50,
) -> dict:
    """
    Run all three strategies on the same sample and compare category distributions.
    Returns a dict mapping strategy → classified results.
    """
    import random
    sample = random.sample(provisions, min(sample_size, len(provisions)))

    comparison = {}
    for strategy in PROMPT_BUILDERS:
        print(f"\n── Strategy: {strategy} ──")
        comparison[strategy] = classify_provisions(
            sample, model=model, strategy=strategy
        )

    # Save comparison summary
    summary = {}
    for strategy, results in comparison.items():
        cats = [r["category"] for r in results]
        dist = {cat: cats.count(cat) for cat in POLICY_CATEGORIES if cats.count(cat) > 0}
        summary[strategy] = dist

    out_path = RESULT_DIR / f"strategy_comparison_{model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Strategy comparison saved to: {out_path}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify FTA provisions with LLMs")
    parser.add_argument("--model",    default="llama",
                        choices=["llama", "qwen", "claude", "openai"])
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "few_shot", "cot"])
    parser.add_argument("--limit",    type=int, default=None,
                        help="Max number of provisions to classify")
    parser.add_argument("--source",   default="all_provisions.json",
                        help="Source JSON filename in data/raw/ "
                             "(e.g. stratified_sample.json)")
    parser.add_argument("--suffix",   default="",
                        help="Extra suffix on output filename")
    parser.add_argument("--sample-mode", default="random",
                        choices=["random", "head"],
                        help="How to choose provisions when --limit is set")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed used for sampling")
    args = parser.parse_args()

    provisions_path = RAW_DIR / args.source
    with open(provisions_path, encoding="utf-8") as f:
        provisions = json.load(f)

    classify_provisions(
        provisions,
        model=args.model,
        strategy=args.strategy,
        limit=args.limit,
        out_suffix=args.suffix,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

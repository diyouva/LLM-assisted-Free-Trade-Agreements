"""
Central configuration for the FTA Comparative Analysis project.

Free models used (no payment required):
  - LLaMA 3.3 70B → Groq API (free key at console.groq.com)
  - Qwen 3 32B    → Groq API (same key, Alibaba Cloud model)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (works from any working directory)
load_dotenv(Path(__file__).parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
AGREE_DIR  = BASE_DIR / "Agreement"
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
RESULT_DIR = DATA_DIR / "results"
CHROMA_DIR = DATA_DIR / "chromadb"

for d in [RAW_DIR, PROC_DIR, RESULT_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Agreement catalogue ────────────────────────────────────────────────────────
# Maps short name → list of PDF filenames (all in AGREE_DIR)
AGREEMENTS = {
    "RCEP": [
        "RCEP Agreement (All Chapters).pdf",
    ],
    "AHKFTA": [
        "AHKFTA.pdf",
        "AHKFTA annex 2-11.pdf",
        # Annex 8-1 is 64 MB tariff schedules — handled separately if needed
    ],
    "AANZFTA": [
        "AANZFTA-legal-text-PRINTED-Signed.pdf",
        "AANZFTA - First Protocol to Amend the Agreement Establishing the AANZFTA (2014).pdf",
        "AANZFTA - Implementing Arrangement for the AANZFTA Economic Co-Operation Work Programme.pdf",
        "AANZFTA - Understanding on Tariff Reduction and:or Elimination of Customs Duties.pdf",
    ],
}

# ── Policy categories for classification ──────────────────────────────────────
POLICY_CATEGORIES = [
    "Tariff Commitments",
    "Rules of Origin",
    "Non-Tariff Measures",
    "Trade in Services",
    "Investment",
    "Dispute Settlement",
    "Intellectual Property",
    "Customs Procedures",
    "Sanitary and Phytosanitary Measures",
    "General Provisions / Definitions",
    "Other",
]

# ── Embedding model ────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "fta_provisions"

# ── LLM models ────────────────────────────────────────────────────────────────
# Free models via Groq (primary) — same API key, no extra signup
LLAMA_MODEL  = "llama-3.3-70b-versatile"   # LLaMA 3.3 70B via Groq (free)
QWEN_MODEL   = "qwen/qwen3-32b"            # Qwen 3 32B via Groq (free) — cross-architecture comparison

# Paid models (optional, kept for reference)
CLAUDE_MODEL = "claude-sonnet-4-5"
OPENAI_MODEL = "gpt-4o"

# ── API keys (from environment) ────────────────────────────────────────────────
# Free keys — get them at:
#   Groq : https://console.groq.com → GROQ_API_KEY  (used for both LLaMA and Qwen)
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")

# Optional paid keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# ── Extraction settings ────────────────────────────────────────────────────────
# Minimum characters for a clause to be kept
MIN_CLAUSE_CHARS = 80
# Maximum chars per chunk sent to LLM
MAX_CHUNK_CHARS  = 1500

# ── Few-shot examples for classification ──────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "text": "Each Party shall reduce or eliminate its customs duties on originating goods "
                "of the other Parties in accordance with its Schedule in Annex I.",
        "category": "Tariff Commitments",
    },
    {
        "text": "A good shall be considered as originating in a Party if it is wholly obtained "
                "or produced entirely in that Party, or satisfies the Product Specific Rules.",
        "category": "Rules of Origin",
    },
]

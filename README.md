# A Computational Framework for Comparative Analysis of Free Trade Agreements

**Diyouva Christa Novith** | Machine Learning Foundation with Python | Carnegie Mellon University — Spring 2026

---

## Overview

This project applies large language models (LLMs) to a standing problem in international trade policy: **comparing the legal architecture of Free Trade Agreements (FTAs) at scale**. The Asia-Pacific region alone maintains dozens of overlapping FTAs — each with thousands of provisions — making manual comparison impractical for analysts and negotiators. This framework automates the extraction, classification, and cross-agreement comparison of FTA provisions using freely available LLMs, validated against a hand-labelled gold set.

**Three agreements analysed:**

| Agreement | Full Name | Parties | Signed |
|-----------|-----------|---------|--------|
| RCEP | Regional Comprehensive Economic Partnership | 15 ASEAN+ nations | 2020 |
| AHKFTA | ASEAN–Hong Kong Free Trade Agreement | 10 ASEAN + HK | 2017 |
| AANZFTA | ASEAN–Australia–New Zealand FTA | 10 ASEAN + AU + NZ | 2009 |

**Research questions:**
1. Can LLMs reliably classify FTA provisions into standard policy categories, and how does accuracy vary across prompt strategies?
2. How do the three agreements differ in their allocation of legal text across policy domains?
3. Do the agreements show structural convergence or fragmentation in their treatment of key trade topics?

---

## Key Findings

- **Qwen 3 32B with chain-of-thought prompting** achieves the best validation performance: **70.0% accuracy, 0.693 macro-F1** on a 50-provision gold set.
- **Prompt strategy matters more than model size** for this task: few-shot prompting *hurt* Qwen (dropped F1 by 15pp) but helped LLaMA (gained 4pp). CoT recovered Qwen's performance by giving the model space to reason before committing.
- **Inter-model agreement is near-random** (Cohen's κ = −0.02 between LLaMA few-shot and Qwen few-shot), meaning the two models disagree on roughly half of all provisions despite similar aggregate accuracy.
- **AHKFTA is Rules-of-Origin heavy** (28% of sampled provisions vs 9% for RCEP), reflecting its tighter origin criteria; RCEP allocates more text to general and services provisions.
- **Tariff Commitments and Rules of Origin** are the most structurally convergent categories across all three agreements; **Dispute Settlement and Trade in Services** are the most fragmented.

---

## Architecture

```
PDFs (3 FTAs)
     │
     ▼
┌─────────────────────┐
│  src/extraction.py  │  pdfplumber → PyMuPDF → Tesseract OCR fallback
│  Provision schema   │  Fields: id, agreement, doc_type, chapter,
│  (paragraph-level)  │          article, paragraph_idx, text, char_count
└─────────────────────┘
     │  all_provisions.json  (~5,000+ provisions)
     ▼
┌─────────────────────┐
│  src/embedding.py   │  sentence-transformers (all-MiniLM-L6-v2)
│  ChromaDB vector    │  Cosine distance, top-3 retrieval per category
│  store (RAG)        │  Used for cross-agreement comparison context
└─────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  src/classification.py  — LLM Classification        │
│                                                      │
│  Models:  LLaMA 3.3 70B (Groq)                      │
│           Qwen 3 32B    (Groq / Alibaba Cloud)       │
│                                                      │
│  Strategies:                                         │
│    zero_shot  — category list + provision text       │
│    few_shot   — 2 labelled examples prepended        │
│    cot        — "think step by step" instruction     │
│                                                      │
│  Stratified sampling: 100 provisions/agreement       │
│  (corrects for RCEP's 53.5% corpus share, seed=42)  │
└──────────────────────────────────────────────────────┘
     │  classified_*.json  (6 run files)
     ▼
┌─────────────────────┐    ┌──────────────────────────┐
│  src/comparison.py  │    │  src/attribute_extraction │
│  RAG-augmented LLM  │    │  Regex (RVC%, de-minimis) │
│  cross-agreement    │    │  + LLM (CTC rules, HS     │
│  narrative analysis │    │  scope, staging categories)│
└─────────────────────┘    └──────────────────────────┘
     │
     ▼
┌─────────────────────┐
│  src/analysis.py    │  Cohen's κ inter-run agreement
│  src/validation.py  │  Macro-F1 vs 50-provision gold set
│  src/visualize.py   │  7 publication figures
└─────────────────────┘
```

---

## Repository Structure

```
.
├── Agreement/                    # Source PDFs (3 FTAs, 7 documents)
├── data/
│   ├── raw/
│   │   ├── all_provisions.json   # Full extracted corpus
│   │   └── validation_provisions.json  # 50-provision gold set
│   └── results/
│       ├── classified_llama_zero_shot.json
│       ├── classified_llama_few_shot.json
│       ├── classified_llama_cot.json
│       ├── classified_qwen_zero_shot.json
│       ├── classified_qwen_few_shot.json
│       ├── classified_qwen_cot.json
│       ├── classified_qwen_few_shot_stratified.json
│       ├── analysis_bundle.json  # κ values, entropy, convergence signal
│       ├── validation_report.json
│       ├── attributes_roo.json   # RoO attribute extraction
│       ├── attributes_tariff.json
│       ├── comparison_qwen.json  # Cross-agreement narratives
│       └── fig_*.png             # 7 report figures
├── src/
│   ├── extraction.py             # PDF → provision schema
│   ├── embedding.py              # ChromaDB vector store
│   ├── classification.py         # LLM classification (LLaMA + Qwen)
│   ├── comparison.py             # RAG-augmented cross-agreement comparison
│   ├── attribute_extraction.py   # Hybrid regex + LLM attribute extraction
│   ├── analysis.py               # Cohen's κ, convergence signal
│   ├── validation.py             # Gold-set accuracy / macro-F1 scoring
│   └── visualize.py              # Figure generation
├── notebooks/
│   └── analysis.ipynb            # Interactive walkthrough of full pipeline
├── config.py                     # Paths, model IDs, policy categories
├── run_pipeline.py               # CLI entrypoint for all pipeline stages
├── REPORT.md                     # Final submission report (policy narrative)
├── REPORT_DRAFT.md               # Technical reference report
└── requirements.txt
```

---

## Setup

### 1. Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Tesseract OCR (for scanned PDF fallback)
brew install tesseract        # macOS
sudo apt install tesseract-ocr  # Ubuntu/Debian
```

### 2. API Key

All LLM calls use the **Groq free tier** — one key covers both LLaMA 3.3 70B and Qwen 3 32B.

```bash
# Get your free key at https://console.groq.com
echo "GROQ_API_KEY=your_key_here" > .env
```

> **Free-tier limits for LLaMA 3.3 70B (Groq):**
> - 30 requests/min, ~6,000 tokens/min, 100,000 tokens/day (rolling 24-hour window)
>
> CoT classification of 100 provisions consumes the full daily quota in one run.
> Plan CoT runs for early morning when the rolling window is fresh.

### 3. Place Agreement PDFs

Put all FTA PDFs in the `Agreement/` directory. The expected filenames are defined in `config.py → AGREEMENTS`.

---

## Running the Pipeline

### Full pipeline (sequential)

```bash
python run_pipeline.py --step all
```

### Step by step

```bash
# 1. Extract provisions from PDFs
python run_pipeline.py --step extract

# 2. Build ChromaDB vector store
python run_pipeline.py --step embed

# 3. Classify provisions
#    --model: llama | qwen
#    --strategy: zero_shot | few_shot | cot
#    --limit: number of provisions (default: all)
python -m src.classification --model llama --strategy zero_shot
python -m src.classification --model qwen  --strategy cot

# Stratified sample (100 per agreement, corrects for corpus imbalance)
python -m src.classification --model qwen --strategy few_shot \
    --source stratified_sample.json --suffix stratified

# 4. Cross-agreement comparison (RAG-augmented LLM analysis)
python run_pipeline.py --step compare --model qwen

# 5. Attribute extraction (RoO numeric fields + categorical flags)
python run_pipeline.py --step attributes

# 6. Statistical analysis (Cohen's κ, convergence signal)
python run_pipeline.py --step analyse

# 7. Generate figures
python -m src.visualize
```

### Validation (against 50-provision gold set)

```bash
# Create stratified validation sample (run once)
python -m src.validation --sample

# Manually label data/results/validation_set.csv (gold_category column)
# See data/results/validation_checked.xlsx for the completed gold set

# Score all classified runs against gold labels
python -m src.validation --evaluate
```

---

## Models & Prompt Strategies

### Models

| Model | Provider | Parameters | Free Tier |
|-------|----------|-----------|-----------|
| LLaMA 3.3 70B | Groq (Meta) | 70B | 100K tokens/day |
| Qwen 3 32B | Groq (Alibaba Cloud) | 32B | Separate quota |

Qwen 3 is a "thinking" model — it emits `<think>...</think>` reasoning tokens before the final answer. The classification code strips these before recording the category label.

### Prompt Strategies

| Strategy | Description | Tokens/call (approx.) |
|----------|-------------|----------------------|
| `zero_shot` | Category list + provision text; no examples | ~400 |
| `few_shot` | Two labelled examples prepended to the prompt | ~900 |
| `cot` | Instruction to reason step-by-step before labelling | ~1,400 |

### Validation Results

| Model + Strategy | Accuracy | Macro-F1 | n |
|-----------------|----------|----------|---|
| Qwen 3 32B — chain-of-thought | **0.700** | **0.693** | 50 |
| LLaMA 3.3 70B — zero-shot | 0.700 | 0.591 | 50 |
| LLaMA 3.3 70B — few-shot | 0.680 | 0.635 | 50 |
| Qwen 3 32B — zero-shot | 0.680 | 0.596 | 50 |
| Qwen 3 32B — few-shot | 0.580 | 0.540 | 50 |
| LLaMA 3.3 70B — chain-of-thought | 0.480 | 0.527 | 50 |

### Inter-Run Agreement (Cohen's κ)

| Pair | κ | Interpretation |
|------|---|----------------|
| LLaMA zero-shot vs LLaMA few-shot | 0.51 | Moderate |
| Qwen zero-shot vs Qwen few-shot | 0.02 | Near-chance |
| LLaMA few-shot vs Qwen few-shot | −0.02 | Worse than chance |

---

## Policy Categories

The 11 classification categories align with the standard WTO/UNCTAD FTA chapter taxonomy:

| Category | Example provision |
|----------|------------------|
| Tariff Commitments | Duty reduction schedules, tariff elimination timelines |
| Rules of Origin | RVC thresholds, Change-in-Tariff-Classification rules |
| Non-Tariff Measures | Import licensing, technical standards, quotas |
| Trade in Services | Mode 1–4 service liberalisation commitments |
| Investment | ISDS, national treatment for investors |
| Dispute Settlement | Panel procedures, consultation timelines |
| Customs Procedures | Documentation, advance rulings, single-window |
| Sanitary and Phytosanitary | Food safety, plant/animal health standards |
| Intellectual Property | Copyright, trademarks, GI protections |
| General Provisions / Definitions | Definitions, scope, general exceptions |
| Other | Provisions not fitting any above category |

---

## Attribute Extraction

Beyond classification, the pipeline extracts structured numeric and categorical attributes from Rules of Origin and Tariff Commitment provisions:

**Rules of Origin attributes:**
- RVC threshold (%) — extracted via regex
- Change-in-Tariff-Classification rule (`CC` / `CTH` / `CTSH`) — LLM
- HS code scope — LLM
- De-minimis threshold (%) — regex

**Tariff Commitment attributes:**
- Phase-out years — regex
- Staging category — LLM
- Product scope — LLM

Key finding: AHKFTA requires Chapter-level Change in Tariff Classification (`CC`) while RCEP requires Heading-level (`CTH`) — both apply a 40% RVC threshold, but AHKFTA imposes stricter transformation requirements on manufacturers.

---

## Iterative Development Notes

This project went through several methodological pivots that are documented in the reports:

1. **Gemini → Qwen substitution**: Initial design used Gemini 1.5 Flash (Google AI Studio). Quota was exhausted mid-run; the architecture was rebuilt around Qwen 3 32B via Groq using the same API key infrastructure as LLaMA.

2. **HS code alignment**: Cross-agreement comparison was originally designed to align provisions by HS code. Sparse article-level metadata made this infeasible; semantic similarity via ChromaDB was used instead.

3. **Stratified sampling**: The raw corpus is 53.5% RCEP provisions. All comparative analyses use a 100-provision stratified sample (seed 42) to prevent RCEP from dominating results.

4. **Annex coverage gap**: Quantitative tariff thresholds cluster in tariff schedules (annexes), which were not segmented into the provision corpus. Numeric attribute extraction is limited to main-text provisions.

---

## Notebooks

`notebooks/analysis.ipynb` is a self-contained walkthrough that:
- Runs the full pipeline end-to-end (skips steps already completed)
- Reproduces all 7 report figures
- Shows Cohen's κ heatmap across all run pairs
- Displays RAG-retrieved cross-agreement comparisons per policy category

Open with:
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Figures Generated

| File | Description |
|------|-------------|
| `fig_provision_counts.png` | Corpus size per agreement |
| `fig_strategy_comparison.png` | Category distribution by model × strategy |
| `fig_category_matrix.png` | Provision count heatmap: category × agreement |
| `fig_category_distribution.png` | Stacked bar: policy mix per agreement |
| `fig_convergence.png` | Entropy-based convergence signal per category |
| `fig_kappa_matrix.png` | Cohen's κ matrix across all run pairs |
| `fig_validation_accuracy.png` | Accuracy and macro-F1 by model + strategy |

---

## Limitations

- **Triage-grade accuracy only**: 70% accuracy is sufficient for analyst triage but not for compliance or legal use.
- **Annex coverage**: Tariff schedules are excluded; numeric thresholds in schedules are not captured.
- **API quota constraints**: Groq free-tier LLaMA daily quota (100K tokens) prevents running full CoT classification and validation in the same day.
- **Three agreements**: Findings are suggestive, not statistically generalisable across the broader FTA landscape.
- **English-only**: All PDFs are English-language versions; translation layers are not included.

---

## Citation

> Novith, D. C. (2026). *A Computational Framework for Comparative Analysis of Free Trade Agreements Using Large Language Models*. Carnegie Mellon University — Machine Learning Foundation with Python (Spring 2026).

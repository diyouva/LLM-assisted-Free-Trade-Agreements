# A Computational Framework for Comparative Analysis of Free Trade Agreements

**Diyouva Christa Novith**

---

## Overview

This project applies large language models (LLMs) to a standing problem in international trade policy: **comparing the legal architecture of Free Trade Agreements (FTAs) at scale**. The Asia-Pacific region alone maintains dozens of overlapping FTAs — each with thousands of provisions — making manual comparison impractical for analysts and negotiators. This framework automates the extraction, classification, and cross-agreement comparison of FTA provisions using freely available LLMs, validated against a hand-labelled gold set.

**Three agreements analysed:**

| Agreement | Full Name | Parties | Signed |
|-----------|-----------|---------|--------|
| RCEP | Regional Comprehensive Economic Partnership | 10 ASEAN + CN, JP, NZ, AU, KR | 2020 |
| AHKFTA | ASEAN–Hong Kong Free Trade Agreement | 10 ASEAN + HK | 2017 |
| AANZFTA | ASEAN–Australia–New Zealand FTA | 10 ASEAN + AU + NZ | 2009 |

**Research questions:**
1. Can LLMs reliably classify FTA provisions into standard policy categories, and how does accuracy vary across prompt strategies?
2. How do the three agreements differ in their allocation of legal text across policy domains?
3. Do the agreements show structural convergence or fragmentation in their treatment of key trade topics?

---

## Key Findings

- **Validation performance is modest on the current gold set.** Best accuracy is **LLaMA 3.3 70B zero-shot** at **48.0%**; best macro-F1 is **Qwen 3 32B chain-of-thought** at **0.442** on the same 50-provision cohort.
- **Prompt effects are mixed rather than uniformly beneficial.** CoT slightly improves Qwen over its other strategies on macro-F1, but the margin is small after the rerun; few-shot no longer supports the earlier "clear improvement" narrative.
- **Inter-run agreement is now moderate to substantial once cohorts are aligned exactly.** For example, `llama_zero_shot` vs `qwen_zero_shot` reaches **κ = 0.702**, and `llama_few_shot` vs `qwen_few_shot` reaches **κ = 0.582** on the shared 200-row cohort.
- **AHKFTA is Rules-of-Origin heavy** (48% of sampled provisions vs 24% for RCEP), reflecting its tighter origin criteria and goods-only scope; RCEP and AANZFTA allocate substantially more text to services, investment, and dispute settlement.
- **Customs Procedures and General Provisions** are the most structurally convergent categories — a shared regional template is emerging; **Dispute Settlement and Trade in Services** are the most fragmented — AHKFTA has zero provisions in both, reflecting its narrower bilateral mandate.

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
     │  all_provisions.json  (4,059 provisions)
     ▼
┌─────────────────────┐
│  src/embedding.py   │  sentence-transformers (all-MiniLM-L6-v2)
│  ChromaDB vector    │  Cosine distance, top-3 retrieval per category
│  store (RAG)        │  Used for cross-agreement comparison context
└─────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  src/classification.py  — LLM Classification         │
│                                                      │
│  Models:  LLaMA 3.3 70B (Groq)                       │
│           Qwen 3 32B    (Groq / Alibaba Cloud)       │
│                                                      │
│  Strategies:                                         │
│    zero_shot  — category list + provision text       │
│    few_shot   — 2 labelled examples prepended        │
│    cot        — "think step by step" instruction     │
│                                                      │
│  Main runs: reproducible random sample (seed=42)     │
│  Comparative runs: stratified 100/agreement          │
│  (corrects for RCEP's 53.5% corpus share)            │
└──────────────────────────────────────────────────────┘
     │  classified_*.json  (6 run files)
     ▼
┌─────────────────────┐    ┌────────────────────────────┐
│  src/comparison.py  │    │  src/attribute_extraction  │
│  RAG-augmented LLM  │    │  Regex (RVC%, de-minimis)  │
│  cross-agreement    │    │  + LLM (CTC rules, HS      │
│  narrative analysis │    │  scope, staging categories)│
└─────────────────────┘    └────────────────────────────┘
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
│   │   ├── stratified_sample.json      # 100/agreement comparison cohort
│   │   └── validation_provisions.json  # Exact JSON cohort used for validation runs
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

### Full pipeline (core sequential stages)

```bash
python run_pipeline.py --step all
```

`--step all` runs the four core stages only: `extract → embed → classify → compare`.
It does **not** generate the stratified comparison sample or validation datasets for you.

### Step by step

#### Core pipeline

```bash
# 1. Extract provisions from PDFs
python run_pipeline.py --step extract

# 1b. Build agreement-balanced comparison cohort (100 per agreement)
python run_pipeline.py --step stratified_sample --per-agreement 100 --seed 42

# 2. Build ChromaDB vector store
python run_pipeline.py --step embed

# 3. Classify provisions
#    --model: llama | qwen
#    --strategy: zero_shot | few_shot | cot
#    --limit: number of provisions (default in run_pipeline.py: 200)
#    --sample-mode: random | head   (use random unless debugging)
python -m src.classification --model llama --strategy zero_shot --limit 200 --sample-mode random --seed 42
python -m src.classification --model qwen  --strategy cot       --limit 100 --sample-mode random --seed 42

# Stratified sample (100 per agreement, corrects for corpus imbalance)
python -m src.classification --model qwen --strategy few_shot \
    --source stratified_sample.json --suffix stratified

# 4. Cross-agreement comparison (RAG-augmented LLM analysis)
python run_pipeline.py --step compare --model qwen

# 5. Attribute extraction (RoO numeric fields + categorical flags)
python -m src.attribute_extraction --model qwen

# 6. Statistical analysis (Cohen's κ, convergence signal)
python -m src.analysis

# 7. Generate figures
python -m src.visualize
```

#### Validation workflow (against a 50-provision gold set)

```bash
# 1. Build a validation CSV from a classified cohort
python run_pipeline.py --step validation_sample \
    --validation-n 50 \
    --validation-source classified_qwen_few_shot_stratified.json \
    --seed 42

# 2. Manually label data/results/validation_checked.xlsx if present
#    (fallback: data/results/validation_set.csv)

# 3. Export the exact validation cohort to JSON for classification reruns
python run_pipeline.py --step validation_export

# 4. Reclassify that exact validation cohort with each model/strategy
python -m src.classification --model llama --strategy zero_shot --source validation_provisions.json --suffix validation
python -m src.classification --model llama --strategy few_shot  --source validation_provisions.json --suffix validation
python -m src.classification --model llama --strategy cot       --source validation_provisions.json --suffix validation
python -m src.classification --model qwen  --strategy zero_shot --source validation_provisions.json --suffix validation
python -m src.classification --model qwen  --strategy few_shot  --source validation_provisions.json --suffix validation
python -m src.classification --model qwen  --strategy cot       --source validation_provisions.json --suffix validation

# 5. Score only runs whose IDs exactly match the labelled validation cohort
python -m src.validation --evaluate
```

### Recommended full rerun after code changes

If you have changed extraction, sampling, validation, or comparison logic, the
saved JSON outputs may be stale. In that case, rerun the pipeline in this order:

```bash
python run_pipeline.py --step extract
python run_pipeline.py --step stratified_sample --per-agreement 100 --seed 42
python run_pipeline.py --step embed

python -m src.classification --model llama --strategy zero_shot --limit 200 --sample-mode random --seed 42
python -m src.classification --model llama --strategy few_shot  --limit 200 --sample-mode random --seed 42
python -m src.classification --model llama --strategy cot       --limit 100 --sample-mode random --seed 42
python -m src.classification --model qwen  --strategy zero_shot --limit 200 --sample-mode random --seed 42
python -m src.classification --model qwen  --strategy few_shot  --limit 200 --sample-mode random --seed 42
python -m src.classification --model qwen  --strategy cot       --limit 100 --sample-mode random --seed 42
python -m src.classification --model qwen  --strategy few_shot  --source stratified_sample.json --suffix stratified

python run_pipeline.py --step validation_sample --validation-n 50 --validation-source classified_qwen_few_shot_stratified.json --seed 42
# manually label validation_checked.xlsx (or validation_set.csv if no workbook exists)
python run_pipeline.py --step validation_export
# rerun the six *_validation classifications here
python -m src.validation --evaluate

python -m src.comparison --model qwen --source classified_qwen_few_shot_stratified.json
python -m src.analysis
python -m src.visualize
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
| LLaMA 3.3 70B — zero-shot | **0.480** | 0.431 | 50 |
| Qwen 3 32B — chain-of-thought | 0.460 | **0.442** | 50 |
| Qwen 3 32B — zero-shot | 0.380 | 0.424 | 50 |
| Qwen 3 32B — few-shot | 0.380 | 0.373 | 50 |
| LLaMA 3.3 70B — few-shot | 0.340 | 0.336 | 50 |
| LLaMA 3.3 70B — chain-of-thought | 0.320 | 0.327 | 50 |

### Inter-Run Agreement (Cohen's κ)

| Pair | κ | Interpretation |
|------|---|----------------|
| LLaMA zero-shot vs LLaMA few-shot | 0.668 | Substantial |
| Qwen zero-shot vs Qwen few-shot | 0.689 | Substantial |
| LLaMA few-shot vs Qwen few-shot | 0.582 | Moderate |
| LLaMA zero-shot vs Qwen zero-shot | 0.702 | Substantial |

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
| `fig_corpus_overview.png` | Corpus size per agreement and per document type |
| `fig_category_heatmap.png` | Category distribution (%) heatmap across all model-strategy runs |
| `fig_kappa_matrix.png` | Cohen's κ matrix across all run pairs |
| `fig_category_x_agreement.png` | Provision count heatmap: category × agreement (stratified sample) |
| `fig_strategy_effect_llama.png` | How prompt strategy shifts category distribution for LLaMA |
| `fig_strategy_effect_qwen.png` | How prompt strategy shifts category distribution for Qwen |
| `fig_convergence.png` | Entropy-based convergence signal per category |
| `fig_validation_accuracy.png` | Accuracy and macro-F1 by model + strategy |

---

## Limitations

- **Triage-grade accuracy only**: the current 32–48% validation accuracy range is useful for analyst prioritisation, not for compliance or legal use.
- **Annex coverage**: Tariff schedules are excluded; numeric thresholds in schedules are not captured.
- **API quota constraints**: Groq free-tier LLaMA daily quota (100K tokens) prevents running full CoT classification and validation in the same day.
- **Three agreements**: Findings are suggestive, not statistically generalisable across the broader FTA landscape.
- **English-only**: All PDFs are English-language versions; translation layers are not included.

---

## Citation

> Novith, D. C. (2026). *A Computational Framework for Comparative Analysis of Free Trade Agreements Using Large Language Models*. Carnegie Mellon University — Machine Learning Foundation with Python (Spring 2026).

# A Computational Framework for Comparative Analysis of Free Trade Agreements

**Diyouva Christa Novith**

An LLM-based pipeline that converts three Asia-Pacific Free Trade Agreements into a structured provision dataset and surfaces design differences across them. The output is positioned as a first-pass triage layer for analysts, not a substitute for legal review.

---

## Overview

The Asia-Pacific region maintains dozens of overlapping Free Trade Agreements (FTAs), each running to thousands of legal provisions. Comparing two or three of them on the same topic is slow, manual work, and the answer often turns on small differences buried deep in the text. Trade economists call this the spaghetti bowl problem.

This framework tests whether an LLM-based pipeline can help with that comparison work. It segments the legal text of three ASEAN-centred FTAs into provisions, asks the model to classify each into a policy category, and uses retrieval to draft side-by-side notes on how each agreement handles the same topic. The whole pipeline runs on free-tier APIs and can be pointed at any new FTA PDF.

**Three agreements analysed:**

| Agreement | Full Name | Parties | Signed |
|-----------|-----------|---------|--------|
| RCEP | Regional Comprehensive Economic Partnership | 10 ASEAN + CN, JP, NZ, AU, KR | 2020 |
| AHKFTA | ASEAN-Hong Kong Free Trade Agreement | 10 ASEAN + HK | 2017 |
| AANZFTA | ASEAN-Australia-New Zealand FTA | 10 ASEAN + AU + NZ | 2009 |

**Research questions:**
1. Can LLMs reliably classify FTA provisions into standard policy categories, and how does accuracy change across models and prompt strategies?
2. How do the agreements differ in observable design features such as thresholds, governance structures, and scope?
3. Do the agreements show structural convergence or fragmentation in their treatment of key trade policy topics?

---

## Headline Findings

- **Triage-grade classification, not provision-level adjudication.** Best accuracy is **LLaMA 3.3 70B zero-shot** at **0.480**; best Macro-F1 is **Qwen 3 32B chain-of-thought** at **0.442** on the 50-provision gold set. No run clears 50%.
- **Prompt strategy is a per-model decision.** Chain-of-thought prompting raises Qwen's Macro-F1 by 1.8 points but drops LLaMA's by 10.4 points. Few-shot prompting hurts both. The asymmetry only became visible after all six combinations were scored against the same gold set.
- **Inter-run agreement is moderate to substantial.** Cohen's κ ranges from 0.582 (LLaMA few-shot vs Qwen few-shot, n=200) to 0.702 (LLaMA zero-shot vs Qwen zero-shot, n=200). Models share a meaningful baseline signal that prompting strategies tend to disrupt rather than amplify.
- **Same threshold, different transformation rule.** RCEP and AHKFTA both use a 40% Regional Value Content threshold, but RCEP applies a Change in Tariff Heading rule at the 4-digit level while AHKFTA applies a Change in Chapter rule at the 2-digit level. A product can in principle satisfy one rule and fail the other.
- **Customs Procedures converges; Dispute Settlement and Intellectual Property fragment.** Entropy ratio reaches 1.00 for Customs Procedures (consistent with the WTO Trade Facilitation baseline). Dispute Settlement (0.47) and Intellectual Property (0.37) score as fragmented, mainly because AHKFTA does not include those chapters at all.

---

## Architecture

```
PDFs (3 FTAs, 7 documents)
     │
     ▼
┌─────────────────────┐
│  src/extraction.py  │  pdfplumber, PyMuPDF, Tesseract OCR fallback
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
│  src/classification.py  (LLM Classification)         │
│                                                      │
│  Models:  LLaMA 3.3 70B (Groq)                       │
│           Qwen 3 32B    (Groq)                       │
│                                                      │
│  Strategies:                                         │
│    zero_shot   category list + provision text        │
│    few_shot    2 labelled examples prepended         │
│    cot         "think step by step" instruction      │
│                                                      │
│  Main runs: reproducible random sample (seed=42)     │
│  Comparative runs: stratified 100/agreement          │
│  (corrects for RCEP's 53.5% corpus share)            │
└──────────────────────────────────────────────────────┘
     │  classified_*.json  (6 run files + 1 stratified)
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
│  src/visualize.py   │  Publication figures
└─────────────────────┘
```

---

## Repository Structure

```
.
├── Agreement/                    # Source PDFs (3 FTAs, 7 documents)
├── data/
│   ├── raw/
│   │   ├── all_provisions.json         # Full extracted corpus
│   │   ├── stratified_sample.json      # 100/agreement comparison cohort
│   │   └── validation_provisions.json  # Cohort for validation runs
│   └── results/
│       ├── classified_llama_zero_shot.json
│       ├── classified_llama_few_shot.json
│       ├── classified_llama_cot.json
│       ├── classified_qwen_zero_shot.json
│       ├── classified_qwen_few_shot.json
│       ├── classified_qwen_cot.json
│       ├── classified_qwen_few_shot_stratified.json
│       ├── analysis_bundle.json     # κ values, entropy, convergence signal
│       ├── validation_report.json   # Accuracy and Macro-F1 per run
│       ├── attributes_roo.json      # RoO attribute extraction
│       ├── attributes_tariff.json
│       ├── comparison_qwen.json     # Cross-agreement narratives
│       └── fig_*.png                # Report figures
├── src/
│   ├── extraction.py             # PDF to provision schema
│   ├── embedding.py              # ChromaDB vector store
│   ├── classification.py         # LLM classification (LLaMA + Qwen)
│   ├── comparison.py             # RAG cross-agreement comparison
│   ├── attribute_extraction.py   # Hybrid regex + LLM attribute extraction
│   ├── analysis.py               # Cohen's κ, convergence signal
│   ├── validation.py             # Gold-set accuracy / Macro-F1 scoring
│   └── visualize.py              # Figure generation
├── scripts/
│   └── build_deliverable_report.py  # Generates Deliverable_Report_FTA_LLM.docx
├── notebooks/
│   └── analysis.ipynb            # Interactive walkthrough of full pipeline
├── config.py                     # Paths, model IDs, policy categories
├── run_pipeline.py               # CLI entrypoint for all pipeline stages
├── index.html                    # Dashboard summarising all results
├── REPORT.md                     # Project report
├── JOURNAL_PAPER.md              # Journal-style write-up
├── METHODOLOGY.md                # Methodology details
├── PENJELASAN.md                 # Conceptual guide (Indonesian)
├── VALIDATION_INSTRUCTIONS.md    # Manual labelling workflow
└── requirements.txt
```

---

## Setup

### 1. Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Tesseract OCR (PDF fallback)
brew install tesseract           # macOS
sudo apt install tesseract-ocr   # Ubuntu/Debian
```

### 2. API Key

All LLM calls use the **Groq free tier**. One key covers both LLaMA 3.3 70B and Qwen 3 32B.

```bash
# Get a free key at https://console.groq.com
echo "GROQ_API_KEY=your_key_here" > .env
```

> **Free-tier limits for LLaMA 3.3 70B (Groq):**
> 30 requests/minute, ~6,000 tokens/minute, 100,000 tokens/day on a rolling 24-hour window.
>
> A CoT classification of 100 provisions consumes the full daily quota in one run. Plan CoT runs for early in the day so the rolling window is fresh.

### 3. Place Agreement PDFs

Put all FTA PDFs in the `Agreement/` directory. The expected filenames are defined in `config.py` under `AGREEMENTS`.

---

## Running the Pipeline

### Full pipeline (core sequential stages)

```bash
python run_pipeline.py --step all
```

`--step all` runs the four core stages: `extract → embed → classify → compare`. It does not generate the stratified comparison sample or the validation cohort automatically.

### Step by step

#### Core pipeline

```bash
# 1. Extract provisions from PDFs
python run_pipeline.py --step extract

# 1b. Build the agreement-balanced comparison cohort (100 per agreement)
python run_pipeline.py --step stratified_sample --per-agreement 100 --seed 42

# 2. Build the ChromaDB vector store
python run_pipeline.py --step embed

# 3. Classify provisions
#    --model: llama | qwen
#    --strategy: zero_shot | few_shot | cot
#    --limit: number of provisions (default 200)
#    --sample-mode: random | head (use random unless debugging)
python -m src.classification --model llama --strategy zero_shot --limit 200 --sample-mode random --seed 42
python -m src.classification --model qwen  --strategy cot       --limit 100 --sample-mode random --seed 42

# Stratified-sample classification (corrects for corpus imbalance)
python -m src.classification --model qwen --strategy few_shot \
    --source stratified_sample.json --suffix stratified

# 4. Cross-agreement comparison (RAG-augmented LLM analysis)
python run_pipeline.py --step compare --model qwen

# 5. Attribute extraction (RoO numeric fields and categorical flags)
python -m src.attribute_extraction --model qwen

# 6. Statistical analysis (Cohen's κ, convergence signal)
python -m src.analysis

# 7. Generate figures
python -m src.visualize
```

#### Validation workflow (50-provision gold set)

```bash
# 1. Build the validation CSV from a classified cohort
python run_pipeline.py --step validation_sample \
    --validation-n 50 \
    --validation-source classified_qwen_few_shot_stratified.json \
    --seed 42

# 2. Manually label data/results/validation_checked.xlsx
#    (fallback: data/results/validation_set.csv)

# 3. Export the exact validation cohort to JSON for reclassification
python run_pipeline.py --step validation_export

# 4. Reclassify the same cohort with each model and strategy
python -m src.classification --model llama --strategy zero_shot --source validation_provisions.json --suffix validation
python -m src.classification --model llama --strategy few_shot  --source validation_provisions.json --suffix validation
python -m src.classification --model llama --strategy cot       --source validation_provisions.json --suffix validation
python -m src.classification --model qwen  --strategy zero_shot --source validation_provisions.json --suffix validation
python -m src.classification --model qwen  --strategy few_shot  --source validation_provisions.json --suffix validation
python -m src.classification --model qwen  --strategy cot       --source validation_provisions.json --suffix validation

# 5. Score only runs whose IDs match the labelled cohort exactly
python -m src.validation --evaluate
```

### Recommended full rerun after code changes

If extraction, sampling, validation, or comparison logic has changed, the saved JSON outputs may be stale. Rerun in this order:

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

## Models and Prompt Strategies

### Models

| Model | Provider | Parameters | Free Tier |
|-------|----------|-----------|-----------|
| LLaMA 3.3 70B | Groq (Meta) | 70B | 100K tokens/day |
| Qwen 3 32B | Groq (Alibaba Cloud) | 32B | Separate quota |

Qwen 3 is a "thinking" model; it emits `<think>...</think>` reasoning tokens before the final answer. The classification code strips these before recording the category label.

### Prompt Strategies

| Strategy | Description | Tokens/call (approx.) |
|----------|-------------|----------------------|
| `zero_shot` | Category list + provision text; no examples | ~400 |
| `few_shot` | Two labelled examples prepended to the prompt | ~900 |
| `cot` | Instruction to reason step-by-step before labelling | ~1,400 |

### Validation Results (50-provision gold set)

| Model + Strategy | Accuracy | Macro-F1 | n | Note |
|-----------------|----------|----------|---|------|
| LLaMA 3.3 70B, zero-shot | **0.480** | 0.431 | 50 | Best raw accuracy |
| Qwen 3 32B, chain-of-thought | 0.460 | **0.442** | 50 | Best Macro-F1 |
| Qwen 3 32B, zero-shot | 0.380 | 0.424 | 50 |  |
| Qwen 3 32B, few-shot | 0.380 | 0.373 | 50 | Few-shot hurts Qwen |
| LLaMA 3.3 70B, few-shot | 0.340 | 0.336 | 50 |  |
| LLaMA 3.3 70B, chain-of-thought | 0.320 | 0.327 | 50 | CoT hurts LLaMA |

### Inter-Run Agreement (Cohen's κ)

| Pair | n shared | κ | Interpretation |
|------|---------|---|----------------|
| LLaMA zero-shot vs Qwen zero-shot | 200 | 0.702 | Substantial, strongest cross-model alignment |
| Qwen zero-shot vs Qwen few-shot | 200 | 0.689 | Substantial, within-model consistency |
| LLaMA zero-shot vs LLaMA few-shot | 200 | 0.668 | Substantial, within-model consistency |
| LLaMA CoT vs Qwen CoT | 100 | 0.640 | Substantial, CoT aligns both models |
| LLaMA few-shot vs Qwen few-shot | 200 | 0.582 | Moderate, few-shot diverges the models more |

---

## Policy Categories

The 11 classification categories follow the standard WTO/UNCTAD FTA chapter taxonomy:

| Category | Example provision |
|----------|------------------|
| Tariff Commitments | Duty reduction schedules, tariff elimination timelines |
| Rules of Origin | RVC thresholds, Change-in-Tariff-Classification rules |
| Non-Tariff Measures | Import licensing, technical standards, quotas |
| Trade in Services | Mode 1 to 4 service liberalisation commitments |
| Investment | ISDS, national treatment for investors |
| Dispute Settlement | Panel procedures, consultation timelines |
| Customs Procedures | Documentation, advance rulings, single window |
| Sanitary and Phytosanitary | Food safety, plant and animal health standards |
| Intellectual Property | Copyright, trademarks, geographic indicators |
| General Provisions / Definitions | Definitions, scope, general exceptions |
| Other | Provisions not fitting any above category |

---

## Attribute Extraction

Beyond classification, the pipeline extracts structured numeric and categorical attributes from Rules of Origin and Tariff Commitment provisions:

**Rules of Origin attributes:**
- RVC threshold (%), regex
- Change-in-Tariff-Classification rule (`CC` / `CTH` / `CTSH`), LLM
- HS code scope, LLM
- De-minimis threshold (%), regex

**Tariff Commitment attributes:**
- Phase-out years, regex
- Staging category, LLM
- Product scope, LLM

Headline finding: AHKFTA requires Chapter-level Change in Tariff Classification (`CC`) while RCEP requires Heading-level (`CTH`). Both apply a 40% RVC threshold, but AHKFTA imposes stricter transformation requirements.

---

## Iterative Development Notes

This project went through several methodological pivots, documented in the reports:

1. **Gemini to Qwen substitution.** Initial design used Gemini 1.5 Flash. Quota was exhausted mid-run; the architecture was rebuilt around Qwen 3 32B via Groq, sharing the same API key infrastructure as LLaMA.

2. **HS code alignment.** Cross-agreement comparison was originally designed to align provisions by HS code. Sparse article-level metadata made this infeasible; semantic similarity via ChromaDB is used instead.

3. **Stratified sampling.** The raw corpus is 53.5% RCEP provisions. Comparative analyses use a 100-provision stratified sample (seed 42) to prevent RCEP from dominating results.

4. **Annex coverage gap.** Quantitative tariff thresholds cluster in tariff schedules (annexes), which were not segmented into the provision corpus. Numeric attribute extraction is limited to main-text provisions.

---

## Notebooks

`notebooks/analysis.ipynb` is a self-contained walkthrough that:
- Runs the full pipeline end to end (skips steps already completed)
- Reproduces all report figures
- Shows the Cohen's κ heatmap across all run pairs
- Displays the RAG-retrieved cross-agreement comparisons per policy category

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
| `fig_category_x_agreement.png` | Provision count heatmap, category by agreement (stratified sample) |
| `fig_strategy_effect_llama.png` | How prompt strategy shifts category distribution for LLaMA |
| `fig_strategy_effect_qwen.png` | How prompt strategy shifts category distribution for Qwen |
| `fig_convergence.png` | Entropy-based convergence signal per category |
| `fig_validation_accuracy.png` | Accuracy and Macro-F1 by model and strategy |

---

## Limitations

- **Triage-grade accuracy only.** The 32 to 48% validation accuracy range supports analyst prioritisation, not compliance or legal use.
- **Annex coverage.** Tariff schedules are excluded; numeric thresholds in schedules are not captured.
- **API quota constraints.** The Groq free-tier LLaMA daily quota (100K tokens) prevents running full CoT classification and validation in the same day.
- **Three agreements.** Findings are suggestive, not statistically generalisable across the broader FTA landscape.
- **English-only.** All PDFs are English-language versions; translation layers are not included.
- **Investment-category caveat.** Four AHKFTA provisions were classified as investment-adjacent by the model, but AHKFTA does not include a dedicated Investment chapter. The apparent convergence on Investment in the entropy chart is classification noise; the genuine picture is closer to fragmentation.

---

## Citation

> Novith, D. C. (2026). *A Computational Framework for Comparative Analysis of Free Trade Agreements Using Large Language Models*.

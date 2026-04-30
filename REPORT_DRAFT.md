# A Computational Framework for Comparative Analysis of Free Trade Agreements

**Diyouva Christa Novith**
Machine Learning Foundation with Python — Carnegie Mellon University — Spring 2026

---

> **Documentation status (April 2026):** This is a technical draft tied to the
> current saved artefacts. Use `README.md` for the executable rerun workflow.

## Abstract

Free Trade Agreements (FTAs) shape trade flows, tariffs, and regulatory regimes for billions of consumers, yet their legal texts run to thousands of pages and overlap inconsistently across regions. This project builds an end-to-end Python pipeline that converts three major Asia-Pacific FTAs — **RCEP**, **AHKFTA**, and **AANZFTA** — into a structured, machine-comparable dataset of 4,059 provisions, and applies two Large Language Models (**LLaMA 3.3 70B** and **Qwen 3 32B**) under three prompt strategies (**zero-shot**, **few-shot**, **chain-of-thought**) to classify each provision into 11 policy categories. A Retrieval-Augmented Generation (RAG) layer then produces narrative cross-agreement comparisons. We quantify model and strategy agreement with Cohen's κ, identify convergent vs. fragmented policy areas across the three agreements, and validate classification accuracy on a hand-labelled sample of 50 provisions.

## 1. Introduction

### 1.1 Motivation
Asian regional trade policy is often described as a *spaghetti bowl* — overlapping bilateral and plurilateral agreements with differing rules of origin, tariff schedules, and regulatory disciplines. Policymakers and customs authorities who must navigate these documents typically rely on manual legal review, which is slow, expensive, and error-prone.

### 1.2 Research Questions
Three questions, drawn directly from the project proposal:
1. Can FTA provisions be **reliably extracted and classified** into standardised policy categories using LLMs?
2. How do comparable provisions **differ across agreements** in observable policy-design features (rules of origin criteria, legal flexibility, structure of commitments)?
3. Do these agreements exhibit **patterns of convergence or fragmentation** reflecting broader regional trade dynamics?

### 1.3 Contribution
We deliver (a) a reusable pipeline that any researcher can point at new FTA PDFs, (b) a 4,059-provision labelled dataset, (c) a cross-model / cross-strategy agreement study, and (d) an RAG-based comparative analysis over 11 policy areas.

## 2. Data

### 2.1 Sources
Seven publicly available PDFs covering:
- **RCEP** — main agreement, 20 chapters (signed 2020, in force 2022).
- **AHKFTA** — main text + Annexes 2–11 (signed 2017, in force 2019).
- **AANZFTA** — main text, First Protocol (2014), Economic Cooperation Work Programme Implementing Arrangement, Tariff Reduction Understanding (signed 2009, in force 2010).

### 2.2 Corpus statistics (after extraction)

| Agreement | Provisions | Share |
|---|---:|---:|
| RCEP | 2,171 | 53.5% |
| AANZFTA | 1,526 | 37.6% |
| AHKFTA | 362 | 8.9% |
| **Total** | **4,059** | 100% |

![Corpus overview](data/results/fig_corpus_overview.png)

### 2.3 Provision schema

Each extracted provision record contains the following fields, matching the structure specified in the project proposal:

| Field | Description | Example |
|---|---|---|
| `id` | Unique provision identifier | `RCEP_Main Agreement_00457` |
| `agreement` | Agreement name | `RCEP`, `AHKFTA`, `AANZFTA` |
| `doc_type` | Document type | `Main Agreement`, `Protocol`, `Annex` |
| `chapter` | Chapter heading (where present) | `Chapter 3 – Rules of Origin` |
| `article` | Article or rule number (where present) | `Article 3.4` |
| `paragraph_idx` | Sequential paragraph index within document | `457` |
| `text` | Provision text (80–1,500 characters) | *full legal text* |
| `char_count` | Character length | `300` |
| `category` | Assigned policy category (classified files only) | `Rules of Origin` |

The `doc_type` field captures the annex/protocol distinction specified in the proposal (annex or appendix number); specific annex sub-numbers are embedded in the source filename rather than parsed as a separate field. The `article` field is populated where the PDF text layer contains explicit article-level headers; otherwise it defaults to empty string (affects roughly 40% of the corpus due to PDF formatting variation).

## 3. Methodology

See companion file `METHODOLOGY.md` for full technical detail. In summary:

1. **Extraction.** `pdfplumber` → `PyMuPDF` → Tesseract OCR fallback chain segments text by Article / Chapter / Rule / Section regex; minimum clause length 80 chars.
2. **Embedding.** `all-MiniLM-L6-v2` sentence transformer, indexed in a persistent ChromaDB store (cosine distance).
3. **Classification.** Two models × three prompt strategies. Free-tier Groq API with strategy-aware rate limiting (token-per-minute aware delays).
4. **Comparison.** For each of 11 categories, retrieve top-3 similar provisions per agreement; prompt the LLM for a structured 5-point comparison (similarities, differences, flexibility, convergence, implications).
5. **Analysis.** Cohen's κ, category × agreement counts, entropy-based convergence signal.
6. **Validation.** 50-provision hand-labelled validation cohort → exact-cohort reruns → accuracy / macro-F1 per run.

### 3.1 Deviations from proposal
Two implementation choices diverged from the original proposal:

**Model substitution.** The proposal specified LLaMA 3 and **Gemini**. Empirical testing showed Gemini's free-tier daily quota exhausted after ~200 calls — insufficient for the three-strategy × 200-provision matrix. We substituted **Qwen 3 32B**, which is free via the same Groq endpoint and is a different model family (Alibaba Cloud), preserving the cross-architecture comparison intent of the original design.

**Cross-agreement alignment method.** The proposal called for aligning provisions across agreements using *article and rule numbers as primary internal alignment keys, supplemented by Harmonized System codes where applicable*. In practice, many provisions extracted from the PDFs lack clean article-number metadata (the `article` field is empty for a large fraction of the corpus), and HS code references are concentrated in tariff schedules rather than main-agreement text. We therefore used **semantic similarity via ChromaDB** (cosine distance over `all-MiniLM-L6-v2` embeddings) as the cross-agreement retrieval mechanism for the RAG comparison step. This approach is more robust to formatting variation across documents, but cannot guarantee structurally equivalent articles are retrieved — a limitation noted in §5.2.

## 4. Results

### 4.1 Classification coverage

| Run | Provisions | Errors | Note |
|---|---:|---:|---|
| LLaMA 3.3 70B zero-shot | 200 | 0 | |
| LLaMA 3.3 70B few-shot | 200 | 0 (after retry) | |
| LLaMA 3.3 70B CoT | 100 | 0 | Capped at 100: Groq daily token budget |
| Qwen 3 32B zero-shot | 200 | 0 | |
| Qwen 3 32B few-shot | 200 | 0 (after retry) | |
| Qwen 3 32B CoT | 100 | 0 (after retry) | Capped at 100: Groq daily token budget |
| Qwen 3 32B few-shot *(stratified 100×3)* | 300 | 0 | Balanced sample — 100 per agreement |

CoT runs were capped at 100 provisions due to Groq free-tier daily quota: the CoT prompt generates ~800–1,200 tokens of reasoning per provision, consuming the daily allocation roughly twice as fast as zero-shot. All six main-run classifications used randomly sampled provisions from `all_provisions.json`; the stratified run used `stratified_sample.json` (seed 42) to correct for RCEP's 53% share of the corpus.

### 4.2 Category distribution per run

![Category heatmap](data/results/fig_category_heatmap.png)

Notable patterns:
- LLaMA few-shot heavily favours *Rules of Origin* (31.5%); Qwen zero-shot leans toward *Tariff Commitments* (33%).
- *Intellectual Property* and *SPS* are rare in both models — partly a true scarcity (< 3% of the corpus) and partly misclassification into *Other* (35% in some runs).
- Qwen few-shot pushes more provisions to *Other* than zero-shot, suggesting the two in-prompt examples make the model more conservative.

### 4.3 Model & strategy agreement (Cohen's κ)

![Kappa matrix](data/results/fig_kappa_matrix.png)

Key findings:
- **Within-model strategy consistency is materially stronger in the current rerun artefacts.** LLaMA zero-shot vs. few-shot: κ = 0.668. Qwen zero-shot vs. few-shot: κ = 0.689.
- **Across-model agreement is no longer near-zero once cohorts are aligned exactly.** LLaMA few-shot vs. Qwen few-shot: κ = 0.582.
- Implication: category boundaries under these prompts are noisy; downstream analysis should rely on *aggregate* distribution trends rather than individual labels.

### 4.4 Effect of prompt strategy (within each model)

| | |
|---|---|
| ![LLaMA](data/results/fig_strategy_effect_llama.png) | ![Qwen](data/results/fig_strategy_effect_qwen.png) |

Adding examples (few-shot) consistently *reduces* the *Tariff Commitments* label count and *increases* *Other*, for both models — the examples appear to raise the bar for positive classification.

### 4.5 Cross-agreement comparison (RQ2)

The stratified sample (100 provisions per agreement, seed 42, classified with Qwen 3 32B few-shot) lets us compare *how* each agreement allocates attention across categories on a balanced basis.

#### 4.5a Provision count matrix

The table below is the primary quantitative comparative matrix requested by the proposal — it shows how each agreement's 100-provision stratified sample distributes across all 11 policy categories.

| Category | RCEP | AHKFTA | AANZFTA | **Total** |
|---|---:|---:|---:|---:|
| Rules of Origin | 24 | 48 | 39 | **111** |
| Tariff Commitments | 6 | 31 | 7 | **44** |
| Trade in Services | 20 | 0 | 10 | **30** |
| Dispute Settlement | 6 | 0 | 22 | **28** |
| Customs Procedures | 6 | 6 | 7 | **19** |
| General Provisions / Definitions | 5 | 9 | 3 | **17** |
| Investment | 8 | 4 | 5 | **17** |
| Intellectual Property | 12 | 0 | 2 | **14** |
| Non-Tariff Measures | 6 | 1 | 4 | **11** |
| Sanitary and Phytosanitary Measures | 5 | 1 | 0 | **6** |
| Other | 2 | 0 | 1 | **3** |
| **Total** | **100** | **100** | **100** | **300** |

![Category × agreement](data/results/fig_category_x_agreement.png)

Key observations:
- **Rules of Origin** is the dominant category (111/300, 37%), with AHKFTA showing extreme concentration (48/100 vs. RCEP 24/100) — AHKFTA devotes nearly half its sampled provisions to origin rules, reflecting its goods-only mandate and rule-intensive preferential treatment framework.
- **Tariff Commitments** is the second-largest category (44/300, 15%), again concentrated in AHKFTA (31%) versus RCEP (6%) — AHKFTA's tariff liberalisation architecture is correspondingly more text-heavy.
- **AHKFTA has zero provisions** in Trade in Services, Dispute Settlement, and Intellectual Property in this sample — reflecting its narrower scope as a goods-focused bilateral FTA. This is a structural absence, not a sampling artefact.
- **AANZFTA leads in Dispute Settlement** (22/28, 79% of all dispute settlement provisions in the sample), reflecting its dedicated dispute resolution chapter. RCEP has only 6 and AHKFTA zero.
- **Other** absorbs only 3 provisions (1%), indicating the 11-category taxonomy is comprehensive for FTA content at the main-agreement level. Concentration risk is in Rules of Origin, partly amplified by the few-shot in-context examples being goods-trade categories.

#### 4.5b Policy-design comparison matrix

The table below synthesises the qualitative RAG comparison (`comparison_qwen.json`) into a structured feature matrix, directly addressing **RQ2** (how comparable provisions differ in observable policy-design features).

| Feature | RCEP | AHKFTA | AANZFTA |
|---|---|---|---|
| **Tariff amendment mechanism** | Consensus + formal procedure; unilateral importer notification | HS 2012 reference; product-specific exporter choice | Unilateral modification rights; selective HS concessions |
| **Rules of Origin governance** | Committee (Annex 3A/3B); CTH (heading-level) rule | Sub-Committee; CC (chapter-level) rule — stricter transformation | Certificate-based; institutionalized exporter compliance |
| **RVC threshold** | 40% | 40% | Not recovered in main-text sample |
| **Non-tariff measures** | Art. 8.4 explicit QR ban; Goods Committee | WTO-aligned prohibition; transparency focus | Procedural facilitation; Goods Committee review |
| **Dispute settlement** | Own mechanism; not tied to WTO DSU | WTO DSU as reference framework | Own mechanism; adjudication / arbitration |
| **Investment chapter** | Dedicated Ch. 10; national treatment | Not present (goods-focused) | Dedicated Ch. 11; national treatment |
| **Customs procedures** | Direct consignment rules | No legalization / authentication required | Risk-based approach; low-risk clearance |
| **SPS measures** | Provisional measures; monitoring mechanism | WTO SPS alignment; limited standalone provisions | WTO SPS alignment; formal subcommittees |

![Convergence signal](data/results/fig_convergence.png)

The entropy-based convergence signal (green = all three agreements allocate attention in comparable proportion; orange = fragmented) shows *Customs Procedures* as the most convergent category (normalised entropy = 1.00), followed by *Investment* (0.96) and *Rules of Origin* (0.97, though this is partly inflated by few-shot bias). *Dispute Settlement* (0.47), *Intellectual Property* (0.37), and *SPS Measures* (0.41) are the most fragmented — driven primarily by AHKFTA's zero-provision allocation to these categories. Each agreement follows a distinct structural template in those areas.

### 4.6 Qualitative RAG comparison — selected excerpts
Below are abridged excerpts from the LLM-generated cross-agreement comparison (`comparison_qwen.json`):

> **Tariff Commitments.** All three agreements establish mutual amendment mechanisms for tariff schedules, requiring consensus or formal procedures. They also emphasise non-discrimination by extending improved tariff concessions to all parties when bilateral upgrades occur. RCEP uniquely allows unilateral notification requirements for importers seeking preferential treatment, a provision absent in the others.

> **Rules of Origin.** All three agreements prioritise RoO as the core mechanism to determine preferential treatment and include provisions for modifying RoO criteria. RCEP emphasises a committee-based governance structure, AHKFTA works through a Sub-Committee, and AANZFTA institutionalises certificate requirements — a structural choice that places the compliance burden directly on exporters.

*[Full analyses in `data/results/comparison_qwen.json`.]*

### 4.7 Structured attribute extraction on Rules of Origin & Tariff Commitments

A hybrid regex + LLM pass was run over 92 stratified provisions (38 Rules of Origin, 54 Tariff Commitments) classified by Qwen 3 32B few-shot. Regex handles deterministic numeric fields (RVC %, de-minimis %, phase-out years); the LLM fills categorical flags (CTC rule family, HS scope, staging category, cumulation type).

**Rules of Origin — key numeric features recovered**

| Agreement | RVC threshold | De-minimis | Dominant CTC rule |
|---|---|---|---|
| RCEP | 40% (n=3) | — | CTH — change in tariff heading |
| AHKFTA | 40% (n=1) | 10% | CC — change of chapter |
| AANZFTA | *not recovered in sample* | — | — |

**Tariff Commitments — coverage patterns**

| Agreement | HS scope | Phase-out signal |
|---|---|---|
| RCEP | `all` (broad, across-the-board commitments) | 5-year instalment reference found |
| AHKFTA | `selected` (targeted concession list, n=9) | staging categories `iad`, `initial` |
| AANZFTA | `selected` (n=5) | max reduction 10% reference |

Policy-design observations directly answering **RQ2**:
- RCEP and AHKFTA share the ASEAN-standard **40% regional value content** bar for RVC-qualifying goods, but choose different default CTC rules — RCEP is heading-level (CTH), AHKFTA is chapter-level (CC), implying AHKFTA demands a stricter transformation for origin.
- AHKFTA and AANZFTA use **selective HS-code concessions** (specific subheadings phased in under named staging categories); RCEP's tariff text references broader *across-the-board* commitments — consistent with RCEP's role as a regional liberalisation umbrella.
- Absence of recovered AANZFTA RVC/de-minimis numerics is a reminder that structured thresholds in legal text cluster in *schedules and annexes*, not in the main provisions the classifier mostly hits — a known limitation; future work should pass annex-level subprovisions through the attribute extractor separately.

Full per-provision attribute records are in `data/results/attributes_qwen_few_shot_stratified.json`.

### 4.8 Validation against manual labels

A stratified random sample of 50 provisions (18 RCEP, 16 AHKFTA, 16 AANZFTA) was hand-labelled by the project author to produce a gold-standard `gold_category`. Each provision was then classified by every model–strategy combination and scored against the gold labels.

| Run | n | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| LLaMA 3.3 70B — zero-shot | 50 | **0.480** | 0.431 |
| Qwen 3 32B — chain-of-thought | 50 | 0.460 | **0.442** |
| Qwen 3 32B — zero-shot | 50 | 0.380 | 0.424 |
| Qwen 3 32B — few-shot | 50 | 0.380 | 0.373 |
| LLaMA 3.3 70B — few-shot | 50 | 0.340 | 0.336 |
| LLaMA 3.3 70B — chain-of-thought | 50 | 0.320 | 0.327 |

![Validation accuracy](data/results/fig_validation_accuracy.png)

Key findings:
- **No run clears 50% accuracy on the current gold set.** That makes the classifier useful for triage and aggregate patterning, but not for trusted provision-level automation.
- **LLaMA zero-shot has the best accuracy** at 48.0%, while **Qwen CoT has the best macro-F1** at 0.442. There is no longer a single dominant "winner" across both metrics.
- Prompt strategy effects remain real but smaller than earlier drafts claimed: CoT helps Qwen slightly on macro-F1, while zero-shot remains strongest for LLaMA.
- The more important result now is methodological: once cohorts are aligned exactly, inter-run agreement is fairly healthy, so the bottleneck is label quality, not pairwise instability.

## 5. Discussion

### 5.1 What works
- Extraction is robust even on scanned PDFs (AHKFTA) thanks to the OCR fallback.
- The RAG comparison produces genuinely useful comparative narratives when the model is given 2K output tokens.
- Convergence signals (entropy-based) are interpretable: *Customs Procedures* is the most convergent category; *Intellectual Property* and *Dispute Settlement* are the most fragmented (driven by AHKFTA's zero-provision allocation).

### 5.2 Limitations
1. **Category boundaries.** Validation remains weak even after the pipeline fixes, which suggests the 11-way taxonomy is still difficult for these prompts. Hierarchical or multi-label classification would likely help.
2. **Single-annotator ground truth.** The 50-provision validation is labelled by the project author; future work should add a second annotator and compute inter-annotator κ.
3. **Category imbalance.** Seven of eleven categories have < 10% of provisions; macro-F1 is therefore sensitive to noise in the tail.
4. **Cross-agreement alignment.** The proposal called for article/rule-number alignment supplemented by HS codes. In practice, extracted provisions often lack clean article-number metadata, and HS codes appear mainly in tariff schedules rather than main-agreement text. Semantic similarity (ChromaDB) was used instead, which is robust to formatting variation but cannot guarantee structurally equivalent provisions are retrieved — the RAG comparison may occasionally pair provisions that address related but not identical obligations.
5. **Annex and schedule coverage.** The attribute extractor recovers numeric thresholds (RVC%, phase-out years) from main-agreement text; most tariff schedules and product-specific-rules annexes were not passed through the attribute extractor separately, so threshold recovery is incomplete for AANZFTA.
6. **API quota constraints.** The Groq free-tier daily token limit (100,000 tokens/day for LLaMA 3.3 70B) still constrains the main classification runs (CoT capped at 100 provisions instead of 200). A paid tier or second API key would improve throughput, but would not by itself solve the current 0.320–0.480 validation accuracy ceiling.

### 5.3 Policy implications
A working computational pipeline means a trade negotiator can, in principle, ask questions like *"Which agreement is most flexible on de-minimis rules?"* and get an evidence-cited answer in seconds rather than days. The accuracy bar is not yet high enough for compliance-grade use, but it is clearly high enough for triage / prioritisation of manual review.

## 6. Conclusion
We successfully converted 4,059 FTA provisions into a structured corpus and addressed all three research questions from the proposal:

**RQ1 (Reliable extraction and classification):** The pipeline extracts provisions successfully across all 7 PDFs including scanned documents (OCR fallback), but current validation quality is only moderate: accuracy ranges from 32% to 48%, and macro-F1 ranges from 0.327 to 0.442. The best accuracy comes from LLaMA zero-shot; the best macro-F1 comes from Qwen CoT. That makes the system useful for triage and corpus-level patterning, not provision-level automation.

**RQ2 (Policy-design differences across agreements):** The provision count matrix (§4.5a) and policy-design comparison matrix (§4.5b) reveal clear structural differences: AHKFTA concentrates 48% of its text in Rules of Origin vs. 24% for RCEP; AHKFTA has zero provisions in Dispute Settlement, Trade in Services, and Intellectual Property; and the three agreements use different default CTC rules and tariff-concession architectures.

**RQ3 (Convergence or fragmentation):** Entropy-based signals confirm that Customs Procedures is the most convergent category, while Dispute Settlement and Intellectual Property are the most fragmented — driven by AHKFTA's zero-provision allocation to those chapters, reflecting its goods-only mandate.

The framework is fully reproducible, uses only free-tier APIs, and generalises to any FTA with a PDF text layer. Qwen 3 32B with chain-of-thought is the recommended classifier for future work.

---

## Appendix A — Run commands

```bash
# 1. Extract provisions
python -m src.extraction

# 2. Build vector store
python -m src.embedding

# 3. Classify — main runs (200 provisions each; CoT capped at 100 for quota)
python -m src.classification --model llama --strategy zero_shot --limit 200
python -m src.classification --model llama --strategy few_shot  --limit 200
python -m src.classification --model llama --strategy cot       --limit 100
python -m src.classification --model qwen  --strategy zero_shot --limit 200
python -m src.classification --model qwen  --strategy few_shot  --limit 200
python -m src.classification --model qwen  --strategy cot       --limit 100

# 3b. Stratified sample across agreements (fixes RCEP-heavy bias)
python -m src.classification --model qwen --strategy few_shot \
    --source stratified_sample.json --suffix stratified

# 3c. Classify validation set with all model–strategy combinations
python -m src.classification --model llama --strategy zero_shot \
    --source validation_provisions.json --suffix validation
python -m src.classification --model llama --strategy few_shot  \
    --source validation_provisions.json --suffix validation
python -m src.classification --model llama --strategy cot       \
    --source validation_provisions.json --suffix validation
python -m src.classification --model qwen  --strategy zero_shot \
    --source validation_provisions.json --suffix validation
python -m src.classification --model qwen  --strategy few_shot  \
    --source validation_provisions.json --suffix validation
python -m src.classification --model qwen  --strategy cot       \
    --source validation_provisions.json --suffix validation

# 4. Cross-agreement comparison (RAG)
python -m src.comparison --model qwen

# 5. Analysis and visualisation
python -m src.analysis
python -m src.visualize

# 6. Attribute extraction
python -m src.attribute_extraction --model qwen --strategy few_shot --suffix stratified \
    --source stratified_sample.json

# 7. Validation scoring
python run_pipeline.py --step validation_sample \
    --validation-n 50 \
    --validation-source classified_qwen_few_shot_stratified.json \
    --seed 42
#   → fill data/results/validation_checked.xlsx (fallback: validation_set.csv)
python run_pipeline.py --step validation_export
python -m src.validation --evaluate
```

## Appendix B — Repository layout

```
Final Project - FTA LLM/
├── Agreement/                 # source PDFs (not tracked)
├── config.py                  # central configuration
├── run_pipeline.py            # orchestration CLI + dataset-prep entrypoints
├── src/
│   ├── extraction.py
│   ├── embedding.py
│   ├── classification.py
│   ├── comparison.py
│   ├── attribute_extraction.py
│   ├── analysis.py
│   ├── validation.py
│   └── visualize.py
├── notebooks/analysis.ipynb   # exploratory + figures
├── data/
│   ├── raw/                   # all_provisions.json, stratified_sample.json
│   ├── processed/
│   ├── results/               # classified_*.json, comparison_*.json, fig_*.png
│   └── chromadb/              # persistent vector store
├── METHODOLOGY.md
└── REPORT_DRAFT.md            # this file
```

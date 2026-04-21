# A Computational Framework for Comparative Analysis of Free Trade Agreements

**Diyouva Christa Novith**
Machine Learning Foundation with Python — Carnegie Mellon University — Spring 2026

---

## Abstract

Free Trade Agreements (FTAs) run to thousands of pages yet must be compared, monitored, and negotiated by policy analysts who rely almost entirely on manual review. This project builds an end-to-end Python pipeline that converts three major Asia-Pacific FTAs — RCEP, AHKFTA, and AANZFTA — into a structured, machine-readable dataset of 3,980 provisions and uses Large Language Models to classify, compare, and extract attributes from them. Two models (LLaMA 3.3 70B and Qwen 3 32B) were tested under three prompt strategies (zero-shot, few-shot, and chain-of-thought) through an iterative process that closely mirrors how a policy analyst would decide which tool to trust. The best configuration — Qwen 3 32B with chain-of-thought prompting — achieves 70% accuracy and a macro-F1 of 0.693 on a 50-provision hand-labelled gold set. The pipeline surfaces concrete policy-design differences across agreements and identifies which policy areas are converging across the region and which remain fragmented.

---

## 1. The Policy Problem

A trade negotiator at a Southeast Asian trade ministry receives a request: *"How do our rules-of-origin provisions compare to RCEP's? Do we have the same de-minimis threshold? Are our staging categories for tariff phase-outs aligned?"* On a good day, answering this question takes a week of cross-referencing legal texts. On a bad day, it takes longer — or the answer is wrong because a key provision in Annex 3B was missed.

This is the *spaghetti bowl* problem. Asia-Pacific trade policy is governed by dozens of overlapping bilateral and plurilateral agreements, each running hundreds or thousands of pages with slightly different architectures, numbering schemes, and legal language. RCEP alone covers 15 parties and 20 chapters. AHKFTA covers goods trade between ASEAN and Hong Kong, but with a different rules-of-origin framework, different staging categories, and different dispute settlement references than either RCEP or AANZFTA. Comparing them manually is slow, error-prone, and expensive.

**This project asks three questions:**

1. Can LLMs reliably extract and classify FTA provisions into standardised policy categories?
2. How do comparable provisions differ across agreements in observable design features (thresholds, flexibility, governance structures)?
3. Do these agreements show patterns of convergence or fragmentation across the region?

The target user is a policy analyst or trade negotiator who needs structured, evidence-backed answers in hours rather than weeks — and who needs to know which answers to trust.

---

## 2. Data

Three publicly available FTAs were selected to span the key Asia-Pacific regional architecture:

| Agreement | Parties | In force | Main documents used |
|---|---|---|---|
| **RCEP** | 15 nations including China, Japan, ASEAN | 2022 | Main agreement (20 chapters) |
| **AHKFTA** | ASEAN + Hong Kong, China | 2019 | Main text + Annexes 2–11 |
| **AANZFTA** | ASEAN + Australia + NZ | 2010 (amended 2014) | Main text, First Protocol, Implementing Arrangement, Tariff Understanding |

Seven PDFs were processed. AHKFTA is partially scanned, requiring OCR. After extraction, the corpus contains **3,980 provisions** — the smallest meaningful legal unit (an article, paragraph, or structured annex entry), filtered to 80–1,500 characters.

| Agreement | Provisions | Share |
|---|---:|---:|
| RCEP | 2,129 | 53.5% |
| AANZFTA | 1,498 | 37.6% |
| AHKFTA | 353 | 8.9% |
| **Total** | **3,980** | 100% |

Each provision is stored with nine metadata fields: `id`, `agreement`, `doc_type`, `chapter`, `article`, `paragraph_idx`, `text`, `char_count`, and (after classification) `category`. The imbalance toward RCEP reflects its greater scope and was corrected in the cross-agreement analysis by drawing a **stratified sample** of 100 provisions per agreement.

![Corpus overview](data/results/fig_corpus_overview.png)

---

## 3. Methodology — and What We Learned Along the Way

### 3.1 Pipeline Overview

The pipeline has six stages:

1. **Extraction** — `pdfplumber` → `PyMuPDF` → Tesseract OCR fallback. The fallback was essential: AHKFTA's scanned pages produced zero output from the first two tools.
2. **Embedding** — `all-MiniLM-L6-v2` sentence transformer → ChromaDB vector store. This enables semantic retrieval of similar provisions across agreements.
3. **Classification** — LLM labels each provision into one of 11 categories (Tariff Commitments, Rules of Origin, Non-Tariff Measures, Trade in Services, Investment, Dispute Settlement, Intellectual Property, Customs Procedures, SPS, General Provisions, Other).
4. **Comparison** — A Retrieval-Augmented Generation (RAG) step fetches the top-3 semantically similar provisions per agreement per category and prompts the LLM to write a structured comparative analysis.
5. **Attribute Extraction** — A hybrid regex + LLM pass extracts structured fields (RVC%, de-minimis %, phase-out years, CTC rule family) from classified Rules of Origin and Tariff provisions.
6. **Validation** — A stratified 50-provision sample was hand-labelled by the author; every model-strategy combination was scored against this gold set.

### 3.2 The Iterative Model Selection Process

**First attempt: Gemini.** The original plan was to compare LLaMA 3 against Google's Gemini. Gemini was tested first because of its reputation for instruction-following on structured tasks. It performed well — but its free-tier quota ran out after approximately 200 API calls, making a full three-strategy × 200-provision matrix infeasible. This was the first real constraint encountered and shaped everything after.

**Switch to Qwen 3 32B.** Qwen 3 32B (Alibaba Cloud) is available on the same Groq endpoint as LLaMA, on a separate daily quota. It is a different model family and architecture — making it a genuine cross-architecture comparison rather than just a size comparison. The switch preserved the original intent of the proposal: compare at least two architecturally distinct models.

**Three prompt strategies, each revealing something different:**

*Zero-shot* was the baseline. The prompt was simple: here are 11 categories, here is a provision, what category is it? The model produced a label on every call, but with surprising variation — LLaMA and Qwen disagreed on the same provision more often than they agreed (cross-model κ = −0.02 on the few-shot runs, near-random).

*Few-shot* added two curated examples: one clearly labeled Rules of Origin provision and one clearly labeled Tariff Commitments provision. The hypothesis was that examples would sharpen the categories. The result was mixed. LLaMA's macro-F1 improved (0.591 → 0.635). Qwen's *dropped* (0.596 → 0.540) — the two examples over-anchored Qwen onto the two example categories, suppressing correct labels for Dispute Settlement, Customs, and Services provisions. This was not obvious before running the validation.

*Chain-of-thought (CoT)* asked the model to reason step-by-step before committing to a category. For Qwen, this was the decisive improvement: macro-F1 rose from 0.596 (zero-shot) to 0.693 (CoT), the largest single gain in the study. The CoT reasoning traces showed the model genuinely working through ambiguous provisions — for example, correctly identifying that a passage about "certificate of origin" requirements belonged in Customs Procedures (not Rules of Origin) because the obligation was procedural rather than definitional.

**Why CoT worked for Qwen but few-shot didn't** is a real finding, not a footnote. Few-shot examples constrain the model's output space toward the examples. CoT expands the reasoning space — it lets the model use its training knowledge about trade law without being anchored to two specific examples. Legal-text classification is a domain where this distinction matters: the categories are defined by legal function, not surface vocabulary, so broader reasoning outperforms narrow examples.

### 3.3 Key Design Decisions

**Stratified vs. random sampling.** RCEP makes up 53.5% of the corpus. Random samples heavily over-represent RCEP. The cross-agreement comparison used a stratified 100-per-agreement sample to produce a fair comparative matrix — an important methodological choice for the validity of the §4 findings.

**Semantic similarity for cross-agreement alignment.** The original proposal called for aligning provisions using article numbers and HS codes. In practice, ~40% of extracted provisions have empty `article` fields (PDF formatting variation), and HS codes cluster in tariff schedules, not main-agreement text. Semantic similarity via ChromaDB was used instead. This is more robust to formatting variation but does not guarantee that the provisions retrieved for comparison are legally equivalent — an important caveat for the RAG outputs.

**Hybrid attribute extraction.** Numeric fields (RVC thresholds, de-minimis percentages, phase-out years) were extracted with regex — deterministic and verifiable. Categorical flags (CTC rule family, HS scope) were extracted with an LLM JSON prompt. The hybrid approach is faster and more auditable than pure LLM extraction for numeric fields.

---

## 4. Results

### 4.1 How Models and Strategies Compare

Before trusting any policy finding, we need to know how reliable the classifications are — and whether two models agree at all.

![Kappa matrix](data/results/fig_kappa_matrix.png)

Cohen's κ measures inter-run agreement (1.0 = perfect, 0 = chance, negative = worse than chance):

| Comparison | κ | Interpretation |
|---|---|---|
| LLaMA zero-shot vs. few-shot | 0.51 | Moderate consistency |
| Qwen zero-shot vs. few-shot | 0.02 | Near-random |
| LLaMA few-shot vs. Qwen few-shot | −0.02 | Effectively no agreement |
| LLaMA zero-shot vs. Qwen zero-shot | 0.26 | Fair |

The practical implication: **individual provision labels are noisy.** No single model-strategy combination should be trusted for compliance-grade decisions. The right use is aggregate distribution analysis — asking "how many provisions in each agreement address Rules of Origin?" rather than "is this specific provision a Rules of Origin provision?"

![Validation accuracy](data/results/fig_validation_accuracy.png)

Against the 50-provision gold set:

| Model & Strategy | Accuracy | Macro-F1 |
|---|---:|---:|
| **Qwen 3 32B — chain-of-thought** | **0.700** | **0.693** |
| LLaMA 3.3 70B — zero-shot | 0.700 | 0.591 |
| LLaMA 3.3 70B — few-shot | 0.680 | 0.635 |
| Qwen 3 32B — zero-shot | 0.680 | 0.596 |
| Qwen 3 32B — few-shot | 0.580 | 0.540 |

Qwen CoT wins on macro-F1 — the metric that penalises a model for ignoring rare categories. This matters for policy analysis: a model that only gets Tariff Commitments right but misses Intellectual Property or SPS provisions is less useful than one that handles the full taxonomy. Macro-F1 captures that. The 70% accuracy ceiling is consistent with the expected difficulty of legal-text classification; published studies on legislative text classification typically report 65–80% with similar annotation setups.

### 4.2 How Agreements Differ — The Comparative Matrices

The core deliverable for policy analysis: a structured comparison of what each agreement actually covers.

**Provision count matrix** (Qwen few-shot, stratified 100-per-agreement sample):

| Category | RCEP | AHKFTA | AANZFTA | Total |
|---|---:|---:|---:|---:|
| Tariff Commitments | 16 | 35 | 24 | **75** |
| Rules of Origin | 9 | 28 | 8 | **45** |
| Other | 30 | 23 | 26 | **79** |
| Non-Tariff Measures | 9 | 1 | 7 | **17** |
| Trade in Services | 12 | 0 | 5 | **17** |
| Customs Procedures | 7 | 8 | 8 | **23** |
| Investment | 3 | 2 | 9 | **14** |
| Dispute Settlement | 9 | 0 | 10 | **19** |
| General Provisions | 1 | 3 | 1 | **5** |
| Intellectual Property | 4 | 0 | 1 | **5** |
| Sanitary & Phytosanitary | 0 | 0 | 1 | **1** |
| **Total** | **100** | **100** | **100** | **300** |

![Category × agreement](data/results/fig_category_x_agreement.png)

Three patterns stand out:

- **AHKFTA is a goods-focused agreement.** It has zero provisions in Trade in Services, Dispute Settlement, and Intellectual Property in this sample — consistent with its design as a bilateral goods-trade instrument between ASEAN and Hong Kong. RCEP and AANZFTA both have dedicated chapters for services and investment; AHKFTA effectively does not.

- **Rules of Origin weight diverges sharply.** AHKFTA devotes 28% of its sample to Rules of Origin provisions; RCEP devotes only 9%. For a customs officer or compliance manager, this signals that AHKFTA's preferential-treatment framework is more rule-intensive per unit of text — more conditions to satisfy.

- **Investment is predominantly an AANZFTA feature.** AANZFTA has a dedicated investment chapter (Chapter 11); AHKFTA does not. This is a scope difference that directly affects investors choosing between agreements for structuring their market access.

**Policy-design feature matrix** (synthesised from RAG comparison, `comparison_qwen.json`):

| Feature | RCEP | AHKFTA | AANZFTA |
|---|---|---|---|
| Tariff amendment mechanism | Consensus + formal procedure; unilateral importer notification | HS 2012 reference; product-specific exporter choice | Unilateral modification rights; selective HS concessions |
| Rules of Origin governance | Committee (Annex 3A/3B); CTH rule | Sub-Committee; CC rule — stricter transformation | Certificate-based; exporter compliance burden |
| RVC threshold | 40% | 40% | Not recovered in main text |
| Dispute settlement | Own mechanism; independent of WTO DSU | WTO DSU as explicit reference | Own mechanism; adjudication / arbitration |
| Investment coverage | Dedicated Ch. 10; national treatment | Not present | Dedicated Ch. 11; national treatment |
| Customs procedures | Direct consignment requirements | No legalization / authentication required | Risk-based clearance for low-risk goods |

One observation worth highlighting for policy: **RCEP and AHKFTA both use a 40% Regional Value Content threshold for rules of origin** — the ASEAN standard. But they choose different Change in Tariff Classification rules: RCEP uses change at the heading level (CTH), while AHKFTA requires a change at the chapter level (CC). A chapter-level change requires more substantial transformation of the goods. This means the same goods can qualify for preferential treatment under RCEP but not AHKFTA — a real compliance risk for exporters operating under both agreements simultaneously.

### 4.3 Convergence and Fragmentation

The entropy-based convergence signal asks: across the three agreements, which policy areas allocate attention in similar proportions, and which diverge?

![Convergence signal](data/results/fig_convergence.png)

- **Most convergent:** General Provisions / Definitions and Dispute Settlement. All three agreements devote comparable shares of their text to these foundational chapters, suggesting a shared regional template.
- **Most fragmented:** Tariff Commitments and Rules of Origin. Each agreement follows a distinct design template — RCEP's broad liberalisation commitments, AHKFTA's product-specific rule-intensity, AANZFTA's selective HS concessions.

This matters for regional integration policy: convergence in Dispute Settlement suggests there is an emerging regional norm that could be built on. Fragmentation in Rules of Origin suggests that cumulation and origin-stacking across agreements will remain a compliance burden for the foreseeable future — exactly the spaghetti bowl problem that motivates this work.

### 4.4 Attribute-Level Findings

A hybrid regex + LLM pass extracted structured numeric attributes from 92 classified provisions (38 Rules of Origin, 54 Tariff Commitments).

**Rules of Origin numerics:**

| Agreement | RVC threshold | De-minimis | Dominant CTC rule |
|---|---|---|---|
| RCEP | 40% | — | CTH (heading-level) |
| AHKFTA | 40% | 10% | CC (chapter-level) — stricter |
| AANZFTA | *not recovered in main text* | — | — |

**Tariff Commitments patterns:**

| Agreement | HS scope | Phase-out signal |
|---|---|---|
| RCEP | Broad (`all`) | 5-year instalment reference |
| AHKFTA | Selective (specific HS subheadings) | Staging categories `iad`, `initial` |
| AANZFTA | Selective (specific HS subheadings) | Max reduction 10% reference |

The absence of AANZFTA RVC numerics is itself a finding: the main-agreement text delegates threshold definitions to annexes and schedules, which were not passed through the attribute extractor. This is a known gap — and a direction for future work.

---

## 5. Discussion

### 5.1 What Worked

The OCR fallback made the difference for AHKFTA. Without it, one of the three agreements would have been excluded entirely. The three-library extraction chain (`pdfplumber` → `PyMuPDF` → Tesseract) added complexity but preserved corpus completeness.

The RAG comparison produced genuinely readable outputs. When the model was given 2K output tokens and three retrieved provisions per agreement per category, the comparative narratives were specific enough to cite in a briefing memo. The Dispute Settlement comparison correctly identified that AHKFTA references the WTO DSU while RCEP and AANZFTA maintain independent mechanisms — a real legal distinction that manual review would also surface.

The iterative prompt engineering process produced clear evidence for CoT over few-shot on Qwen. Running all three strategies and scoring them against gold labels meant we could make an evidence-based recommendation rather than rely on intuition.

### 5.2 What Didn't Work — and Why It Matters

**Gemini ran out of quota.** The original design relied on Gemini as the second model. After ~200 API calls the free-tier daily limit was exhausted, and a three-strategy × 200-provision matrix requires ~1,200 calls minimum. This forced a model switch mid-project and illustrates a real constraint in production: free-tier LLM APIs are not suitable for large-batch legal-document processing without paid access.

**Low inter-model agreement is a serious limitation.** A cross-model κ of −0.02 between LLaMA few-shot and Qwen few-shot means the two models are essentially labelling independently of each other. Neither can serve as a gold standard for the other. For a policy tool in production, this would require either human-in-the-loop review or ensemble voting — neither of which was implemented here.

**26% of provisions fall into "Other."** The 11-category taxonomy captures most substantive content, but roughly one in four provisions is classified as "Other." Many of these are procedural, transitional, or administrative provisions that genuinely don't fit a substantive policy category. A hierarchical or multi-label taxonomy would likely improve coverage without sacrificing precision.

**Structured thresholds live in annexes, not main text.** The attribute extractor was run on main-agreement provisions. The 40% RVC threshold shows up in the text; AANZFTA's equivalent is in a schedule. Future work should pass schedule-level subprovisions through the attribute extractor separately.

### 5.3 What This Means for Policy Practice

A trade policy analyst using this pipeline today could:

- **In under an hour:** Extract all provisions from a new FTA PDF, embed them, and run the classification to get a category distribution. This replaces 2–3 days of manual triage.
- **With 70% confidence:** Accept the category label on any individual provision. That is triage-grade, not compliance-grade — enough to prioritise which sections to read first, not enough to rely on for a formal legal opinion.
- **Immediately useful:** The comparative matrix and RAG narratives are specific enough to brief a negotiator on key structural differences across agreements. The finding that AHKFTA uses a stricter CTC rule than RCEP (CC vs. CTH) is actionable for an exporter choosing which agreement to invoke.

The tool is not ready for autonomous use. The accuracy ceiling, single-annotator gold labels, and inter-model disagreement all warrant caution. The correct framing is: this reduces the time a skilled analyst spends on triage from weeks to hours, and flags where the agreements differ in a structured format that the analyst can then verify against primary sources.

---

## 6. Conclusion

This project demonstrates that an LLM-based pipeline can extract, classify, and comparatively analyse FTA provisions at a scale and speed that manual review cannot match. Three key findings stand out:

**Model and prompt engineering decisions matter more than expected.** Qwen's few-shot performance was *worse* than its zero-shot baseline; its chain-of-thought performance was the best result in the study. LLaMA showed the opposite pattern on few-shot. These differences were discovered through systematic validation — not apparent from reading the model documentation or testing on one or two examples.

**The agreements are structurally more different than their shared ASEAN context suggests.** AHKFTA is a goods-only instrument that lacks services, investment, and dispute settlement chapters. It shares RCEP's 40% RVC threshold but uses a stricter transformation rule. AANZFTA delegates threshold definitions to annexes in a way that makes automatic extraction harder. These are real differences with real compliance implications, and the pipeline surfaces them in a structured format.

**Tariff Commitments and Rules of Origin are the fragmented core of regional integration.** These are also the provisions exporters and customs authorities care most about. The finding that these areas are the most divergent across agreements — while Dispute Settlement and General Provisions are converging — suggests that procedural harmonisation is progressing faster than substantive liberalisation in the Asia-Pacific.

The pipeline is fully reproducible, uses only free-tier APIs, and can be pointed at any new FTA PDF. The recommended configuration is Qwen 3 32B with chain-of-thought prompting.

---

## Appendix A — Technical Pipeline

```
PDFs → extraction.py → all_provisions.json (3,980 provisions)
                    → embedding.py → ChromaDB vector store
                    → classification.py → classified_{model}_{strategy}.json
                                       → comparison.py → comparison_qwen.json
                                       → attribute_extraction.py → attributes_*.json
                    → analysis.py → analysis_bundle.json (κ, matrix, convergence)
                    → visualize.py → fig_*.png
                    → validation.py → validation_report.json
```

**Models:** LLaMA 3.3 70B (Meta, via Groq free tier) and Qwen 3 32B (Alibaba Cloud, via Groq free tier).
**Strategies:** zero-shot, few-shot (2 curated examples), chain-of-thought (step-by-step reasoning).
**Classification runs:** 6 main runs (200 provisions each, CoT capped at 100 by API quota) + 1 stratified run (300 provisions, 100 per agreement).
**Validation:** 50-provision stratified gold set, hand-labelled by author; 5 of 6 model-strategy combinations evaluated (LLaMA CoT excluded — API quota exhausted).

## Appendix B — Repository Layout

```
Final Project - FTA LLM/
├── Agreement/          source PDFs (7 documents)
├── config.py           central configuration (paths, API keys, categories)
├── run_pipeline.py     CLI orchestrator (steps 1–4)
├── retry_failed.py     error-recovery utility
├── src/
│   ├── extraction.py
│   ├── embedding.py
│   ├── classification.py
│   ├── comparison.py
│   ├── attribute_extraction.py
│   ├── analysis.py
│   ├── validation.py
│   └── visualize.py
├── data/
│   ├── raw/            extracted provisions JSON
│   ├── results/        classification, comparison, figures, validation
│   └── chromadb/       persistent vector store
├── METHODOLOGY.md      full technical documentation
├── REPORT_DRAFT.md     technical reference report
└── REPORT.md           this document (submission)
```

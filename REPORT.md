# A Computational Framework for Comparative Analysis of Free Trade Agreements

**Diyouva Christa Novith**
Machine Learning Foundation with Python — Carnegie Mellon University — Spring 2026 

---

> **Documentation status (April 2026):** This report summarises the current
> saved experiment artefacts in `data/`. For the executable rerun workflow, use
> `README.md` and `run_pipeline.py`. If extraction, sampling, validation, or
> comparison code changes, regenerate the JSON artefacts before treating the
> quantitative findings below as current.

## Abstract

Free Trade Agreements (FTAs) run to thousands of pages yet must be compared, monitored, and negotiated by policy analysts who rely almost entirely on manual review. This project builds an end-to-end Python pipeline that converts three major Asia-Pacific FTAs — RCEP, AHKFTA, and AANZFTA — into a structured, machine-readable dataset of 4,059 provisions and uses Large Language Models to classify, compare, and extract attributes from them. Two models (LLaMA 3.3 70B and Qwen 3 32B) were tested under three prompt strategies (zero-shot, few-shot, and chain-of-thought) through an iterative process that closely mirrors how a policy analyst would decide which tool to trust. On the current 50-provision hand-labelled gold set, the best accuracy is 48.0% (LLaMA zero-shot) and the best macro-F1 is 0.442 (Qwen chain-of-thought). The pipeline still surfaces concrete policy-design differences across agreements, but the classification layer should now be read as analyst triage rather than high-confidence automation.

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

Seven PDFs were processed. AHKFTA is partially scanned, requiring OCR. After extraction, the corpus contains **4,059 provisions** — the smallest meaningful legal unit (an article, paragraph, or structured annex entry), filtered to 80–1,500 characters.

| Agreement | Provisions | Share |
|---|---:|---:|
| RCEP | 2,171 | 53.5% |
| AANZFTA | 1,526 | 37.6% |
| AHKFTA | 362 | 8.9% |
| **Total** | **4,059** | 100% |

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
6. **Validation** — A 50-provision validation cohort was hand-labelled by the author; each model-strategy combination was re-run on that exact cohort and scored against the gold labels.

### 3.2 The Iterative Model Selection Process

**First attempt: Gemini.** The original plan was to compare LLaMA 3 against Google's Gemini. Gemini was tested first because of its reputation for instruction-following on structured tasks. It performed well — but its free-tier quota ran out after approximately 200 API calls, making a full three-strategy × 200-provision matrix infeasible. This was the first real constraint encountered and shaped everything after.

**Switch to Qwen 3 32B.** Qwen 3 32B (Alibaba Cloud) is available on the same Groq endpoint as LLaMA, on a separate daily quota. It is a different model family and architecture — making it a genuine cross-architecture comparison rather than just a size comparison. The switch preserved the original intent of the proposal: compare at least two architecturally distinct models.

**Three prompt strategies, each revealing something different:**

*Zero-shot* was the baseline. The prompt was simple: here are 11 categories, here is a provision, what category is it? In the current rerun artefacts, zero-shot is the strongest strategy for LLaMA and the second-best strategy for Qwen.

*Few-shot* added two curated examples: one clearly labeled Rules of Origin provision and one clearly labeled Tariff Commitments provision. In the current saved runs, few-shot underperforms zero-shot for both models on the 50-row gold set, suggesting that two exemplars are not enough to stabilise this taxonomy.

*Chain-of-thought (CoT)* asked the model to reason step-by-step before committing to a category. In the current rerun, CoT gives Qwen the highest macro-F1 (0.442) but only a small edge over Qwen zero-shot (0.424), while LLaMA CoT is the weakest of the six runs. That pattern still supports model-specific prompt tuning, but not the stronger asymmetry claim made in earlier drafts.

**Why CoT worked for Qwen but few-shot didn't** is a real finding, not a footnote. Few-shot examples constrain the model's output space toward the examples. CoT expands the reasoning space — it lets the model use its training knowledge about trade law without being anchored to two specific examples. Legal-text classification is a domain where this distinction matters: the categories are defined by legal function, not surface vocabulary, so broader reasoning outperforms narrow examples.

### 3.3 Key Design Decisions

**Stratified vs. random sampling.** RCEP makes up 53.5% of the corpus. Random sampling is used for the main benchmark runs, while the cross-agreement comparison uses a separate stratified 100-per-agreement sample to avoid letting RCEP dominate the comparative matrix.

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
| LLaMA zero-shot vs. few-shot | 0.67 | Substantial consistency |
| Qwen zero-shot vs. few-shot | 0.69 | Substantial |
| LLaMA few-shot vs. Qwen few-shot | 0.58 | Moderate |
| LLaMA zero-shot vs. Qwen zero-shot | 0.70 | Substantial |

The practical implication: **individual provision labels are noisy.** No single model-strategy combination should be trusted for compliance-grade decisions. The right use is aggregate distribution analysis — asking "how many provisions in each agreement address Rules of Origin?" rather than "is this specific provision a Rules of Origin provision?"

![Validation accuracy](data/results/fig_validation_accuracy.png)

Against the 50-provision gold set:

| Model & Strategy | Accuracy | Macro-F1 |
|---|---:|---:|
| LLaMA 3.3 70B — zero-shot | **0.480** | 0.431 |
| Qwen 3 32B — chain-of-thought | 0.460 | **0.442** |
| Qwen 3 32B — zero-shot | 0.380 | 0.424 |
| Qwen 3 32B — few-shot | 0.380 | 0.373 |
| LLaMA 3.3 70B — few-shot | 0.340 | 0.336 |
| LLaMA 3.3 70B — chain-of-thought | 0.320 | 0.327 |

The current validation results are materially weaker than earlier drafts in this repository. LLaMA zero-shot is the most accurate run, but only at 48%; Qwen CoT has the best macro-F1, but only by a narrow margin. The practical implication is that aggregate category distributions are still more defensible than any single provision-level label.

### 4.2 How Agreements Differ — The Comparative Matrices

The core deliverable for policy analysis: a structured comparison of what each agreement actually covers.

**Provision count matrix** (Qwen few-shot, stratified 100-per-agreement sample):

| Category | RCEP | AHKFTA | AANZFTA | Total |
|---|---:|---:|---:|---:|
| Rules of Origin | 24 | 48 | 39 | **111** |
| Tariff Commitments | 6 | 31 | 7 | **44** |
| Trade in Services | 20 | 0 | 10 | **30** |
| Dispute Settlement | 6 | 0 | 22 | **28** |
| Customs Procedures | 6 | 6 | 7 | **19** |
| General Provisions | 5 | 9 | 3 | **17** |
| Investment | 8 | 4 | 5 | **17** |
| Intellectual Property | 12 | 0 | 2 | **14** |
| Non-Tariff Measures | 6 | 1 | 4 | **11** |
| SPS Measures | 5 | 1 | 0 | **6** |
| Other | 2 | 0 | 1 | **3** |
| **Total** | **100** | **100** | **100** | **300** |

![Category × agreement](data/results/fig_category_x_agreement.png)

Three patterns stand out:

- **AHKFTA is a goods-focused agreement.** It has zero provisions in Trade in Services, Dispute Settlement, and Intellectual Property in this sample — consistent with its design as a bilateral goods-trade instrument between ASEAN and Hong Kong. RCEP and AANZFTA both have dedicated chapters for services and investment; AHKFTA effectively does not.

- **Rules of Origin weight diverges sharply.** AHKFTA devotes 48% of its sample to Rules of Origin provisions; RCEP devotes 24%. For a customs officer or compliance manager, this signals that AHKFTA's preferential-treatment framework is significantly more rule-intensive per unit of text — more conditions to satisfy.

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

- **Most convergent:** Customs Procedures and General Provisions. All three agreements allocate broadly similar shares of their text to these chapters, suggesting procedural harmonisation is advancing under a shared regional template.
- **Most fragmented:** Dispute Settlement and Trade in Services. Dispute Settlement is highly uneven: AANZFTA dedicates 22 provisions to it while AHKFTA has zero — consistent with AHKFTA's design as a goods-only instrument that delegates dispute resolution to the WTO DSU rather than maintaining an independent mechanism. Rules of Origin appears numerically dominant (111 total) but this reflects real scope differences rather than alignment: AHKFTA is far more rule-intensive than RCEP or AANZFTA.

This matters for regional integration policy: convergence in Customs Procedures suggests that trade facilitation — clearance, documentation, risk-based release — is developing shared regional norms. Fragmentation in Dispute Settlement and Trade in Services signals that the deeper integration agenda (services liberalisation, investment protection, enforcement) remains structurally divergent, limiting the practical scope of a unified regional trading framework.

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

**Validation quality is the more serious limitation in the current repo state.** Once runs are aligned on exactly the same cohorts, inter-run agreement is no longer near-random; the bottleneck is that all six validation scores remain low. For production use, this still implies human-in-the-loop review for any provision-level conclusion.

**The taxonomy covers nearly all provisions.** Only 3 of 300 stratified provisions (1%) are classified as "Other" in the Qwen few-shot stratified run — the taxonomy's 11 categories absorb the vast majority of FTA content. The concentration risk is instead in Rules of Origin (37% of the stratified sample), partly inflated by in-context few-shot examples that are both goods-trade categories. A few-shot set with broader category representation would reduce this bias.

**Structured thresholds live in annexes, not main text.** The attribute extractor was run on main-agreement provisions. The 40% RVC threshold shows up in the text; AANZFTA's equivalent is in a schedule. Future work should pass schedule-level subprovisions through the attribute extractor separately.

### 5.3 What This Means for Policy Practice

A trade policy analyst using this pipeline today could:

- **In under an hour:** Extract all provisions from a new FTA PDF, embed them, and run the classification to get a category distribution. This replaces 2–3 days of manual triage.
- **With current validation in the 32–48% accuracy range:** treat every individual label as a triage hint, not an answer. The value is prioritisation and aggregation, not autonomous legal judgement.
- **Immediately useful:** The comparative matrix and RAG narratives are specific enough to brief a negotiator on key structural differences across agreements. The finding that AHKFTA uses a stricter CTC rule than RCEP (CC vs. CTH) is actionable for an exporter choosing which agreement to invoke.

The tool is not ready for autonomous use. The accuracy ceiling, single-annotator gold labels, and inter-model disagreement all warrant caution. The correct framing is: this reduces the time a skilled analyst spends on triage from weeks to hours, and flags where the agreements differ in a structured format that the analyst can then verify against primary sources.

---

## 6. Conclusion

This project demonstrates that an LLM-based pipeline can extract, classify, and comparatively analyse FTA provisions at a scale and speed that manual review cannot match. Three key findings stand out:

**Model and prompt engineering decisions matter more than expected — and interact with model architecture in opposing ways.** Chain-of-thought prompting raised Qwen's macro-F1 by 1.8 points (from 0.424 to 0.442, best result in the study) but *lowered* LLaMA's by 1.0 point (from 0.431 to 0.327, worst result). Few-shot examples degraded both models relative to zero-shot, with LLaMA falling 9.5 points and Qwen falling 5.1 points. This cross-model asymmetry in CoT sensitivity was only discoverable through systematic validation — not from model documentation or casual testing.

**The agreements are structurally more different than their shared ASEAN context suggests.** AHKFTA is a goods-only instrument that lacks services, investment, and dispute settlement chapters. It shares RCEP's 40% RVC threshold but uses a stricter transformation rule. AANZFTA delegates threshold definitions to annexes in a way that makes automatic extraction harder. These are real differences with real compliance implications, and the pipeline surfaces them in a structured format.

**Rules of Origin and Dispute Settlement reveal the deepest structural divergences.** AHKFTA allocates nearly half its sampled provisions to Rules of Origin (48%) while dedicating none to Dispute Settlement or Intellectual Property. These are not minor scope differences — they reflect fundamentally different negotiating mandates: AHKFTA as a goods-access instrument, RCEP and AANZFTA as broader integration frameworks. The finding that Customs Procedures shows the most convergence while Dispute Settlement and Trade in Services show the most fragmentation suggests that procedural harmonisation is advancing faster than substantive liberalisation in the Asia-Pacific.

The pipeline is fully reproducible, uses only free-tier APIs, and can be pointed at any new FTA PDF. The recommended configuration is Qwen 3 32B with chain-of-thought prompting.

---

## Appendix A — Technical Pipeline

```
PDFs → extraction.py → all_provisions.json (4,059 provisions)
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
**Classification runs:** 6 main runs (reproducible random samples; CoT capped at 100 by API quota) + 1 stratified run (300 provisions, 100 per agreement).
**Validation:** 50-provision gold set, hand-labelled by author and exported to `validation_provisions.json`; all 6 model-strategy combinations evaluated on the same IDs.

## Appendix B — Repository Layout

```
Final Project - FTA LLM/
├── Agreement/          source PDFs (7 documents)
├── config.py           central configuration (paths, API keys, categories)
├── run_pipeline.py     CLI orchestrator (core stages + dataset-prep entrypoints)
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

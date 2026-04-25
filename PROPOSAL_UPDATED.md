# A Computational Framework for Comparative Analysis of Free Trade Agreements
**Diyouva Christa Novith**
Machine Learning Foundation with Python — Carnegie Mellon University — Spring 2026

---

> **Historical note:** this file is the project proposal. It records the planned
> design, not necessarily the final implementation details. For the current code
> path, use `README.md` / `METHODOLOGY.md`; for final results, use `REPORT.md`
> and the artefacts under `data/`.

## Policy and Organizational Decision Problem

Policymakers and trade analysts frequently rely on Free Trade Agreements (FTAs) to guide decisions on market access, tariff structures, and regulatory compliance. However, FTAs are complex, lengthy, and often overlapping across regions, creating what is commonly referred to as the **"spaghetti bowl" problem**. This complexity makes it difficult to systematically compare provisions across agreements and identify inconsistencies or opportunities. As a result, decision-making is often time-consuming and dependent on manual legal review.

This project develops a large language model (LLM)-based framework to extract, classify, and compare FTA clauses to support more efficient and evidence-based trade policy analysis. The primary stakeholders include government trade ministries, customs authorities, and policy analysts responsible for negotiating, implementing, and evaluating trade agreements.

---

## Research Questions and Modeled Outcomes

The project addresses three core questions:

**First,** can provisions from FTAs be reliably extracted and classified into standardized policy categories such as tariff commitments, rules of origin, and non-tariff measures using LLMs?

**Second,** how do comparable provisions differ across agreements in terms of observable policy design features such as rules of origin criteria, legal flexibility, and structure of commitments?

**Third,** do these agreements exhibit indicative patterns of convergence or fragmentation in their legal design that reflect broader regional trade dynamics?

The primary modeled outcome is a structured representation of agreement provisions, where each provision is assigned a category and key attributes that enable cross-agreement comparison. Additional outputs include comparative matrices summarizing differences across agreements and qualitative assessments of policy design variation.

---

## Data Sources

The analysis uses publicly available legal texts for:

- **RCEP** — Regional Comprehensive Economic Partnership (signed 2020, in force 2022)
- **AHKFTA** — ASEAN–Hong Kong, China FTA (signed 2017, in force 2019)
- **AANZFTA** — ASEAN–Australia–New Zealand FTA (signed 2009, in force 2010, amended 2014)

Seven source PDFs are processed, including main agreement text and selected annexes, schedules, and protocol amendments. The unit of analysis is the smallest meaningful legal provision — an article, paragraph, clause, rule, or structured annex entry — filtered to 80–1,500 characters.

The resulting dataset contains **3,980 provisions** across the three agreements (RCEP: 2,129 / AANZFTA: 1,498 / AHKFTA: 353). Key fields per provision: `id`, `agreement`, `doc_type`, `chapter`, `article`, `paragraph_idx`, `text`, `char_count`, and `category`.

**Alignment approach:** Cross-agreement provision alignment uses semantic similarity via a ChromaDB vector store (sentence-transformers `all-MiniLM-L6-v2`, cosine distance). Article and rule number alignment was explored but found infeasible due to sparse article metadata from PDF formatting variation; Harmonized System codes appear primarily in tariff schedules rather than main-agreement text. Semantic similarity is more robust to formatting variation, though it does not guarantee legally equivalent provisions are always paired.

---

## Analytical Approach and Evaluation Plan

The empirical strategy applies natural language processing and LLMs through a Retrieval-Augmented Generation (RAG) framework across six stages:

1. **PDF Extraction** — `pdfplumber` → `PyMuPDF` → Tesseract OCR fallback chain. The OCR fallback is essential for AHKFTA, whose source PDFs are scanned images.

2. **Embedding** — `all-MiniLM-L6-v2` sentence transformer; provisions stored in a ChromaDB persistent vector store. Enables semantic retrieval of similar provisions across agreements.

3. **Classification** — LLMs classify each provision into one of 11 predefined policy categories. Attribute extraction (hybrid regex + LLM) recovers structured numeric fields (RVC%, de-minimis%, phase-out years) and categorical flags (CTC rule family, HS scope).

4. **Cross-Agreement Comparison** — RAG step: top-3 semantically similar provisions per agreement per category are retrieved and passed to an LLM for structured comparative analysis across five dimensions: similarities, differences, flexibility/rigidity, convergence/fragmentation, and policy implications.

5. **Analysis** — Cohen's κ for inter-run agreement; entropy-based convergence signal per category.

6. **Validation** — 50-provision stratified gold set, hand-labelled by the project author. All model-strategy combinations are evaluated for accuracy and macro-F1.

**Models compared:**

| Model | Provider | Architecture | Free Tier |
|-------|----------|-------------|-----------|
| LLaMA 3.3 70B | Groq (Meta) | Decoder-only, instruction-tuned | 100K tokens/day |
| Qwen 3 32B | Groq (Alibaba Cloud) | Thinking model with `<think>` reasoning traces | Separate daily quota |

> **Note on model selection:** The original proposal named Gemini 1.5 Flash (Google AI Studio) as the second model. During implementation, Gemini's free-tier daily quota was exhausted after approximately 200 API calls, making a full three-strategy × multi-provision matrix infeasible. Qwen 3 32B was substituted because (a) it is hosted on the same Groq endpoint as LLaMA under a separate daily quota, (b) it represents a genuinely different model architecture (thinking model vs. standard decoder), and (c) it preserves the proposal's intent of comparing at least two architecturally distinct free models.

**Prompt engineering strategy** — applied iteratively, from simple to complex:

1. **Zero-shot** — category list + provision text; no examples
2. **Few-shot** — two curated in-context examples prepended
3. **Chain-of-thought (CoT)** — explicit step-by-step reasoning instruction before label commitment

Model performance is validated by manually reviewing 50 stratified provisions against source text. Accuracy and macro-F1 are reported for all six model-strategy combinations.

---

## Constraints and Mitigation

| Constraint | Mitigation |
|-----------|-----------|
| Variability in PDF formatting (scanned pages, inconsistent heading styles) | Three-library extraction chain with OCR fallback; minimum/maximum character filters |
| LLM output inconsistency | Structured output parsing; validation against hand-labelled gold set; Cohen's κ quantifies inter-run noise |
| Legal language ambiguity in classification | Iterative prompt refinement; "Other" category for unclassifiable provisions; multi-strategy comparison reveals sensitivity |
| Free-tier API quota constraints | Daily TPD limits (100K tokens/day for LLaMA 3.3 70B) managed via exponential backoff with minute/hour format parsing; CoT runs planned for low-usage windows |
| Sparse article-level metadata | Semantic similarity (ChromaDB) used for cross-agreement alignment instead of article-number keys |
| Annex and schedule coverage | Attribute extractor run on main-agreement text; threshold recovery incomplete for AANZFTA (delegates to schedules) |

The dataset is publicly available and does not contain confidential information. The scope is limited to three agreements to ensure feasibility within the project timeline.

---

## Results Summary

| Research Question | Key Finding |
|------------------|-------------|
| **RQ1 — Reliable classification** | Best configuration: Qwen 3 32B with CoT — 70.0% accuracy, 0.693 macro-F1. CoT helps Qwen (+10pp) but hurts LLaMA (−6pp); prompt strategy must be tuned per model. |
| **RQ2 — Policy design differences** | AHKFTA is goods-only (no services/investment/dispute chapters); uses CC (chapter-level CTC) vs. RCEP's CTH (heading-level) — same 40% RVC but stricter transformation. AANZFTA delegates thresholds to schedules. |
| **RQ3 — Convergence/fragmentation** | General Provisions and Dispute Settlement are convergent (shared regional template); Tariff Commitments and Rules of Origin are fragmented (each agreement follows a distinct design). |

---

## Policy Contribution

This project contributes to trade policy analysis by providing a scalable, reproducible method to transform unstructured legal agreements into structured and comparable policy data. By enabling systematic comparison of provisions across agreements — with quantified reliability estimates — the framework supports more efficient policy evaluation and reduces reliance on manual document review.

A trade policy analyst using this pipeline can extract and classify all provisions from a new FTA in under an hour, produce category-level distribution comparisons across agreements, and retrieve LLM-generated comparative narratives on any policy domain. At 70% accuracy, the tool is triage-grade: it reduces time-to-insight from weeks to hours and flags structurally significant differences for analyst verification against primary sources.

The pipeline is fully reproducible, requires only a free Groq API key, and generalises to any FTA with a PDF text layer. The recommended configuration for production use is Qwen 3 32B with chain-of-thought prompting.

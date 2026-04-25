# Methodology — A Computational Framework for Comparative Analysis of Free Trade Agreements

> Author: **Diyouva Christa Novith**
> Course: Machine Learning Foundation with Python, Carnegie Mellon University (Spring 2026)

> **Documentation status (April 2026):** This file explains the current code
> structure and methodology. For the exact rerun sequence, use `README.md`.

## 1. Overview

The project builds a multi-stage pipeline that transforms raw legal PDFs of three ASEAN Free Trade Agreements (RCEP, AHKFTA, AANZFTA) into a structured, comparable, and queryable dataset of policy provisions. Each stage is a standalone Python module under `src/`, while `run_pipeline.py` orchestrates the core stages and the dataset-prep entrypoints.

```
PDFs ─► Extraction ─► Embedding ─► Classification ─► Comparison ─► Analysis
```

## 2. Stage 1 — PDF Extraction & Clause Segmentation

**Module:** `src/extraction.py`

### Inputs
- `Agreement/*.pdf` — 7 source documents totalling ~3,000 pages.

### Processing
1. **Primary extractor:** `pdfplumber` for text-based PDFs.
2. **Fallback 1:** `PyMuPDF (fitz)` when `pdfplumber` returns < 100 chars on the first page (heuristic for non-standard PDF structure).
3. **Fallback 2:** `pytesseract` (Tesseract OCR) for scanned/image-based PDFs. Triggered automatically when both text-based extractors fail.
4. **Clause segmentation:** provisions are segmented from detected legal headers while preserving those headers in the emitted text block so `article` / `chapter` metadata can survive extraction.
5. **Minimum size filter:** clauses < 80 characters are dropped (typical of page numbers or stray OCR artefacts).
6. **Maximum chunk filter:** clauses > 1,500 characters are subdivided at paragraph boundaries.

### Output

`data/raw/all_provisions.json` — a list of 3,980 dictionaries with fields:

| Field | Description |
|---|---|
| `id` | `{agreement}_{doc_type}_{index:05d}` |
| `agreement` | One of `RCEP`, `AHKFTA`, `AANZFTA` |
| `doc_type` | e.g. `Main Agreement`, `First Protocol` |
| `chapter` | Extracted from heading where possible |
| `article` | Article / Rule number |
| `paragraph_idx` | Ordinal within article |
| `text` | Clean provision text |
| `char_count` | Length |

**Provisions per agreement:** RCEP = 2,129, AANZFTA = 1,498, AHKFTA = 353.

## 3. Stage 2 — Embeddings & Vector Store

**Module:** `src/embedding.py`

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, lightweight).
- **Store:** ChromaDB persistent collection `fta_provisions` (HNSW, cosine distance).
- **Batching:** 256 provisions per batch during index build.
- **Retrieval APIs:** `retrieve_similar(...)` for raw vector retrieval and `rank_provisions_by_query(...)` for ranking already-classified candidates.

Embeddings are produced once and cached via module-level singletons (`_model`, `_chroma_client`) — repeated `retrieve_similar` calls do **not** re-load the model.

## 4. Stage 3 — LLM Classification

**Module:** `src/classification.py`

### Models compared

| Alias | Model | Provider | Size |
|---|---|---|---|
| `llama` | LLaMA 3.3 70B Versatile | Groq | 70 B |
| `qwen` | Qwen 3 32B | Groq | 32 B |

Both accessed via the same Groq free-tier API key. Qwen 3 is a *thinking* model that emits `<think>…</think>` reasoning before the answer; the parser strips that block.

*(The proposal originally named Gemini; we switched to Qwen mid-project because Gemini's free-tier daily quota was insufficient for repeated experimentation, while Groq hosts multiple free models under one key.)*

### Prompt strategies (all three applied to each model)

1. **Zero-shot** — category list + instruction; no examples.
2. **Few-shot** — two curated RoO & Tariff examples (trimmed from five after rate-limit analysis).
3. **Chain-of-thought (CoT)** — asks the model to reason through (a) main legal subject, (b) key obligations, (c) best category, then emit `CATEGORY: <name>`.

### Rate-limit handling
- Groq free tier ≈ 6,000 tokens/min per model for 70 B class; 15,000/min for 32 B class.
- Strategy-aware delays: zero-shot 2 s, few-shot 10 s, CoT 12–15 s.
- Retry loop extracts `try again in Xs` from the 429 body when present; otherwise falls back to linear back-off `15 × attempt` seconds, up to 8 attempts.

### Output

`data/results/classified_{model}_{strategy}.json` — one row per provision with `category`, `raw_response`, `model`, `strategy` fields merged on the original provision record.

### Sampling policy
- Main classification runs use reproducible random sampling (`seed=42`) when `--limit` is set.
- A separate run on `data/raw/stratified_sample.json` (100 provisions per agreement, seed 42) ensures cross-agreement comparison is not biased toward RCEP, which dominates the full corpus.

## 5. Stage 4 — Cross-Agreement Comparison (RAG)

**Module:** `src/comparison.py`

For each policy category (11 total):
1. Load a classified run, preferably the stratified cohort.
2. Filter provisions to those already assigned to the target category within each agreement.
3. Rank those candidates semantically (or with lexical fallback if embeddings are unavailable locally) and take the top 3 per agreement.
4. A structured comparison prompt asks the LLM to identify:
   - similarities,
   - differences,
   - flexibility vs rigidity,
   - convergence vs fragmentation,
   - implications for negotiators / compliance officers.
5. Response capped at ~400 words via `max_tokens=2048` (high enough to accommodate Qwen's `<think>` block plus the answer).

Output: `data/results/comparison_{model}.json`.

## 6. Stage 5 — Analysis

**Module:** `src/analysis.py`

Pure-Python analysis on the classified JSONs — no further API calls.

- `category_matrix(run)` → `{category: {agreement: count}}`.
- `compare_two_runs(a, b)` → raw agreement %, **Cohen's κ**, list of disagreements.
- `convergence_signal(matrix)` → entropy + coefficient of variation per category; flags each category as *convergent* or *fragmented*.

Exports `analysis_bundle.json` and `analysis_disagreements.json`.

## 7. Stage 6 — Manual Validation

**Module:** `src/validation.py`

1. `--sample --source classified_qwen_few_shot_stratified.json` creates `validation_set.csv` from a balanced classified cohort.
2. Analyst hand-labels the `gold_category` column.
3. `--export-validation-provisions` writes the exact IDs from `validation_set.csv` back to `data/raw/validation_provisions.json`.
4. Validation classification runs are executed on that exact JSON cohort.
5. `--evaluate` scores only runs whose ID set exactly matches the labelled validation cohort. Exports `validation_report.json`.

## 8. Attribute Extraction (Optional RQ2 deep-dive)

**Module:** `src/attribute_extraction.py`

For richer cross-agreement comparison on policy *design features*, a secondary pass extracts structured attributes from provisions in selected categories:

| Category | Extracted attributes |
|---|---|
| Rules of Origin | `rvc_pct`, `de_minimis_pct`, `ctc_rule`, `cumulation_type`, `wholly_obtained_clause` |
| Tariff Commitments | `phase_out_years`, `max_reduction_pct`, `hs_scope`, `staging_category` |
| Trade in Services | `mfn_clause`, `national_treatment_clause`, `market_access_clause`, `mode_of_supply` |
| Dispute Settlement | `consultation_days`, `panel_formation_days`, `appeal_mechanism` |
| Non-Tariff Measures | `measure_type`, `transparency_requirement` |
| Investment | `expropriation_clause`, `isds_clause`, `performance_requirements_restricted` |

A hybrid approach is used: **regex** extracts deterministic numeric fields (e.g. `RVC 40%`), while the **LLM** fills in semantic flags (`cumulation_type = "full"`). Results are merged with regex values winning on overlap.

## 9. Reproducibility Checklist

- Random seeds fixed (`seed=42` for stratified sampling and main random sampling; validation sample seed is configurable from the CLI).
- API keys via `.env` (never committed — see `.gitignore`).
- All intermediate artefacts saved under `data/` for re-use.
- Every classification run records the exact prompt strategy and model version.

## 10. Known Limitations

1. **Legal language ambiguity.** "Other" captures provisions that span multiple categories; future work could use multi-label classification.
2. **Free-tier quotas.** LLaMA CoT validation required a separate day's quota allocation (100,000 tokens/day rolling window). Completed result: accuracy 0.480, macro-F1 0.527 — the lowest of all six runs, confirming that CoT hurts LLaMA while helping Qwen.
3. **Attribute extraction remains annex-sensitive.** Structured fields (RVC %, CTC rule, phase-out years) are best recovered from the stratified sample, but completeness still depends on whether the relevant schedule or annex text entered the extracted provision corpus.
4. **Agreements limited to three** (RCEP, AHKFTA, AANZFTA) by design — the framework generalises to any FTA with a PDF text layer.

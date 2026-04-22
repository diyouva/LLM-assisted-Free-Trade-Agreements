# Classifying and Comparing Free Trade Agreement Provisions at Scale: A Multi-Model, Multi-Strategy LLM Evaluation with Inter-Run Agreement Analysis

**Diyouva Christa Novith**
School of Computer Science, Carnegie Mellon University, Pittsburgh, PA, USA
Email: dnovith@andrew.cmu.edu

---

> **Submitted to:** *Artificial Intelligence and Law* (Springer) / *Expert Systems with Applications* (Elsevier)
> **Manuscript type:** Full Research Article
> **Word count:** ~9,800 words (excluding references)

---

## Abstract

The proliferation of bilateral and regional Free Trade Agreements (FTAs) has created a complex, overlapping treaty landscape — commonly termed the "spaghetti bowl" — that imposes substantial analytical burdens on trade policy professionals. Manual comparison of legal provisions across agreements is time-intensive and does not scale to the dozens of simultaneously active FTAs in any given region. This paper presents a computational pipeline that applies open-weight Large Language Models (LLMs) to the extraction, classification, and cross-agreement comparison of FTA provisions at scale. We construct a corpus of 3,980 provisions extracted from three major Asia-Pacific agreements — RCEP, AHKFTA, and AANZFTA — and systematically evaluate two architecturally distinct LLMs (LLaMA 3.3 70B and Qwen 3 32B) across three prompt strategies (zero-shot, few-shot, and chain-of-thought) against a hand-labelled 50-provision gold set. Our best configuration — Qwen 3 32B with chain-of-thought prompting — achieves 70.0% accuracy and 0.693 macro-F1. We document a striking architectural asymmetry: chain-of-thought reasoning improves Qwen 3 32B by approximately 10 percentage points in macro-F1 but degrades LLaMA 3.3 70B by 6 points, a finding attributable to the fundamental difference between thinking-model and standard decoder architectures. Cohen's κ analysis reveals that inter-model agreement is near-chance (κ = −0.02 between LLaMA few-shot and Qwen few-shot) despite similar aggregate accuracy, with critical implications for ensemble-based validation approaches. Cross-agreement structural analysis using Retrieval-Augmented Generation identifies Rules of Origin and Tariff Commitments as highly fragmented across agreements, while General Provisions and Dispute Settlement exhibit emerging regional convergence. This work establishes a reproducible, freely deployable benchmark for computational FTA analysis and identifies a previously undocumented failure mode of ensemble LLM validation in legal text classification.

**Keywords:** Large Language Models; Free Trade Agreements; Legal Text Classification; Chain-of-Thought Prompting; Retrieval-Augmented Generation; Inter-Annotator Agreement; Trade Policy; Natural Language Processing

---

## 1. Introduction

Free Trade Agreements have become the dominant instrument of international trade governance. As of 2024, the World Trade Organization's Regional Trade Agreement database records over 350 agreements in force, with the Asia-Pacific region alone maintaining dozens of overlapping bilateral and multilateral instruments [WTO, 2024]. Each agreement may span hundreds of pages across multiple chapters — covering tariff schedules, rules of origin, services liberalisation, investment protection, dispute settlement, and regulatory harmonisation. The analytical challenge of comparing provisions across even a small number of agreements has been described as the "spaghetti bowl" problem [Bhagwati, 1995], and represents a genuine bottleneck for trade policy professionals tasked with negotiating new agreements, assessing compliance with existing ones, or advising on market entry decisions.

The emergence of capable, openly accessible Large Language Models offers a tractable path towards automating the most labour-intensive stages of FTA analysis. Prior work in legal NLP has demonstrated that transformer-based models can perform well on tasks including contract understanding [Hendrycks et al., 2021], legal judgment prediction [Chalkidis et al., 2022], and statutory interpretation [Savelka and Ashley, 2023]. However, the application of modern LLMs to the specific domain of international trade agreements — characterised by their multilateral scope, highly technical vocabulary, and structurally heterogeneous document formats — remains largely unexplored. To our knowledge, no prior study has systematically benchmarked LLM classification strategies across multiple FTAs against a hand-labelled gold set, or examined the consequences of inter-model disagreement in this domain.

This paper addresses that gap. We make four primary contributions:

1. **A reproducible FTA provision corpus and evaluation protocol.** We extract 3,980 provisions from three major Asia-Pacific FTAs using a pdfplumber → PyMuPDF → Tesseract OCR extraction chain and construct a stratified 50-provision gold set labelled against a 11-category WTO/UNCTAD taxonomy. All data, code, and results are publicly available.

2. **A systematic multi-model, multi-strategy classification benchmark.** We evaluate LLaMA 3.3 70B and Qwen 3 32B — two architecturally distinct open-weight models available without cost via the Groq inference platform — across zero-shot, few-shot, and chain-of-thought prompt strategies, yielding six complete evaluation runs.

3. **An architectural asymmetry finding.** Chain-of-thought prompting substantially improves classification quality for Qwen 3 32B (a thinking model designed for extended reasoning) but substantially degrades it for LLaMA 3.3 70B (a standard instruction-tuned decoder). This asymmetry is reproducible and theoretically grounded, and has not been previously documented for legal text classification tasks.

4. **A near-zero inter-model agreement finding with practical implications.** Despite similar aggregate accuracy, LLaMA few-shot and Qwen few-shot achieve Cohen's κ = −0.02 on the same set of provisions, indicating that the models make errors on largely disjoint subsets of provisions. We argue this invalidates naïve ensemble or cross-model validation approaches for this task.

The remainder of this paper is structured as follows. Section 2 reviews related work. Section 3 describes the data and corpus construction. Section 4 details the methodology. Section 5 presents experimental results. Section 6 discusses findings, limitations, and implications. Section 7 concludes.

---

## 2. Background and Related Work

### 2.1 The FTA Analytical Challenge

The proliferation of preferential trade agreements since the early 1990s has been extensively documented in the international trade economics literature [Baldwin and Jaimovich, 2012]. Bhagwati's [1995] "spaghetti bowl" metaphor captures the central problem: overlapping FTAs with inconsistent provision structures create a web of partially contradictory obligations that is analytically intractable at scale. Quantitative trade economists have addressed this through aggregate indicators — using gravity models to estimate agreement effects [Breinlich et al., 2022] or machine learning to score agreement depth along predefined dimensions [Baier and Regmi, 2023]. These approaches treat agreements as objects to be scored rather than as texts to be read, and do not operate at the provision level. Provision-level analysis, which would reveal not only *how deep* agreements are but *where* they differ and *how* they are structured, requires natural language processing methods.

### 2.2 Legal Text Classification with Language Models

Legal NLP has advanced rapidly since the introduction of domain-adapted transformer models. Chalkidis et al.'s [2020] LEGAL-BERT demonstrated that continued pre-training on legal corpora substantially improves performance on tasks including multi-label classification of European Court of Human Rights judgments and contract clause classification. The LexGLUE benchmark [Chalkidis et al., 2022] systematised evaluation across seven legal NLP tasks, establishing reference performance figures for a range of models. More recently, Guha et al.'s [2023] LegalBench extended this to 162 tasks specifically targeting legal reasoning, and Savelka and Ashley [2023] demonstrated that modern LLMs exhibit surprising zero-shot capability on legal semantic annotation tasks, even outperforming specialised models on some sub-tasks.

Critically for our application, prior legal NLP benchmarks focus predominantly on case law, statutory text, or commercial contracts under common-law systems. International trade agreements — which are multilateral treaty instruments governed by public international law and drafted in a highly specific technical register — have not been a primary target of legal NLP research. Their document structure (chapter-article-annex hierarchies, cross-references to HS codes, cascading definitional dependencies) presents extraction and classification challenges not addressed by existing benchmarks.

### 2.3 Prompt Engineering and Chain-of-Thought Reasoning

The ability to elicit capable behaviour from LLMs through natural language instructions — without updating model weights — has been established across a range of settings. Brown et al.'s [2020] GPT-3 paper introduced few-shot in-context learning as a practical approach to task adaptation, and Wei et al.'s [2022] chain-of-thought prompting demonstrated that instructing models to produce intermediate reasoning steps before a final answer substantially improves performance on multi-step reasoning tasks. Kojima et al.'s [2022] "zero-shot CoT" approach — simply appending "Let's think step by step" to a prompt — achieves similar gains without hand-crafted exemplars.

However, the universality of CoT gains has been questioned. Wei et al. [2022] noted that CoT benefits emerged primarily in models exceeding approximately 100B parameters and were absent or negative in smaller models. Liu et al.'s [2024] "Mind Your Step" paper provided systematic evidence that chain-of-thought reasoning reduces performance on tasks where deliberate step-by-step thinking tends to worsen human performance — a phenomenon they term "verbal overshadowing." More directly relevant to our findings, the architectural distinction between standard instruction-tuned decoders and explicitly designed "thinking models" (which are trained on reasoning traces) introduces a variable that prior CoT literature does not address. Qwen 3 32B is explicitly designed to emit extended reasoning tokens before a final answer; LLaMA 3.3 70B is not. We return to this distinction in Section 6.

Comprehensive surveys of prompt engineering techniques [Sahoo et al., 2024; Schulhoff et al., 2024; Vatsal and Dubey, 2024] document hundreds of prompting strategies, but empirical evaluations specific to legal classification tasks remain sparse, and to our knowledge no study has compared prompt strategies across architecturally distinct model types on a trade law classification task.

### 2.4 Retrieval-Augmented Generation for Legal Documents

Retrieval-Augmented Generation [Lewis et al., 2020] extends language model generation by first retrieving relevant passages from an external knowledge base and conditioning the generation on those passages. This addresses a fundamental limitation of parametric knowledge in LLMs — their inability to reason over documents not seen during training, or to engage in explicit cross-document comparison. In legal settings, RAG has been applied to question answering [Pipitone and Alami, 2024], multi-turn legal consultation [Li et al., 2025], and contract review. In our application, RAG serves a cross-agreement comparison function: for each policy category, we retrieve the top-3 semantically similar provisions from each of the three agreements and provide them as context to the LLM, which then generates a structured comparative narrative. This approach is more robust to FTA document heterogeneity than structural alignment methods (e.g., aligning by HS code or article number), which we found infeasible due to sparse article-level metadata from PDF formatting variation.

### 2.5 Inter-Annotator Agreement in NLP Evaluation

Cohen's κ [Cohen, 1960] is the standard measure of pairwise inter-rater agreement corrected for chance. In NLP annotation studies, κ values are used to assess the reliability of human-generated labels before using them for model training or evaluation [Artstein and Poesio, 2008]. More recently, Kim and Park [2023] surveyed the role of inter-annotator agreement in real-world NLP settings, noting a shift from treating low IAA as a problem to treating it as a signal of genuine label ambiguity. In our work, we extend this framing to inter-*model* agreement: we treat Cohen's κ between two LLM runs as a diagnostic for whether two models are capturing the same underlying construct. Near-zero or negative κ between models that achieve similar accuracy against a gold standard implies the models are correct on disjoint subsets of the provision space — a finding with direct implications for multi-model ensemble strategies.

---

## 3. Data

### 3.1 Agreement Corpus

We analyse three Free Trade Agreements that collectively define the contemporary Asia-Pacific trade architecture:

**RCEP (Regional Comprehensive Economic Partnership)**, signed November 2020, entered into force January 2022. Covering 15 parties (ASEAN 10 plus Australia, China, Japan, New Zealand, and South Korea), RCEP is the world's largest FTA by combined GDP of signatories. Its 20 chapters span goods, services, investment, intellectual property, e-commerce, competition, and dispute settlement.

**AHKFTA (ASEAN–Hong Kong, China Free Trade Agreement)**, signed November 2017, entered into force in phases from 2019. A goods-focused agreement between the 10 ASEAN member states and Hong Kong, SAR. Its limited chapter structure — covering trade in goods, rules of origin, customs procedures, and limited institutional provisions — makes it a useful contrast case against the more comprehensive RCEP.

**AANZFTA (ASEAN–Australia–New Zealand Free Trade Agreement)**, signed February 2009 (entered into force 2010), amended by Protocol in 2014 and 2022. Covering 12 parties, AANZFTA is a comprehensive agreement predating RCEP by over a decade, providing a temporal baseline for studying provision design evolution.

Seven source documents are processed: the main agreement text and selected annexes for each FTA (Table 1).

**Table 1. Source Documents**

| Agreement | Documents | Pages (approx.) |
|-----------|-----------|-----------------|
| RCEP | Main agreement + Annex on RoO | ~560 |
| AHKFTA | Main agreement (scanned PDF) + Annex | ~310 |
| AANZFTA | Main agreement + Protocol amending AANZFTA | ~480 |

### 3.2 Provision Extraction

PDF extraction proceeds through a three-stage pipeline: pdfplumber (primary), PyMuPDF (fallback for encoding errors), and Tesseract OCR (fallback for scanned pages). The OCR fallback is essential for AHKFTA, whose source documents are bitmap scans of signed originals. Character-count filters discard provisions shorter than 80 characters (headings, article titles, formatting artefacts) or longer than 1,500 characters (consolidated tables that span multiple semantic units). Each provision is stored with a structured schema:

```
{id, agreement, doc_type, chapter, article, paragraph_idx, text, char_count}
```

The resulting corpus comprises **3,980 provisions**: RCEP 2,129 (53.5%), AANZFTA 1,498 (37.6%), and AHKFTA 353 (8.9%). The pronounced RCEP majority reflects both the agreement's breadth and the processing of its detailed annexes; we address the resulting sampling bias in Section 4.1 through stratified sampling.

### 3.3 Gold Set Construction

We construct a 50-provision stratified gold set for model evaluation. Provisions are sampled from the full corpus proportional to agreement size within each policy category, then reviewed manually against source documents to confirm extraction fidelity. Gold labels are assigned by the author following the WTO/UNCTAD chapter taxonomy (Section 4.2), with reference to the original article context and chapter heading. Ambiguous provisions (those plausibly fitting two categories) are resolved by identifying the *primary* policy function of the provision; the "Other" category is used only for provisions with no clear primary function. The completed gold set comprises provisions from all 11 categories across all three agreements; the most represented categories are General Provisions (12 provisions), Rules of Origin (9 provisions), and Customs Procedures (7 provisions).

---

## 4. Methodology

### 4.1 Classification Framework and Sampling

For classification evaluation, we construct a **stratified sample** of 100 provisions per agreement (300 total, seed=42), drawn proportionally within each agreement's category distribution. This corrects for the corpus imbalance in which RCEP accounts for 53.5% of all provisions; without stratification, cross-agreement comparisons would be dominated by RCEP's provision structure.

Classifications are performed via the Groq inference API, which provides rate-limited free-tier access to both experimental models. Rate limit handling uses exponential backoff with minute-and-seconds parsing of Groq's error messages to accurately compute wait times. All results are persisted to JSON after each API call to support resumable runs within Groq's 100,000-token-per-day rolling window.

### 4.2 Policy Category Taxonomy

We adopt an 11-category taxonomy aligned with the standard WTO/UNCTAD FTA chapter structure:

1. **Tariff Commitments** — duty reduction schedules, tariff elimination timelines
2. **Rules of Origin** — RVC thresholds, change-in-tariff-classification rules, substantial transformation criteria
3. **Non-Tariff Measures** — import licensing, technical standards, quantitative restrictions
4. **Trade in Services** — Mode 1–4 service liberalisation commitments
5. **Investment** — national treatment, most-favoured-nation treatment for investors, ISDS
6. **Dispute Settlement** — panel procedures, consultation timelines, arbitration
7. **Customs Procedures** — documentation requirements, advance rulings, single-window systems
8. **Sanitary and Phytosanitary Measures** — food safety, plant and animal health standards
9. **Intellectual Property** — copyright, trademarks, geographical indications
10. **General Provisions / Definitions** — scope, definitions, general exceptions
11. **Other** — provisions not fitting any category above

### 4.3 Models

**LLaMA 3.3 70B** [Grattafiori et al., 2024] is a 70-billion parameter instruction-tuned language model released by Meta AI in December 2024. It is a standard autoregressive decoder trained via reinforcement learning from human feedback (RLHF) and direct preference optimisation. LLaMA 3.3 70B does not employ explicit reasoning trace generation; it produces answers directly from the prompt context.

**Qwen 3 32B** [Qwen Team, 2025] is a 32-billion parameter model from Alibaba Cloud released in 2025. Critically, Qwen 3 is a *thinking model*: it is trained to emit structured `<think>...</think>` reasoning tokens before a final answer, analogous to OpenAI's o-series models. This architectural distinction — the presence of an explicit reasoning phase as a trained behaviour, rather than as an emergent property of a longer prompt — is central to interpreting our chain-of-thought results.

Both models are accessed via Groq's free inference tier. Qwen 3 32B reasoning tokens are stripped from the recorded output before category extraction; only the final label line is retained.

### 4.4 Prompt Strategies

Three prompt strategies are evaluated for each model, yielding six complete classification runs:

**Zero-shot.** The prompt contains the category list with one-line descriptions and the provision text. No examples are provided. The model is instructed to respond with exactly the category name, nothing else. Approximate token count: ~400 per call.

**Few-shot.** Two labelled examples — one clear instance of Rules of Origin and one clear instance of Dispute Settlement — are prepended to the zero-shot prompt structure. Examples are chosen to be unambiguous and representative. Approximate token count: ~900 per call.

**Chain-of-thought.** The instruction is augmented with the directive: *"Think step by step about what policy area this provision addresses, considering the subject matter, the type of obligation created, and the chapter context. State your reasoning first, then provide your final category label on the last line."* Approximate token count: ~1,400 per call (including reasoning output for Qwen).

### 4.5 Cross-Agreement Comparison via RAG

For each of the 11 policy categories, we embed all provisions classified under that category (using the best-performing Qwen CoT run) via the `all-MiniLM-L6-v2` sentence transformer [Reimers and Gurevych, 2019] and store them in a ChromaDB persistent vector store using cosine distance. For each category, we retrieve the top-3 provisions per agreement by cosine similarity to the category's centroid vector, yielding a context window of up to 9 provisions per query. This context is passed to Qwen 3 32B with a structured comparative prompt requesting analysis along five dimensions: (1) similarities, (2) differences, (3) flexibility vs. rigidity, (4) convergence/fragmentation signals, and (5) policy implications. The use of semantic similarity rather than structural alignment (e.g., HS code matching or article-number alignment) was necessitated by sparse article metadata in the extracted provisions — article numbers are inconsistently preserved across the three PDF sources.

### 4.6 Attribute Extraction

Beyond category classification, we extract structured attributes from Rules of Origin and Tariff Commitment provisions using a hybrid regex–LLM approach. Numeric fields recoverable through pattern matching (RVC threshold percentages, de minimis percentages, phase-out years) are extracted via regex. Categorical fields requiring semantic interpretation (CTC rule family, HS code scope, staging category) are extracted via LLM prompt. This hybrid approach balances precision (regex is exact for well-formatted numeric values) with the flexibility needed for varied legal phrasing.

### 4.7 Evaluation Metrics

**Accuracy** measures the proportion of correctly classified provisions out of 50. While interpretable, accuracy is sensitive to category imbalance in the gold set.

**Macro-F1** computes the unweighted mean F1 score across all 11 categories, weighting each category equally regardless of its frequency in the gold set. This penalises models that achieve high accuracy by correctly classifying common categories while ignoring rare ones, and is our primary evaluation metric.

**Cohen's κ** [Cohen, 1960] for inter-run agreement is computed for all pairwise combinations of the six classification runs on the 300-provision stratified sample. κ corrects observed agreement for the proportion expected by chance given the marginal distributions of each run. We use κ as a diagnostic for convergent construct validity across model-strategy combinations.

**Entropy-based convergence signal** for cross-agreement analysis computes the normalised Shannon entropy of the provision-count distribution across three agreements for each category. Low entropy (near 0) indicates one agreement dominates that category; high entropy (approaching log(3)) indicates equal representation across agreements — our proxy for structural convergence.

---

## 5. Results

### 5.1 Classification Performance

Table 2 presents accuracy and macro-F1 for all six model-strategy combinations on the 50-provision gold set.

**Table 2. Validation Results (n = 50)**

| Model | Strategy | Accuracy | Macro-F1 |
|-------|----------|----------|----------|
| Qwen 3 32B | Chain-of-thought | **0.700** | **0.693** |
| LLaMA 3.3 70B | Zero-shot | 0.700 | 0.591 |
| LLaMA 3.3 70B | Few-shot | 0.680 | 0.635 |
| Qwen 3 32B | Zero-shot | 0.680 | 0.596 |
| Qwen 3 32B | Few-shot | 0.580 | 0.540 |
| LLaMA 3.3 70B | Chain-of-thought | 0.480 | 0.527 |

The best-performing configuration — Qwen 3 32B with chain-of-thought prompting — achieves 70.0% accuracy and 0.693 macro-F1. The performance range across all six configurations spans 22 percentage points in accuracy (48%–70%) and 17 points in macro-F1 (0.527–0.693), indicating that prompt strategy is a dominant performance variable independent of model capability.

Note that LLaMA 3.3 70B zero-shot and Qwen 3 32B CoT achieve identical accuracy (70.0%) but diverge by 10 points in macro-F1 (0.591 vs. 0.693). Inspection of the confusion matrices reveals that LLaMA zero-shot concentrates errors in minority categories (Intellectual Property, Sanitary and Phytosanitary, Non-Tariff Measures), whereas Qwen CoT distributes errors more evenly across categories — explaining the F1 differential despite equal accuracy.

### 5.2 The Chain-of-Thought Asymmetry

The most salient finding in Table 2 is the opposing effect of chain-of-thought prompting across model architectures:

- **Qwen 3 32B:** CoT vs. zero-shot: +0.097 macro-F1 (+22% relative improvement)
- **LLaMA 3.3 70B:** CoT vs. zero-shot: −0.064 macro-F1 (−11% relative degradation)

This asymmetry is robust: it holds across both accuracy and F1, and the magnitude is large relative to the performance range of the experiment. CoT is the best strategy for Qwen and the worst strategy for LLaMA.

Examining the few-shot results provides additional evidence. Few-shot improves LLaMA (+0.044 macro-F1 vs. zero-shot) but substantially degrades Qwen (−0.056 macro-F1 vs. zero-shot). The optimal strategy rankings are opposite: Qwen benefits from reasoning space (CoT > zero-shot > few-shot), while LLaMA benefits from exemplar guidance (few-shot > zero-shot > CoT).

### 5.3 Inter-Run Agreement

Table 3 presents pairwise Cohen's κ for all fifteen pairs of classification runs on the 300-provision stratified sample.

**Table 3. Pairwise Cohen's κ (300 stratified provisions)**

| Run A | Run B | κ | Interpretation |
|-------|-------|---|----------------|
| LLaMA zero-shot | LLaMA few-shot | 0.51 | Moderate |
| LLaMA zero-shot | LLaMA CoT | 0.38 | Fair |
| LLaMA few-shot | LLaMA CoT | 0.32 | Fair |
| Qwen zero-shot | Qwen few-shot | 0.02 | Near-chance |
| Qwen zero-shot | Qwen CoT | 0.44 | Moderate |
| Qwen few-shot | Qwen CoT | 0.19 | Slight |
| LLaMA zero-shot | Qwen zero-shot | 0.31 | Fair |
| LLaMA zero-shot | Qwen few-shot | 0.08 | Slight |
| LLaMA zero-shot | Qwen CoT | 0.36 | Fair |
| LLaMA few-shot | Qwen zero-shot | 0.14 | Slight |
| LLaMA few-shot | Qwen few-shot | **−0.02** | **Below chance** |
| LLaMA few-shot | Qwen CoT | 0.29 | Fair |
| LLaMA CoT | Qwen zero-shot | 0.22 | Fair |
| LLaMA CoT | Qwen few-shot | 0.11 | Slight |
| LLaMA CoT | Qwen CoT | 0.27 | Fair |

Three findings merit emphasis. First, intra-model consistency is substantially higher than inter-model consistency: LLaMA runs share κ values of 0.32–0.51, while cross-model pairs peak at 0.36 and frequently fall below 0.20. Second, within Qwen, few-shot dramatically reduces consistency (Qwen zero-shot vs. few-shot: κ = 0.02), confirming that the few-shot exemplars fundamentally redirect Qwen's classification behaviour. Third, and most significant, LLaMA few-shot versus Qwen few-shot achieves κ = −0.02 — a value below the expected value under random labelling, indicating systematic opposing biases introduced by the same few-shot examples acting on architecturally different models.

### 5.4 Cross-Agreement Structural Analysis

The stratified provision distribution (100 provisions per agreement) reveals substantive structural differences across agreements (Table 4).

**Table 4. Provision Distribution by Category and Agreement (Stratified Sample, %)**

| Category | RCEP | AHKFTA | AANZFTA |
|----------|------|--------|---------|
| General Provisions | 21 | 18 | 22 |
| Rules of Origin | 9 | 28 | 12 |
| Customs Procedures | 14 | 16 | 13 |
| Tariff Commitments | 8 | 11 | 9 |
| Dispute Settlement | 10 | 7 | 11 |
| Trade in Services | 12 | 0 | 11 |
| Investment | 9 | 0 | 8 |
| Non-Tariff Measures | 7 | 10 | 7 |
| Intellectual Property | 4 | 5 | 3 |
| Sanitary & Phytosanitary | 3 | 4 | 3 |
| Other | 3 | 1 | 1 |

The most striking difference is AHKFTA's Rules of Origin concentration (28% vs. 9–12% for the other agreements). AHKFTA is a goods-only agreement with no services, investment, or dispute settlement chapters beyond basic consultation provisions, meaning its provision space is concentrated in the chapters that goods-trade agreements do include. RCEP and AANZFTA exhibit more balanced distributions reflecting their comprehensive scope.

### 5.5 Attribute Extraction Results

Hybrid attribute extraction from Rules of Origin provisions reveals the following cross-agreement design pattern:

**Table 5. Rules of Origin Attribute Comparison**

| Attribute | RCEP | AHKFTA | AANZFTA |
|-----------|------|--------|---------|
| RVC threshold | 40% | 40% | 40% |
| Primary CTC rule | CTH (heading-level) | CC (chapter-level) | CTH/CTSH (variable) |
| De minimis | 10% | 10% | 10% |
| HS scope | All chapters | All chapters | Selected chapters |

Notably, while all three agreements converge on a 40% RVC threshold and 10% de minimis tolerance, they diverge on the CTC rule family. AHKFTA's requirement of Change in Chapter (CC) — requiring imported inputs to change at the 2-digit HS code level — imposes substantially stricter transformation requirements than RCEP's Change in Tariff Heading (CTH, 4-digit level). A manufacturer sourcing textiles (HS Chapter 52) would satisfy RCEP's CTH test by producing garments (HS Chapter 62, different heading) but might satisfy AHKFTA's CC requirement only if production involves a change at the 2-digit chapter level — which is often the same transformation but sometimes requires additional processing stages. AANZFTA's delegation of many threshold specifications to product-specific schedules (Annex 2) renders systematic extraction incomplete from main-text provisions.

### 5.6 Convergence Analysis

Entropy-based convergence signals (Table 6) indicate that structural convergence and fragmentation are category-specific rather than uniform across the treaty corpus.

**Table 6. Category-Level Convergence Signal (Normalised Entropy)**

| Category | Entropy | Signal |
|----------|---------|--------|
| General Provisions | 0.89 | Convergent |
| Dispute Settlement | 0.87 | Convergent |
| Customs Procedures | 0.82 | Moderately convergent |
| Sanitary & Phytosanitary | 0.79 | Moderately convergent |
| Trade in Services | 0.61 | Fragmented |
| Investment | 0.58 | Fragmented |
| Non-Tariff Measures | 0.56 | Fragmented |
| Intellectual Property | 0.52 | Fragmented |
| Rules of Origin | 0.43 | Highly fragmented |
| Tariff Commitments | 0.38 | Highly fragmented |

*Normalised entropy computed as H / H_max where H_max = log(3) for three agreements. Higher values indicate more equal provision distribution across agreements (convergence).*

General Provisions (0.89) and Dispute Settlement (0.87) exhibit the highest convergence, suggesting that definitional structures and consultation/panel procedures have moved towards a regional template — consistent with the ASEAN-centred negotiating architecture that underlies all three agreements. Tariff Commitments (0.38) and Rules of Origin (0.43) are the most fragmented, reflecting each agreement's pursuit of distinct domestic policy objectives and the differing negotiating priorities of the parties.

---

## 6. Discussion

### 6.1 The Chain-of-Thought Asymmetry: Architectural Interpretation

The opposing effect of CoT prompting on LLaMA 3.3 70B and Qwen 3 32B is the most theoretically significant finding of this study. We propose two complementary explanations.

**Architectural alignment.** Qwen 3 32B is trained as a thinking model — its pre-training and fine-tuning explicitly include extended reasoning trace generation as a target behaviour. When a CoT instruction is added to the prompt, the model is being directed to use a capability it was specifically optimised for. LLaMA 3.3 70B, by contrast, is a standard instruction-tuned decoder not trained to produce reasoning traces. The CoT instruction therefore activates a behaviour pattern that is not natively supported, potentially causing the model to generate verbose reasoning that is inconsistent with its classification commitments — what we might call *instructed overthinking*.

**Task complexity and token budget.** Legal text classification of FTA provisions is a moderate-difficulty semantic task: it requires domain knowledge but does not require multi-step mathematical reasoning. Wei et al.'s [2022] original CoT findings were demonstrated primarily on multi-step arithmetic and symbolic reasoning tasks. Liu et al.'s [2024] evidence that CoT reduces performance on tasks where step-by-step deliberation *hurts* human performance is relevant here: FTA classification may involve holistic semantic judgements that benefit from direct label commitment rather than sequential reasoning. Qwen's thinking model training may provide the model with a more disciplined form of reasoning that avoids this pitfall; LLaMA's improvised CoT may generate text that commits the model to an incorrect intermediate framing.

This finding has direct practical implications: **prompt strategy must be selected with reference to model architecture**, not applied uniformly. A practitioner applying a CoT prompt that works well for Qwen to LLaMA — or vice versa — would inadvertently select the worst configuration for one of the models.

### 6.2 Inter-Model Disagreement and Ensemble Validation

The κ = −0.02 finding for LLaMA few-shot vs. Qwen few-shot requires careful interpretation. The negative value does not indicate that either model is performing poorly in absolute terms; both achieve 58–68% accuracy against the gold standard. Rather, it indicates that each model's errors are concentrated on provisions where the other model is correct — a pattern of *complementary* rather than *correlated* errors.

This is both a methodological warning and, potentially, an opportunity. The methodological warning: a practitioner who runs two models as a "cross-check" and treats agreement as confirmation of correctness will be misled. Given κ ≈ 0, agreement between LLaMA few-shot and Qwen few-shot conveys almost no additional evidence of correctness beyond what either model achieves alone. Ensemble strategies such as majority voting, which assume correlated errors to be unlikely, are not valid for this pair of configurations.

The potential opportunity: if errors are genuinely complementary — if the provisions that LLaMA classifies incorrectly are disproportionately those that Qwen classifies correctly — then an oracle that could select the better model per-provision would substantially outperform either model alone. Developing such a model-selection meta-classifier — potentially based on provision characteristics such as length, syntactic complexity, or semantic density — is a promising direction for future work.

The mechanism behind the complementary error pattern is likely the different signals that few-shot examples activate in each model. The same two exemplars that guide LLaMA toward a Rules-of-Origin interpretation pattern may prime Qwen away from that pattern, given the different inductive biases of their respective fine-tuning procedures. This is consistent with the observation that Qwen's few-shot run shows near-zero κ with Qwen's zero-shot run (κ = 0.02), indicating that the exemplars nearly invert Qwen's baseline classification behaviour.

### 6.3 Policy and Analytical Implications

At 70% accuracy, the pipeline is best characterised as a **triage-grade instrument**: it is sufficiently reliable to reduce analyst time-to-insight from days to hours by flagging the most likely policy category for each provision, while retaining the expectation that analyst review of flagged provisions against the source text is required before any compliance or negotiation conclusion is drawn. This is consistent with the use cases described in the human-AI collaboration literature on legal technology, where LLMs serve as a first-pass filter rather than a final decision-maker.

The structural findings — AHKFTA's Rules of Origin concentration, the CC vs. CTH divergence, the convergence of General Provisions and Dispute Settlement — are substantively meaningful from a trade policy perspective. The emergence of a convergent template in definitional and dispute resolution chapters, while RoO and tariff structures remain fragmented, is consistent with the "shallow convergence" hypothesis in regional integration research: that procedural harmonisation precedes substantive harmonisation because it is less politically costly. The quantitative confirmation of this pattern from provision-level text data, rather than from aggregate agreement-depth scoring, represents a contribution to the computational trade policy literature.

### 6.4 Limitations

Several limitations bound the generalisability of these findings.

**Gold set size.** The 50-provision gold set, while stratified, is small relative to the full 3,980-provision corpus. Category-level F1 estimates for rare categories (Intellectual Property: 3 gold provisions; Sanitary and Phytosanitary: 3 provisions) have high variance and should be interpreted cautiously.

**Annex coverage gap.** Quantitative tariff commitments — duty rates, phase-out timelines, staging categories — are predominantly located in tariff schedules (Annex 1 equivalents) that span hundreds of rows of structured table data. Our extraction pipeline preserves these as text fragments rather than structured records, meaning that the Tariff Commitments category in our classification corpus is largely composed of commitment framework provisions rather than actual rate schedules. Systematic extraction of numeric tariff data would require a dedicated table-extraction pipeline not developed here.

**Single-labeller gold set.** The gold set was labelled by the project author. Inter-human agreement was not measured. While the labelling followed a documented taxonomy and was cross-checked against source text, the absence of a second human labeller prevents us from quantifying human-level disagreement on this task — which would provide a natural ceiling for model performance.

**Corpus size.** Three agreements, while tractable for a systematic evaluation study, are insufficient to support generalisable claims about the full Asia-Pacific FTA landscape. Extending the corpus to include CPTPP, AFTA, and major bilateral agreements such as the China-ASEAN FTA and Singapore-Australia FTA would substantially strengthen convergence/fragmentation findings.

**API quota constraints.** Groq's free-tier rolling daily quota (100,000 tokens per 24 hours for LLaMA 3.3 70B) prevents simultaneous execution of all classification and validation runs. CoT runs — which generate ~1,400 tokens per provision — consume the daily quota within a single 100-provision run. This is a reproducibility constraint that would be resolved by paid inference access.

---

## 7. Conclusion

We have presented a computational pipeline for the extraction, classification, and cross-agreement comparison of FTA provisions, systematically evaluated across six model-strategy combinations on a hand-labelled gold set. The best-performing configuration — Qwen 3 32B with chain-of-thought prompting — achieves 70.0% accuracy and 0.693 macro-F1, a level appropriate for analyst triage but not for autonomous compliance determination.

The study's primary methodological contribution is the demonstration of an architectural asymmetry in chain-of-thought prompting effects: CoT substantially improves a thinking-model architecture (Qwen 3 32B) while substantially degrading a standard decoder architecture (LLaMA 3.3 70B). This finding implies that prompt strategy must be co-selected with model architecture, and that CoT should not be applied as a universal improvement heuristic in legal classification tasks. The near-zero inter-model κ finding (κ = −0.02 for the LLaMA few-shot vs. Qwen few-shot pair, despite similar accuracy) invalidates naïve ensemble cross-validation approaches and points toward complementary error structures that may be exploitable through meta-classifier approaches.

The substantive contribution — cross-agreement structural analysis of RCEP, AHKFTA, and AANZFTA — identifies a pattern of selective convergence: procedural categories (General Provisions, Dispute Settlement) are converging toward a regional template, while economically sensitive categories (Rules of Origin, Tariff Commitments) remain highly fragmented, with each agreement maintaining a distinct design logic.

Future directions include: extension to additional agreements to test convergence/fragmentation hypotheses at regional scale; development of per-provision meta-classifiers to exploit complementary error structures across models; structured extraction of tariff schedule data; and fine-tuning of smaller open-weight models on the FTA-specific taxonomy to reduce dependence on large-model API access.

The full pipeline, corpus, and evaluation code are released to support reproducibility and extension by the research community.

---

## Acknowledgements

The author thanks the Groq team for providing free-tier access to LLaMA 3.3 70B and Qwen 3 32B inference. Source FTA documents are publicly available from the ASEAN Secretariat, the WTO RTA database, and respective national trade ministries.

---

## References

**Ariai, F., Mackenzie, J., and Demartini, G.** (2024). Natural Language Processing for the Legal Domain: A Survey of Tasks, Datasets, Models, and Challenges. *ACM Computing Surveys*. arXiv:2410.21306. DOI:10.1145/3777009.

**Artstein, R. and Poesio, M.** (2008). Inter-Coder Agreement for Computational Linguistics. *Computational Linguistics*, 34(4), 555–596.

**Baier, S. L. and Regmi, N. R.** (2023). Using Machine Learning to Capture Heterogeneity in Trade Agreements. *Open Economies Review*, 34(4), 863–894. DOI:10.1007/s11079-022-09685-3.

**Baldwin, R. and Jaimovich, D.** (2012). Are Free Trade Agreements Contagious? *Journal of International Economics*, 88(1), 1–16.

**Bhagwati, J.** (1995). U.S. Trade Policy: The Infatuation with Free Trade Agreements. In C. Barfield (Ed.), *The Dangerous Drift to Preferential Trade Agreements*. AEI Press, Washington D.C.

**Breinlich, H., Corradi, V., Rocha, N., Ruta, M., Santos Silva, J. M. C., and Zylkin, T.** (2022). Machine Learning in International Trade Research: Evaluating the Impact of Trade Agreements. World Bank Policy Research Working Paper No. 9629 / CEPR Discussion Paper No. 17325.

**Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D.** (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 1877–1901. arXiv:2005.14165.

**Chalkidis, I., Androutsopoulos, I., and Aletras, N.** (2020). LEGAL-BERT: The Muppets Straight Out of Law School. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, 2898–2904.

**Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D., and Aletras, N.** (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. In *Proceedings of the 60th Annual Meeting of the ACL*, 4310–4330. DOI:10.18653/v1/2022.acl-long.297.

**Cohen, J.** (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37–46.

**Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K.** (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of NAACL-HLT 2019*, 4171–4186.

**Grattafiori, A. et al. [Llama Team, AI @ Meta].** (2024). The Llama 3 Herd of Models. arXiv:2407.21783.

**Guha, N., Nyarko, J., Ho, D. E., Ré, C., Chilton, A., Narayana, A., Chohlas-Wood, A., Peters, A., Waldon, B., Rockmore, D. N., Zambrano, D., Talisman, D., Hoque, E., Surani, F., Fagan, F., Sarfaty, G., Dickinson, G. M., Porat, H., Hegland, J., Wu, J., Nudell, J., Niklaus, J., Nay, J., Choi, J. H., Tobia, K., Hagan, M., Ma, M., Livermore, M., Rasumov-Rahe, N., Holzenberger, N., Kolt, N., Henderson, P., Rehaag, S., Goel, S., Gao, S., Williams, S., Gandhi, S., Zur, T., Iyer, V., and Li, Z.** (2023). LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. *Advances in Neural Information Processing Systems (NeurIPS 2023)*. arXiv:2308.11462.

**Hendrycks, D., Burns, C., Chen, A., and Ball, S.** (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Understanding. In *Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021)*, Datasets and Benchmarks Track.

**Kim, N. and Park, C.** (2023). Inter-Annotator Agreement in the Wild: Uncovering Its Emerging Roles and Considerations in Real-World Scenarios. arXiv:2306.14373.

**Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y.** (2022). Large Language Models are Zero-Shot Reasoners. *Advances in Neural Information Processing Systems (NeurIPS 2022)*, 35. arXiv:2205.11916.

**Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., and Kiela, D.** (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 9459–9474. arXiv:2005.11401.

**Li, H., Chen, Y., Hu, Y., Ai, Q., Chen, J., Yang, X., Yang, J., Wu, Y., Liu, Z., and Liu, Y.** (2025). LexRAG: Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation Conversation. In *Proceedings of the 48th International ACM SIGIR Conference*. arXiv:2502.20640.

**Liu, R., Geng, J., Wu, A. J., Sucholutsky, I., Lombrozo, T., and Griffiths, T. L.** (2024). Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse. arXiv:2410.21333.

**Pipitone, N. and Alami, G. H.** (2024). LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain. arXiv:2408.10343.

**Qwen Team, Alibaba Cloud.** (2025). Qwen3 Technical Report. arXiv:2505.09388.

**Qwen Team, Alibaba Cloud.** (2024). Qwen2.5 Technical Report. arXiv:2412.15115.

**Reimers, N. and Gurevych, I.** (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of EMNLP-IJCNLP 2019*, 3982–3992.

**Sahoo, P., Singh, A. K., Saha, S., Jain, V., Mondal, S., and Chadha, A.** (2024). A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications. arXiv:2402.07927.

**Savelka, J. and Ashley, K. D.** (2023). The Unreasonable Effectiveness of Large Language Models in Zero-Shot Semantic Annotation of Legal Texts. *Frontiers in Artificial Intelligence*, 6, Article 1279794. DOI:10.3389/frai.2023.1279794.

**Schulhoff, S. et al.** (2024). The Prompt Report: A Systematic Survey of Prompting Engineering Techniques. arXiv:2406.06608.

**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Poliakoff, I.** (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30.

**Vatsal, S. and Dubey, H.** (2024). A Survey of Prompt Engineering Methods in Large Language Models for Different NLP Tasks. arXiv:2407.12994.

**Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D.** (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems (NeurIPS 2022)*, 35. arXiv:2201.11903.

**WTO.** (2024). Regional Trade Agreements Database. World Trade Organization. Retrieved from https://rtais.wto.org.

---

*Manuscript submitted April 2026. All code, data, and supplementary materials available at: [repository URL].*

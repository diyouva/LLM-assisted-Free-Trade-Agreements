# Automating Free Trade Agreement Analysis: An LLM-Based Pipeline for Provision Classification and Cross-Agreement Comparison

**Diyouva Christa Novith**
Heinz College of Information Systems and Public Policy, Carnegie Mellon University, Pittsburgh, PA, USA

---

## Abstract

The proliferation of overlapping Free Trade Agreements (FTAs) in the Asia-Pacific creates a "spaghetti bowl" of legal obligations that imposes substantial analytical burdens on trade policy professionals. This paper presents a computational pipeline that applies open-weight Large Language Models to the extraction, classification, and cross-agreement comparison of FTA provisions at scale. We construct a corpus of 4,059 provisions from three major agreements — RCEP, AHKFTA, and AANZFTA — and evaluate two architecturally distinct LLMs (LLaMA 3.3 70B and Qwen 3 32B) across three prompt strategies (zero-shot, few-shot, chain-of-thought) against a 50-provision hand-labelled gold set. Best accuracy reaches 48.0% (LLaMA zero-shot); best macro-F1 reaches 0.442 (Qwen chain-of-thought). Cohen's κ analysis on aligned cohorts shows moderate-to-substantial inter-run agreement (κ = 0.58–0.84), indicating that classification instability is less severe than absolute accuracy limitations. Cross-agreement structural analysis reveals that AHKFTA concentrates 48% of its classified provisions in Rules of Origin compared to 24% for RCEP and 39% for AANZFTA under the few-shot stratified run, though this distribution is sensitive to classifier bias. The pipeline is best understood as a reproducible analyst-triage tool: it reduces provision-level analysis from weeks to hours but requires human verification for any compliance or legal conclusion.

**Keywords:** Large Language Models; Free Trade Agreements; Legal Text Classification; Prompt Engineering; Retrieval-Augmented Generation; Trade Policy; Cohen's Kappa

---

## 1. Introduction

Free Trade Agreements have become the dominant instrument of international trade governance. The World Trade Organization records over 350 agreements in force as of 2024, with the Asia-Pacific region maintaining dozens of overlapping instruments [WTO, 2024]. Each agreement spans hundreds of pages covering tariff schedules, rules of origin, services liberalisation, investment protection, dispute settlement, and regulatory harmonisation. Comparing provisions across even a small number of agreements — the task regularly demanded of trade negotiators, customs authorities, and policy analysts — is what Bhagwati [1995] termed the "spaghetti bowl" problem.

The emergence of capable, openly accessible Large Language Models (LLMs) offers a tractable path toward automating the most labour-intensive stages of FTA analysis. Prior work in legal NLP has demonstrated transformer-based models' capacity for contract understanding [Hendrycks et al., 2021], legal judgment prediction [Chalkidis et al., 2022], and statutory interpretation [Savelka and Ashley, 2023]. However, the application of LLMs to international trade agreements — characterised by multilateral scope, highly technical vocabulary, and structurally heterogeneous document formats — remains largely unexplored. To our knowledge, no prior study has benchmarked LLM classification strategies across multiple FTAs against a hand-labelled gold set or examined inter-model agreement in this domain.

This paper addresses that gap with four contributions:

1. **A reproducible FTA provision corpus and evaluation protocol.** We extract 4,059 provisions from three Asia-Pacific FTAs using a multi-library PDF extraction chain and construct a stratified 50-provision gold set against an 11-category WTO/UNCTAD taxonomy.

2. **A systematic multi-model, multi-strategy classification benchmark.** We evaluate LLaMA 3.3 70B and Qwen 3 32B across zero-shot, few-shot, and chain-of-thought strategies, yielding six complete evaluation runs on the same gold set.

3. **An empirical prompt-sensitivity finding.** Prompt strategy materially changes validation outcomes (16 percentage points in accuracy, 0.115 in macro-F1), but no configuration produces high-confidence provision-level labels. The best-performing strategy differs between the two models.

4. **A cross-agreement structural analysis.** Using Retrieval-Augmented Generation (RAG), we identify category-level differences across agreements and examine convergence patterns, while documenting the sensitivity of these findings to classifier choice.

---

## 2. Background and Related Work

### 2.1 The FTA Analytical Challenge

The proliferation of preferential trade agreements since the 1990s has been extensively documented [Baldwin and Jaimovich, 2012]. Quantitative trade economists have addressed this through aggregate indicators — gravity models to estimate agreement effects [Breinlich et al., 2022] or machine learning to score agreement depth [Baier and Regmi, 2023]. These approaches treat agreements as objects to be scored rather than texts to be read, and do not operate at provision level. Provision-level analysis would reveal not only *how deep* agreements are but *where* they differ and *how* they are structured — a capability that requires natural language processing.

### 2.2 Legal Text Classification

Legal NLP has advanced rapidly with domain-adapted transformers. LEGAL-BERT [Chalkidis et al., 2020] demonstrated that continued pre-training on legal corpora improves multi-label classification of judgments and contract clauses. The LexGLUE benchmark [Chalkidis et al., 2022] systematised evaluation across seven legal tasks, and LegalBench [Guha et al., 2023] extended this to 162 tasks targeting legal reasoning. Savelka and Ashley [2023] showed that modern LLMs exhibit surprising zero-shot capability on legal semantic annotation.

Critically, prior benchmarks focus on case law, statutory text, or commercial contracts. International trade agreements — multilateral treaty instruments with chapter-article-annex hierarchies, HS code cross-references, and cascading definitional dependencies — have not been a primary target of legal NLP research.

### 2.3 Prompt Engineering

Few-shot in-context learning [Brown et al., 2020] and chain-of-thought prompting [Wei et al., 2022] have established that eliciting capable behaviour through natural language instructions is practical across settings. However, CoT gains are not universal: they emerge primarily in large models and can reduce performance on certain task types [Liu et al., 2024]. The architectural distinction between standard instruction-tuned decoders and explicitly designed "thinking models" — which emit reasoning traces as a trained behaviour — introduces a variable not addressed by prior CoT literature. We examine this distinction empirically.

### 2.4 Retrieval-Augmented Generation

RAG [Lewis et al., 2020] extends LLM generation by first retrieving relevant passages from an external knowledge base. In legal settings, RAG has been applied to question answering [Pipitone and Alami, 2024] and multi-turn consultation [Li et al., 2025]. In our application, RAG serves a cross-agreement comparison function: for each policy category, we retrieve semantically similar provisions from each agreement and prompt the LLM for structured comparative analysis.

### 2.5 Inter-Annotator Agreement

Cohen's κ [Cohen, 1960] is the standard measure of pairwise inter-rater agreement corrected for chance. Kim and Park [2023] documented a shift from treating low inter-annotator agreement as a problem to treating it as a signal of genuine label ambiguity. We extend this framing to inter-*model* agreement, treating κ between LLM runs as a diagnostic for whether models capture the same underlying construct.

---

## 3. Data

### 3.1 Agreement Corpus

We analyse three Free Trade Agreements spanning the Asia-Pacific regional architecture:

| Agreement | Parties | In Force | Chapters |
|-----------|---------|----------|----------|
| **RCEP** | 15 nations (ASEAN-10 + AU, CN, JP, KR, NZ) | 2022 | 20 (comprehensive) |
| **AHKFTA** | ASEAN-10 + Hong Kong, China | 2019 | Limited (goods-focused) |
| **AANZFTA** | ASEAN-10 + Australia + New Zealand | 2010 | Comprehensive + 2014 Protocol |

Seven source PDFs are processed, including main agreement texts and selected annexes. AHKFTA's source documents are bitmap scans of signed originals, requiring OCR.

### 3.2 Provision Extraction

Extraction proceeds through a three-stage fallback chain: pdfplumber (primary), PyMuPDF (encoding fallback), and Tesseract OCR (scanned pages). Character-count filters discard provisions shorter than 80 characters (headings, artefacts). Clause segmentation uses regex patterns matching Article, Chapter, Section, Rule, and Annex headers to identify provision boundaries. Each provision is stored with structured metadata: `id`, `agreement`, `doc_type`, `chapter`, `article`, `paragraph_idx`, `text`, and `char_count`.

The resulting corpus comprises **4,059 provisions**: RCEP 2,171 (53.5%), AANZFTA 1,526 (37.6%), and AHKFTA 362 (8.9%). The pronounced RCEP majority reflects its broader scope and detailed annexes; we address this bias through stratified sampling.

### 3.3 Gold Set Construction

We construct a 50-provision stratified gold set for model evaluation. Provisions are balanced across agreements (RCEP: 18, AHKFTA: 16, AANZFTA: 16) and manually labelled by the author following the WTO/UNCTAD chapter taxonomy. The category distribution in the gold set is: Tariff Commitments (11), General Provisions (10), Dispute Settlement (10), Customs Procedures (5), Rules of Origin (5), Trade in Services (4), Other (2), Intellectual Property (1), Investment (1), Non-Tariff Measures (1), Sanitary and Phytosanitary Measures (0).

The gold set is small relative to the corpus and covers some categories sparsely. This constrains the reliability of per-category F1 estimates, particularly for categories with fewer than 5 gold instances. We return to this limitation in Section 6.

---

## 4. Methodology

### 4.1 Classification Framework

Classifications are performed via the Groq inference API, which provides free-tier access to both models. We construct a **stratified sample** of 100 provisions per agreement (300 total, seed=42) for cross-agreement comparison, and use reproducible random sampling (seed=42) for the main 200-provision benchmark runs. All model-strategy combinations are also run on the exact 50-provision validation cohort for gold-set evaluation.

### 4.2 Policy Category Taxonomy

We adopt an 11-category taxonomy aligned with the standard WTO/UNCTAD FTA chapter structure: (1) Tariff Commitments, (2) Rules of Origin, (3) Non-Tariff Measures, (4) Trade in Services, (5) Investment, (6) Dispute Settlement, (7) Intellectual Property, (8) Customs Procedures, (9) Sanitary and Phytosanitary Measures, (10) General Provisions / Definitions, (11) Other.

### 4.3 Models

**LLaMA 3.3 70B** [Grattafiori et al., 2024] is a 70-billion parameter instruction-tuned decoder from Meta AI, trained via RLHF and direct preference optimisation. It does not employ explicit reasoning trace generation.

**Qwen 3 32B** [Qwen Team, 2025] is a 32-billion parameter "thinking model" from Alibaba Cloud that emits structured `<think>...</think>` reasoning tokens before a final answer. This architectural distinction — an explicit reasoning phase as trained behaviour — is central to interpreting our chain-of-thought results.

Both models are accessed via Groq's free inference tier under separate daily quotas.

*Note on model selection:* The original design specified Gemini 1.5 Flash as the second model. Gemini's free-tier quota was exhausted after approximately 200 API calls, insufficient for the experimental matrix. Qwen 3 32B was substituted because it is available on the same endpoint under a separate quota and represents a genuinely different architecture.

### 4.4 Prompt Strategies

**Zero-shot.** Category list with descriptions plus provision text. No examples. Approximate token count: ~400 per call.

**Few-shot.** Two curated examples prepended: one Rules of Origin and one Tariff Commitments provision. Approximate token count: ~900 per call.

**Chain-of-thought.** Instruction to reason step-by-step about the main legal subject, key obligations, and best category match before committing to a label. Approximate token count: ~1,400 per call.

### 4.5 Cross-Agreement Comparison via RAG

For each policy category, we retrieve the top-3 semantically similar provisions per agreement using `all-MiniLM-L6-v2` embeddings [Reimers and Gurevych, 2019] stored in a ChromaDB vector store (cosine distance). Retrieved provisions are passed to Qwen 3 32B with a structured prompt requesting analysis along five dimensions: similarities, differences, flexibility vs. rigidity, convergence vs. fragmentation, and policy implications.

### 4.6 Attribute Extraction

We extract structured attributes from Rules of Origin and Tariff Commitment provisions using a hybrid approach. Numeric fields (RVC thresholds, de minimis percentages, phase-out years) are extracted via regex. Categorical fields (CTC rule family, HS scope, staging category) are extracted via LLM prompt. Regex values take priority on overlap.

### 4.7 Evaluation Metrics

**Accuracy:** proportion correctly classified (n=50). **Macro-F1:** unweighted mean F1 across all categories, our primary metric since it penalises models that ignore rare categories. **Cohen's κ:** pairwise inter-run agreement corrected for chance. **Entropy-based convergence:** normalised Shannon entropy of the provision-count distribution across agreements per category, serving as a proxy for structural convergence.

---

## 5. Results

### 5.1 Classification Performance

**Table 1. Validation Results (n = 50)**

| Model | Strategy | Accuracy | Macro-F1 |
|-------|----------|----------|----------|
| LLaMA 3.3 70B | Zero-shot | **0.480** | 0.431 |
| Qwen 3 32B | Chain-of-thought | 0.460 | **0.442** |
| Qwen 3 32B | Zero-shot | 0.380 | 0.424 |
| Qwen 3 32B | Few-shot | 0.380 | 0.373 |
| LLaMA 3.3 70B | Few-shot | 0.340 | 0.336 |
| LLaMA 3.3 70B | Chain-of-thought | 0.320 | 0.327 |

No configuration exceeds 48% accuracy. The performance range spans 16 percentage points in accuracy and 0.115 in macro-F1. LLaMA zero-shot achieves the highest accuracy, while Qwen CoT achieves the highest macro-F1. That split suggests the metric choice depends on whether the downstream task values overall hit rate or balanced performance across rare categories.

### 5.2 Prompt Sensitivity

The prompt-sensitivity finding is more nuanced than a simple "CoT helps" narrative. For Qwen, CoT produces the highest macro-F1 (0.442) but only a marginal improvement over zero-shot (0.424) — a difference of 0.018, equivalent to approximately one additional correct classification. For LLaMA, CoT produces the *worst* performance on both metrics (accuracy 0.320, macro-F1 0.327), suggesting that explicit step-by-step reasoning can degrade a standard decoder when the task involves ambiguous category boundaries.

Few-shot performs poorly for both models on the current gold set. An important methodological observation is that both few-shot examples in the prompt are from goods-trade categories (Rules of Origin and Tariff Commitments). This choice biases the few-shot classifier: `qwen_few_shot` classifies 26.5% of provisions as Rules of Origin compared to 15.5% for `qwen_zero_shot`. The effect is amplified in the stratified run, where `qwen_few_shot_stratified` labels 37.0% (111/300) of provisions as Rules of Origin. This bias propagates into downstream analyses that use the stratified run as input.

### 5.3 Inter-Run Agreement

**Table 2. Pairwise Cohen's κ (aligned cohorts)**

| Run A | Run B | n | κ | Interpretation |
|-------|-------|---|---|----------------|
| LLaMA zero-shot | Qwen zero-shot | 200 | 0.702 | Substantial |
| Qwen zero-shot | Qwen few-shot | 200 | 0.689 | Substantial |
| LLaMA zero-shot | LLaMA few-shot | 200 | 0.668 | Substantial |
| LLaMA zero-shot | Qwen few-shot | 200 | 0.651 | Substantial |
| LLaMA CoT | Qwen CoT | 100 | 0.640 | Substantial |
| LLaMA few-shot | Qwen few-shot | 200 | 0.582 | Moderate |

On the validation cohort (n=50), inter-run agreement is even higher: LLaMA zero-shot vs. Qwen zero-shot reaches κ = 0.835. Once runs are compared on exactly matched cohorts, agreement is moderate to substantial rather than near-random. The dominant bottleneck is low absolute validation performance against gold labels, not pairwise instability.

### 5.4 Cross-Agreement Structural Analysis

The stratified provision distribution (100 per agreement, classified with Qwen 3 32B few-shot) reveals substantive structural differences, though the absolute counts must be interpreted with caution given the few-shot bias documented in §5.2.

**Table 3. Provision Distribution by Category and Agreement (Stratified Sample, Qwen Few-Shot)**

| Category | RCEP | AHKFTA | AANZFTA | Total |
|----------|-----:|-------:|--------:|------:|
| Rules of Origin | 24 | 48 | 39 | 111 |
| Tariff Commitments | 6 | 31 | 7 | 44 |
| Trade in Services | 20 | 0 | 10 | 30 |
| Dispute Settlement | 6 | 0 | 22 | 28 |
| Customs Procedures | 6 | 6 | 7 | 19 |
| General Provisions | 5 | 9 | 3 | 17 |
| Investment | 8 | 4 | 5 | 17 |
| Intellectual Property | 12 | 0 | 2 | 14 |
| Non-Tariff Measures | 6 | 1 | 4 | 11 |
| SPS Measures | 5 | 1 | 0 | 6 |
| Other | 2 | 0 | 1 | 3 |
| **Total** | **100** | **100** | **100** | **300** |

Three structural patterns are robust across classifier configurations:

**AHKFTA is a goods-focused agreement.** It has zero classified provisions in Trade in Services, Dispute Settlement, and Intellectual Property across all classifier runs — consistent with its design as a bilateral goods-trade instrument. This finding is independent of classifier bias because it reflects genuine chapter-level absence.

**Rules of Origin weight diverges sharply.** Even in the less-biased zero-shot runs, AHKFTA provisions are disproportionately classified as Rules of Origin relative to RCEP and AANZFTA. In the random-sampled `qwen_zero_shot` run, AHKFTA accounts for 10/31 RoO provisions despite comprising only 21/200 of the sample (10.5% of sample, 32.3% of RoO), suggesting genuine RoO concentration.

**Investment and services coverage differ by design.** RCEP and AANZFTA have dedicated investment and services chapters; AHKFTA does not. This scope difference directly affects investors choosing between agreements.

### 5.5 Attribute Extraction

Hybrid attribute extraction from 92 stratified provisions (38 Rules of Origin, 54 Tariff Commitments) recovers structured policy-design features:

**Table 4. Rules of Origin Attribute Comparison**

| Attribute | RCEP | AHKFTA | AANZFTA |
|-----------|------|--------|---------|
| RVC threshold | 40% | 40% | Not recovered in main text |
| Primary CTC rule | CTH (heading-level) | CC (chapter-level) | Variable (CTH/CTSH) |
| De minimis | — | 10% | — |

All three agreements converge on a 40% Regional Value Content threshold where recoverable, but diverge on the Change in Tariff Classification rule family. AHKFTA's requirement of Change in Chapter (CC) — requiring transformation at the 2-digit HS code level — imposes substantially stricter requirements than RCEP's Change in Tariff Heading (CTH, 4-digit level). A manufacturer sourcing inputs from one HS chapter would need a more substantial transformation to qualify under AHKFTA than RCEP, despite the identical RVC threshold.

AANZFTA's main-text provisions delegate many threshold specifications to product-specific schedules (Annex 2), rendering systematic extraction from the provision corpus incomplete — itself a finding about agreement design. AANZFTA places a greater compliance-information burden on exporters by distributing rules across annexes rather than consolidating them in main-text provisions.

### 5.6 Convergence and Fragmentation

The entropy-based convergence analysis is sensitive to classifier choice. Using the stratified few-shot run, Customs Procedures appears most convergent (entropy ratio 0.998) while Intellectual Property appears most fragmented (0.373). However, the convergence of Rules of Origin (ratio 0.966) is partly an artefact of the few-shot classifier labelling a high proportion of provisions as RoO across all three agreements.

Cross-checking against the random-sampled zero-shot runs (which are less affected by few-shot bias but not stratified), three qualitative patterns are more defensible:

1. **AHKFTA's structural absence** from Trade in Services, Investment, Dispute Settlement, and Intellectual Property is robust across all classifier configurations — a genuine fragmentation signal reflecting the agreement's goods-only design.

2. **Customs Procedures** provisions are relatively evenly distributed across agreements in all runs, suggesting procedural harmonisation.

3. **Tariff Commitments and Rules of Origin** show AHKFTA concentration in all runs, though the magnitude varies with classifier choice.

The convergence analysis should be interpreted as suggestive rather than definitive, given the 32–48% validation accuracy of the underlying classifier.

---

## 6. Discussion

### 6.1 Prompt Strategy is Empirical, Not Theoretical

Earlier iterations of this project argued for a strong architectural asymmetry in CoT effects: that thinking models (Qwen) benefit from CoT while standard decoders (LLaMA) are harmed by it. The current evidence is more modest. Qwen CoT edges out Qwen zero-shot on macro-F1 by 0.018 — a difference that could be driven by the classification of a single provision on a 50-item gold set. LLaMA CoT does underperform LLaMA zero-shot more substantially (0.104 macro-F1 gap), but the small sample size prevents strong architectural conclusions.

The operational implication is clear: prompt strategy must be tuned empirically for each pipeline configuration, and changes to extraction, sampling, or cohort alignment can change the apparent winner.

### 6.2 Few-Shot Example Selection Creates Systematic Bias

The finding that both few-shot examples come from goods-trade categories and measurably inflate RoO classification rates is a methodological caution for applied legal NLP. In-context examples do not merely "help" the model — they anchor its output distribution. When the examples are thematically narrow, they create a systematic bias that propagates through every downstream analysis. Future work should either diversify the example set across all 11 categories (which increases token cost and may exceed rate limits) or rely on zero-shot classification for analyses where distributional accuracy matters.

### 6.3 Inter-Run Agreement is Healthier Than Absolute Accuracy

The κ analysis yields a constructive finding: once cohorts are exactly aligned, inter-model agreement is moderate to substantial (κ = 0.58–0.84). This means the two models capture broadly similar category signals from the same provisions, even though both models score poorly against gold labels. The bottleneck is therefore the absolute difficulty of the 11-way classification task — likely exacerbated by genuinely ambiguous provisions that could plausibly belong to multiple categories — rather than random disagreement between models.

This has practical implications for ensemble strategies. With κ > 0.6, a majority-vote ensemble of the six runs would primarily reinforce shared errors rather than correct them. The path to better accuracy likely runs through taxonomy redesign (hierarchical or multi-label), larger gold sets, or model fine-tuning — not through ensembling.

### 6.4 Policy Implications

At 32–48% accuracy, the pipeline is a **triage-grade instrument**. It can reduce analyst time-to-insight by flagging likely policy areas and producing aggregate distributions, but every substantive conclusion requires verification against source text.

The structural findings — AHKFTA's goods-only design, the CC vs. CTH divergence in Rules of Origin, the AANZFTA delegation pattern — are substantively meaningful from a trade policy perspective. The CC vs. CTH finding is directly actionable: an exporter operating under both RCEP and AHKFTA faces materially different origin requirements despite the identical 40% RVC threshold, creating a genuine compliance risk that the pipeline surfaces in structured form.

### 6.5 Limitations

**Gold set size and composition.** The 50-provision gold set has 0–1 instances for four of eleven categories. Macro-F1 estimates for these sparse categories have extreme variance. Expanding to 200+ provisions with at least 10 per category would materially strengthen the evaluation.

**Single annotator.** No inter-human κ was measured. We cannot quantify the human-level disagreement ceiling for this task, which would contextualise the 48% machine accuracy.

**Classifier-dependent convergence analysis.** The entropy-based convergence signal operates on LLM-classified labels with 32–48% accuracy. Systematic biases (particularly from few-shot examples) propagate into the convergence analysis. Cross-checking against multiple classifier configurations partially addresses this, but a ground-truth convergence analysis would require provision-level labels for the full stratified sample.

**Annex coverage gap.** Quantitative tariff commitments are predominantly located in schedule annexes not segmented into the provision corpus. The Tariff Commitments category in our classification corpus is largely composed of commitment framework provisions rather than actual rate schedules.

**Three agreements.** Findings are suggestive, not statistically generalisable. Extending to CPTPP, AFTA, and major bilateral agreements would strengthen convergence claims.

**API quota constraints.** CoT runs were capped at 100 provisions (vs. 200 for zero-shot and few-shot) due to Groq's rolling daily token limit, introducing a confound in the comparison of sample sizes across strategies.

**No statistical significance testing.** With n=50, the difference between the best (48%) and worst (32%) accuracy is 8 provisions. No bootstrap confidence intervals or McNemar's test was applied to assess whether observed differences are statistically meaningful.

---

## 7. Conclusion

We have presented a computational pipeline for the extraction, classification, and cross-agreement comparison of FTA provisions, evaluated across six model-strategy combinations. The best accuracy is 48.0% (LLaMA zero-shot) and the best macro-F1 is 0.442 (Qwen CoT) — appropriate for analyst triage but not autonomous legal classification.

Three findings stand out:

**First, prompt strategy interacts with model architecture in operationally meaningful ways.** CoT marginally improves Qwen's macro-F1 but substantially degrades LLaMA's. Few-shot examples create measurable distributional bias. These effects are only discoverable through systematic validation — model documentation alone cannot predict prompt-task interaction.

**Second, the agreements are structurally more different than their shared ASEAN context suggests.** AHKFTA is a goods-only instrument that shares RCEP's 40% RVC threshold but uses a stricter CTC transformation rule (CC vs. CTH). AANZFTA delegates threshold definitions to annexes in ways that resist automated extraction. These are real differences with compliance implications for exporters operating under multiple agreements simultaneously.

**Third, inter-model agreement is moderate to substantial once cohorts are aligned, making absolute accuracy — not pairwise disagreement — the binding constraint.** Future improvement should prioritise taxonomy redesign, larger gold sets, and potentially model fine-tuning over ensemble approaches.

The pipeline is fully reproducible, requires only a free Groq API key, and generalises to any FTA with a PDF text layer. The recommended configuration for aggregate analysis is Qwen 3 32B with chain-of-thought prompting; for provision-level triage, LLaMA 3.3 70B zero-shot.

---

## Acknowledgements

The author thanks the Groq team for free-tier access to LLaMA 3.3 70B and Qwen 3 32B inference. Source FTA documents are publicly available from the ASEAN Secretariat and the WTO RTA database.

---

## References

Artstein, R. and Poesio, M. (2008). Inter-Coder Agreement for Computational Linguistics. *Computational Linguistics*, 34(4), 555–596.

Baier, S. L. and Regmi, N. R. (2023). Using Machine Learning to Capture Heterogeneity in Trade Agreements. *Open Economies Review*, 34(4), 863–894.

Baldwin, R. and Jaimovich, D. (2012). Are Free Trade Agreements Contagious? *Journal of International Economics*, 88(1), 1–16.

Bhagwati, J. (1995). U.S. Trade Policy: The Infatuation with Free Trade Agreements. In C. Barfield (Ed.), *The Dangerous Drift to Preferential Trade Agreements*. AEI Press.

Breinlich, H., Corradi, V., Rocha, N., Ruta, M., Santos Silva, J. M. C., and Zylkin, T. (2022). Machine Learning in International Trade Research. World Bank Policy Research Working Paper No. 9629.

Brown, T. B. et al. (2020). Language Models are Few-Shot Learners. *NeurIPS 2020*, 33, 1877–1901.

Chalkidis, I., Androutsopoulos, I., and Aletras, N. (2020). LEGAL-BERT: The Muppets Straight Out of Law School. *Findings of EMNLP 2020*, 2898–2904.

Chalkidis, I. et al. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding. *ACL 2022*, 4310–4330.

Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37–46.

Grattafiori, A. et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.

Guha, N. et al. (2023). LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. *NeurIPS 2023*.

Hendrycks, D. et al. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Understanding. *NeurIPS 2021*, Datasets Track.

Kim, N. and Park, C. (2023). Inter-Annotator Agreement in the Wild. arXiv:2306.14373.

Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*, 33, 9459–9474.

Li, H. et al. (2025). LexRAG: Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation. *SIGIR 2025*.

Liu, R. et al. (2024). Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse. arXiv:2410.21333.

Pipitone, N. and Alami, G. H. (2024). LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain. arXiv:2408.10343.

Qwen Team, Alibaba Cloud. (2025). Qwen3 Technical Report. arXiv:2505.09388.

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP 2019*, 3982–3992.

Savelka, J. and Ashley, K. D. (2023). The Unreasonable Effectiveness of Large Language Models in Zero-Shot Semantic Annotation of Legal Texts. *Frontiers in AI*, 6, 1279794.

Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*, 35.

WTO. (2024). Regional Trade Agreements Database. Retrieved from https://rtais.wto.org.

---

*All code, data, and evaluation artefacts are publicly available at the project repository.*

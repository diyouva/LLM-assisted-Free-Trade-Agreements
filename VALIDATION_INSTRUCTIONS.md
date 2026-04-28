# How to fill in the validation gold labels

> **Workflow note (April 2026):** current repo flow:
> 1. `python run_pipeline.py --step validation_sample ...`
> 2. manually label `data/results/validation_checked.xlsx` if it exists
>    (fallback: `data/results/validation_set.csv`)
> 3. `python run_pipeline.py --step validation_export`
> 4. rerun the six `*_validation` classifications on `data/raw/validation_provisions.json`
> 5. `python -m src.validation --evaluate`

> **Time required:** 30–45 minutes
> **What you're doing:** reading 50 short legal provisions and assigning each one the single best policy category. This becomes the *ground truth* against which every LLM classification run is scored.

---

## 1. Why this matters

The project compares 6 different LLM classification runs (2 models × 3 prompt strategies). The LLMs often disagree with each other on the same provision. Without a human-labelled gold standard, we can only measure how much the LLMs agree with **each other** — we cannot say which one is **actually correct**.

Your labels in the `gold_category` column become the authoritative answer. The project then reports validation metrics only if this column is filled in and the exact same cohort is reclassified.

---

## 2. Open the file

1. Open `data/results/validation_checked.xlsx` if it exists. If not, open `data/results/validation_set.csv`.
2. You'll see **7 columns**:

| Column | Purpose | Filled by |
|---|---|---|
| `id` | Unique provision ID (e.g. `RCEP_Main Agreement_00042`) | Already filled |
| `agreement` | RCEP / AHKFTA / AANZFTA | Already filled |
| `article` | Article or Rule number if available | Already filled |
| `text_preview` | First ~250 chars of the provision | Already filled |
| **`gold_category`** | **The correct category (you fill this)** | **← YOU** |
| `notes` | Optional — write any uncertainty or reasoning | Optional |

3. Keep the existing text exactly as-is. Only add values to `gold_category` (and optionally `notes`).

---

## 3. How to label each row — the 3-step process

For each provision, read the `text_preview` and pick **exactly one** category from the list below. If a provision seems to touch multiple categories, pick the category that describes its *main legal subject matter* — the thing the provision is fundamentally *about*.

### Step 1 — Read the text
Ignore formatting noise like line breaks or page references. Focus on what the provision is doing.

### Step 2 — Ask yourself "what is this about?"
Use the decision guide below.

### Step 3 — Write the category name in `gold_category`
Use the **exact spelling** shown in the allowed list (case doesn't matter, but spelling does).

---

## 4. The 11 allowed categories — definitions & recognition cues

Copy any of these into `gold_category` exactly:

| Category | What it covers | Recognition cues |
|---|---|---|
| **Tariff Commitments** | Reducing or eliminating customs duties; tariff schedules; preferential rates | "reduce/eliminate customs duties", "tariff schedule", "phase-out", "rate of duty", "MFN rate", "concession" |
| **Rules of Origin** | How to determine whether a good qualifies as "originating" | "originating", "non-originating", "wholly obtained", "RVC", "regional value content", "CTH / CTSH / change in tariff", "de minimis", "cumulation", "product-specific rules" |
| **Non-Tariff Measures** | Non-tariff barriers: quotas, licensing, TBT (technical barriers), import bans | "quantitative restriction", "import licence", "technical regulation", "standards", "prohibit" (non-SPS context), "TBT" |
| **Trade in Services** | Cross-border services trade, sector commitments, modes of supply | "service suppliers", "cross-border supply", "commercial presence", "mode 1/2/3/4", "market access", "schedule of specific commitments" |
| **Investment** | Investor protections, expropriation, ISDS, investment liberalisation | "investor", "investment", "expropriation", "fair and equitable treatment", "ISDS", "performance requirements", "protection of investment" |
| **Dispute Settlement** | Procedures for resolving disputes between parties | "consultations", "panel", "arbitration", "compliance", "appeal", "request for establishment", timelines like "60 days", "120 days" |
| **Intellectual Property** | Copyright, patents, trademarks, geographical indications | "copyright", "patent", "trademark", "geographical indication", "enforcement of IP", "WIPO", "TRIPS" |
| **Customs Procedures** | Customs administration, clearance, release of goods, origin verification procedures | "customs administration", "release of goods", "advance ruling", "post-clearance audit", "risk management", "origin verification", "express shipments" |
| **Sanitary and Phytosanitary Measures** | Food safety, animal & plant health regulations | "SPS", "sanitary", "phytosanitary", "food safety", "animal health", "plant health", "pest", "IPPC", "OIE", "Codex" |
| **General Provisions / Definitions** | Preamble language, defined terms, scope clauses, relationship to other agreements, final clauses | "For the purposes of this Chapter", "means ...", "Definitions", "Objectives", "Scope", "Entry into force", "Amendment", "Depositary" |
| **Other** | Doesn't clearly fit any category above | (use sparingly — try a specific category first) |

---

## 5. Worked examples

### Example 1
> *"Each Party shall progressively reduce or eliminate its customs duties on originating goods in accordance with its Schedule in Annex I."*

**Decision:** The sentence is about **reducing customs duties**. Even though it says "originating goods," the action is tariff reduction, not the rule that defines origin.
**gold_category:** `Tariff Commitments`

### Example 2
> *"A good shall be considered as originating in a Party if it is wholly obtained or produced entirely in that Party, or if it satisfies the Product Specific Rules set out in Annex II."*

**Decision:** This defines *when a good qualifies as originating* — that's the textbook definition of Rules of Origin.
**gold_category:** `Rules of Origin`

### Example 3
> *"For the purposes of this Chapter: 'customs authority' means the competent authority that is responsible under the laws of a Party for the administration of customs laws..."*

**Decision:** This is a definitions clause.
**gold_category:** `General Provisions / Definitions`

### Example 4
> *"The Parties shall endeavour to ensure that any sanitary or phytosanitary measure is applied only to the extent necessary to protect human, animal or plant life or health..."*

**Decision:** Explicit SPS language.
**gold_category:** `Sanitary and Phytosanitary Measures`

### Example 5
> *"If consultations fail to resolve the dispute within 60 days of the request, the complaining Party may request the establishment of a panel."*

**Decision:** Procedural dispute mechanism.
**gold_category:** `Dispute Settlement`

### Example 6 — a tricky one
> *"Each Party shall publish its laws and regulations relating to customs matters on the internet in a timely manner."*

**Decision:** This is a transparency obligation specifically for customs. It could look like "Non-Tariff Measures" (transparency is an NTM theme), but the *subject matter* is customs administration.
**gold_category:** `Customs Procedures`
**notes (optional):** *transparency cue, but customs-specific*

---

## 6. Common pitfalls and how to resolve them

| Situation | What to do |
|---|---|
| Provision mentions both tariffs and rules of origin | Pick the one that describes the **main action**. "Reduce duties on originating goods" → Tariff Commitments. "Good is originating if RVC ≥ 40%" → Rules of Origin. |
| Provision is a definition that applies to a specific chapter (e.g. "'panel' means...") | Still `General Provisions / Definitions`. |
| Provision is just a header or numbering ("Article 3.5") with little text | Label as `Other` and add a note. |
| Provision is truncated / starts mid-sentence | Use your best judgment from what's there; if unreadable, label `Other` and note it. |
| You're torn between two categories | Pick the dominant one and put the alternative in `notes` (e.g. `notes = also fits Non-Tariff Measures`). |
| You genuinely can't tell | Label `Other` and note the difficulty — that's valuable data too. |

---

## 7. Quality checklist before you save

- [ ] All 50 rows have a `gold_category` filled (none left blank).
- [ ] Category names exactly match one of the 11 in the table above (spelling matters).
- [ ] If you opened `validation_checked.xlsx`, keep it as `.xlsx` and save in place.
- [ ] If you opened `validation_set.csv`, keep it as `.csv` and save in place.
- [ ] File path stays the same; `src.validation` prefers `validation_checked.xlsx` when both files exist.

---

## 8. What happens after you save

Run this command in the terminal (from the project root):

```bash
python3 -m src.validation --evaluate
```

That will produce `data/results/validation_report.json` with numbers like:

```
classified_llama_zero_shot_validation  n=50  acc=0.480  macroF1=0.431
classified_qwen_cot_validation         n=50  acc=0.460  macroF1=0.442
classified_qwen_zero_shot_validation   n=50  acc=0.380  macroF1=0.424
```

…and the corresponding figure `fig_validation_accuracy.png` will be generated automatically when you next run `python3 -m src.visualize`. Those numbers go straight into §4.7 of the report.

---

## 9. FAQ

**Q. Should I read the full provision or just the preview?**
A. The 250-character preview is usually enough. If it's truncated mid-sentence and ambiguous, you can open `data/raw/all_provisions.json` and search by `id` for the full text.

**Q. How strict should I be?**
A. Label what the provision is *primarily* about, not everything it *mentions*. A clause that mentions "tariffs" but is fundamentally a definitions clause is still `General Provisions / Definitions`.

**Q. Can I change my mind later?**
A. Yes — just re-save the CSV and re-run `python3 -m src.validation --evaluate`. Nothing is locked.

**Q. Does the order matter?**
A. No. You can skip around and label easy ones first.

**Q. Can I use AI to help me label?**
A. You can use AI to *explain* what a provision says in plain English, but the final category judgment should be yours — otherwise the validation is circular (we'd be measuring an AI against an AI).

---

*When you're done, the project has a real accuracy number it can defend. That's the single most important deliverable still missing from the submission.*

# Week 07 · Wednesday — NLP Foundations & Sentiment Analysis

**IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 1**

---

## 📌 Overview

This notebook tackles two real-world NLP challenges on the **ShopSense Indian e-commerce dataset**:

| Task | Description |
|------|-------------|
| **Q1** | NLP pipeline for 5 hard language patterns (Negation, Sarcasm, Code-mixing, Implicit, Comparative) |
| **Q2** | Sentiment modeling — review-level vs aspect-level, improvement strategies, ABSA extraction |

---

## 📂 Files

```
week07/wednesday/
├── week07_wednesday_nlp.ipynb    ← Main notebook (submit this)
└── README.md                     ← This file
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn gensim
```

### 2. Place dataset files

Copy both CSVs to the parent `week07/` folder (two levels up from this folder):

```
week07/
├── shopsense_reviews.csv
├── shopsense_customers.csv
└── wednesday/
    └── week07_wednesday_nlp.ipynb
```

### 3. Run the notebook

```bash
cd week07/wednesday
jupyter notebook week07_wednesday_nlp.ipynb
```

Run all cells top-to-bottom (`Kernel → Restart & Run All`).

---

## 🔬 What the Notebook Covers

### Q1 — NLP Pipeline for 5 Hard Patterns

| Pattern | Example | Fix Applied |
|---------|---------|-------------|
| **Negation** | "not bad at all" → Positive | NOT_ scope tagging |
| **Sarcasm** | "Wow great! Broke on day 1" → Negative | Multi-signal incongruence detection |
| **Code-mixing** | "bahut accha lekin delivery late" | Hinglish lexicon translation |
| **Implicit** | "Returned it within 2 hours" → Negative | Event/action rule patterns |
| **Comparative** | "Better than my old Samsung" → Positive | Comparative word polarity detection |

Each pattern includes:
- ✅ Preprocessing steps (clean → tokenize → transform)
- ✅ Baseline failure demonstration with real predictions
- ✅ Explanation of WHY the baseline fails
- ✅ Implementation of the fix
- ✅ Before vs after evaluation

### Q2 — Sentiment Modeling

- **(a)** Why aspect-level is harder: granularity, multi-label, cross-clause dependency
- **(b)** Roadmap from 71% → 80%+ F1 with technique-by-technique gain estimates
- **(c)** ABSA extraction on `"Amazing camera quality but the battery is atrocious and customer support was unhelpful."`
  - Output: `camera → Positive`, `battery → Negative`, `customer support → Negative`
- Word2Vec vs TF-IDF comparison
- Error analysis by language and error type

---

## 📊 Expected Outputs

| Model | F1 (Weighted) |
|-------|--------------|
| TF-IDF Unigram (baseline) | ~0.88 |
| TF-IDF Bigram | ~0.89 |
| TF-IDF + Signal Features | ~0.90 |
| Word2Vec Mean Pool | ~0.82 (small corpus) |
| **BERT (estimated)** | **~0.93–0.95** |

---

## 🏗️ Engineering Quality

| Indicator | Implementation |
|-----------|----------------|
| **Readable naming** | `apply_negation_scope()`, `compute_sarcasm_score()`, `extract_aspect_sentiment_pairs()` |
| **Modular (2+ functions/task)** | Every task split into preprocessing + evaluation functions |
| **Low hardcoding** | Constants defined at top: `TFIDF_MAX_FEATURES`, `RANDOM_SEED`, `NEGATION_SCOPE` |
| **Defensive handling** | `try/except` in `clean_html_and_noise()`, column validation in `load_and_validate_reviews()` |

---

## 🔗 References

- [ABSA Survey (Do et al., 2019)](https://arxiv.org/abs/1906.02525)
- [MuRIL: Multilingual Representations for Indian Languages](https://arxiv.org/abs/2103.10730)
- [Negation Scope in NLP (Morante & Blanco, 2012)](https://aclanthology.org/W12-3905/)

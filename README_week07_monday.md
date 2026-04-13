# Week 07 · Monday — TF-IDF from Scratch, Cosine Similarity & BM25

**IIT Gandhinagar · NLP Foundations · Cohort 1 Sudhanshu Biswas**

---

## What this notebook does

| Question | Task | Method |
|----------|------|--------|
| Q1a | TF-IDF matrix (10K reviews) | From scratch — sparse CSR matrix |
| Q1b | Top-5 review retrieval | Cosine similarity (dot product on L2-normalised rows) |
| Q1c | Comparison vs sklearn | `TfidfVectorizer` + average L2 difference |
| Q1d | Top word in Electronics | Per-column mean TF-IDF within category |
| Q2a | Manual TF-IDF arithmetic | Step-by-step: TF, IDF, TF-IDF for 'fabric' / Doc_42 |
| Q2b | IDF('the') vs IDF('embroidery') | Explains discriminative power of IDF |
| Q2c | Rebuttal to word-frequency | 3-sentence argument for TF-IDF |
| Bonus | BM25 ranking (k1=1.5, b=0.75) | From scratch + side-by-side comparison |

---

## Setup

```bash
pip install pandas numpy scipy scikit-learn matplotlib jupyter nbformat
```

---

## Run

```bash
# Navigate to this folder
cd week07/monday

# Execute the notebook
jupyter nbconvert --to notebook --execute week07_monday_tfidf.ipynb \
    --output week07_monday_tfidf_executed.ipynb

# Or open interactively
jupyter notebook week07_monday_tfidf.ipynb
```

---

## Files

| File | Description |
|------|-------------|
| `week07_monday_tfidf.ipynb` | Source notebook (clean, no outputs) |
| `week07_monday_tfidf_executed.ipynb` | Executed notebook with all outputs |
| `shopsense_reviews.csv` | Input dataset (10K reviews) |
| `shopsense_customers.csv` | Customer dataset |
| `tfidf_electronics_analysis.png` | Generated chart (top words + score distribution) |

---

## Expected Outputs

- **TF-IDF matrix shape**: ~(9000, 200–250) depending on tokeniser
- **Top-5 query results**: Electronics reviews matching "wireless earbuds battery life poor"
- **Avg L2 diff (scratch vs sklearn)**: ~0.0 (near-identical when vocab aligns)
- **Top Electronics word**: `quality` (high TF within category + moderate IDF)
- **IDF('the')**: ~2.09 (appears in ~33% of clothing docs, low signal)
- **IDF('embroidery')**: ~8.32 (not found — maximum IDF, maximum signal)
- **BM25 vs TF-IDF**: Different review_ids in top-5 (BM25 de-emphasises long verbose reviews)

---

## Engineering Checklist

- [x] Readable naming: `compute_tf()`, `compute_idf()`, `build_tfidf_matrix()`, `rank_documents()`
- [x] Modular: 8+ reusable functions, each with a docstring
- [x] Low hardcoding: all paths and hyperparameters defined as constants at top
- [x] Defensive handling: `try/except` on file I/O, column validation before processing

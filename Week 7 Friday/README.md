# ShopSense Sentiment Analysis Pipeline
**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar — Week 07 · Friday**

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 2. Generate the dataset (if not present)
python generate_data.py

# 3. Run the full pipeline
jupyter notebook shopsense_sentiment_pipeline.ipynb
# OR
python shopsense_pipeline.py
```

**Python version:** 3.10+  
**Required packages:** `pandas numpy matplotlib seaborn scikit-learn`

## File Structure
```
week-07/friday/
├── shopsense_sentiment_pipeline.ipynb   # Main notebook (submit this)
├── shopsense_pipeline.py                # Standalone script version
├── generate_data.py                     # Synthetic dataset generator
├── shopsense_reviews.csv                # Generated dataset (5,000 reviews)
├── README.md                            # This file
└── outputs/
    ├── 01_class_distribution.png
    ├── 02_cm_lr.png
    ├── 03_model_comparison.csv
    ├── 03_model_comparison.png
    ├── 04_cost_model.csv
    ├── 04_cost_comparison.png
    ├── 05_technical_brief.txt
    ├── 06_before_after.png
    └── 07_cost_of_failure.png
```

## What Each Step Delivers

| Step | Deliverable | Difficulty |
|---|---|---|
| 1. Data Analysis | Class distribution + accuracy paradox explanation | Easy |
| 2. Model Evaluation | LR metrics + business-language summary | Easy |
| 3. Model Comparison | LR vs NB across latency, unseen categories, Hinglish | Medium |
| 4. Cost Model | Daily/annual financial projection per model | Medium |
| 5. Technical Brief | 1-page recommendation + monitoring plan | Medium |
| 6. Failure Analysis | Root causes of 94% accuracy trap + before/after fix | Hard |
| 7. Cost of Failure | Faulty pipeline cost vs recommended + shared risks | Hard |

## Key Finding
A classifier can report **94.4% accuracy** while catching **0% of negative reviews**.  
This pipeline demonstrates why recall (negative class) is the correct metric for ShopSense,  
and how class imbalance + wrong metrics caused a silent production failure.

## AI Usage
AI tools were used to structure the pipeline. All prompts and critiques are documented  
in the final notebook cell.

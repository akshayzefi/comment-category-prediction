# Comment Category Prediction Challenge

## Overview
A machine learning pipeline that classifies online comments into 4 categories (0–3) using text content combined with post metadata (upvotes, downvotes, timestamps, demographic tags). The dataset is highly imbalanced — Class 0 makes up ~58% of samples while Class 3 makes up only ~2.8% — so models were evaluated using **macro F1-score** instead of accuracy, since accuracy alone can be misleading when one class dominates.

## Dataset
- 198,000 training samples, 14 features
- Features: `createddate`, `postid`, emoticon counts, `upvote`/`downvote`, `race`, `religion`, `gender`, `disability`, and the raw `comment` text
- Target distribution: Class 0 (57.7%), Class 2 (31.5%), Class 1 (8.0%), Class 3 (2.8%)
- Source: Kaggle competition dataset (train.csv, test.csv, Sample.csv) — not included in this repo due to size; download from the competition page

## Approach

### Feature Engineering
- **Text features:** TF-IDF vectorization using word-level and character-level n-grams
- **Datetime features:** extracted from `createddate` (hour, day of week, etc.)
- **Engagement features:** upvote/downvote ratios and net vote counts
- **Categorical encoding:** race, religion, gender, disability encoded via LabelEncoder
- All features combined into a single sparse matrix using `scipy.sparse.hstack`

### Models Trained
| Model | Validation Macro F1 |
|---|---|
| Logistic Regression | 0.7959 |
| SGD Classifier (ElasticNet) | 0.8092 |
| LightGBM | 0.8201 |
| LightGBM (tuned) | 0.8141 |
| **Ensemble (SGD + LightGBM + Logistic Regression, averaged probabilities)** | **0.8306** |

The final ensemble — averaging predicted probabilities from all three base models — outperformed every individual model, including the tuned LightGBM, and was used for the final submission.

### Why Macro F1?
With Class 0 dominating at ~58% of samples, a naive model that always predicts Class 0 would still score high on accuracy. Macro F1 forces the model to perform well across all four classes equally, which matters here given the real-world cost of misclassifying minority-class comments.

## Repository Structure
comment-category-prediction/
- notebooks/
  - comment_category_prediction.ipynb
- README.md
- requirements.txt
- .gitignore
## Setup

```bash
git clone https://github.com/akshayzefi/comment-category-prediction.git
cd comment-category-prediction
pip install -r requirements.txt
```

## Requirements
pandas
numpy
scikit-learn
lightgbm
scipy
seaborn
matplotlib


## Key Learnings
- Handling severe class imbalance requires choosing the right evaluation metric (macro F1) from the start, not just at the end
- Gradient-boosted trees (LightGBM) outperformed linear models on this mixed text + tabular feature set
- Simple probability-averaging ensembles can beat individually tuned models when base learners make different types of errors
- Hyperparameter tuning on a single model doesn't always beat a well-chosen ensemble of untuned models

## Author
Akshay Das - BS in Data Science & its Applications, IIT Madras

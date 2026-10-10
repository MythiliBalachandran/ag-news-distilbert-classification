# AG News Classification Using DistilBERT

## Overview

This project classifies news articles into four categories: World, Sports, Business, and Sci/Tech using a fine-tuned DistilBERT transformer model.

It also compares DistilBERT with a traditional machine learning baseline using TF-IDF and Logistic Regression.

## Tech Stack

- Python, PyTorch, Hugging Face Transformers
- Scikit-learn, Pandas, Matplotlib, Seaborn
- DistilBERT, TF-IDF, Logistic Regression

## Dataset

Used the AG News dataset, which contains four news categories. The experiments currently use a 10,000-example subset of the training data and an 800-example subset of the official test data.

## Results

**Validation performance**

| Model                        | Accuracy | Weighted F1 |
|------------------------------|----------|-------------|
| TF-IDF + Logistic Regression | 87.93%   | 87.86%      |
| DistilBERT                   | 91.33%   | 91.33%      |

**Preliminary test performance — DistilBERT**

- Test accuracy: 90.38%
- Weighted F1-score: 90.37%
- Test samples: 800

The validation results show a 3.40 percentage-point accuracy improvement with DistilBERT over the baseline.

## Error Analysis

Evaluated 1,500 validation examples and identified 130 misclassifications. Business and Sci/Tech were the most frequently confused categories.

The project generates a confusion matrix and a CSV containing misclassified examples.

## Run the Project

Install the dependencies:

```bash
pip install pandas numpy torch transformers accelerate scikit-learn matplotlib seaborn joblib
```

Run the scripts from the project directory:

```bash
python baseline_model.py
python error_analysis.py
python predict_news.py
```

The prediction script loads the saved DistilBERT model and predicts the category of a news headline and description.

To retrain DistilBERT, run `python distilbert_finetune.py`.

## Future Improvements

- Evaluate on the complete official test set.
- Compare both models on the same held-out test examples.
- Improve the prediction interface and investigate classification errors.

## References

- [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news)
- [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)

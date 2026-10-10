from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configuration
DATA_DIR = Path("data")
RANDOM_STATE = 42
TRAIN_LIMIT = 10000

# Load the same AG News dataset
train_df = pd.read_csv(DATA_DIR / "train.csv")

train_df["text"] = (
    train_df["Title"].fillna("").astype(str)
    + " "
    + train_df["Description"].fillna("").astype(str)
)
train_df["label"] = train_df["Class Index"].astype(int) - 1

# Use the same 10,000-example training subset and 15% validation split
subset_df, _ = train_test_split(
    train_df,
    train_size=TRAIN_LIMIT,
    stratify=train_df["label"],
    random_state=RANDOM_STATE,
)

X_train, X_val, y_train, y_val = train_test_split(
    subset_df["text"],
    subset_df["label"],
    test_size=0.15,
    stratify=subset_df["label"],
    random_state=RANDOM_STATE,
)

# TF-IDF + Logistic Regression baseline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )),
])

print("Training TF-IDF + Logistic Regression baseline...")
model.fit(X_train, y_train)

predictions = model.predict(X_val)

print("\nValidation results")
print("------------------")
print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
print(f"Weighted F1: {f1_score(y_val, predictions, average='weighted'):.4f}")

print("\nClassification report")
print(classification_report(
    y_val,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=["World", "Sports", "Business", "Sci/Tech"],
    digits=4,
))

# Save the fitted baseline for later predictions
import joblib

joblib.dump(model, "tfidf_logistic_baseline.joblib")
print("Baseline model saved to tfidf_logistic_baseline.joblib")

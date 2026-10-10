
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

DATA_DIR = Path("data")
MODEL_DIR = Path("final_agnews_model")
RANDOM_STATE = 42
TRAIN_LIMIT = 10000

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Recreate the same validation split
df = pd.read_csv(DATA_DIR / "train.csv")
df["text"] = (
    df["Title"].fillna("").astype(str)
    + " "
    + df["Description"].fillna("").astype(str)
)
df["label"] = df["Class Index"].astype(int) - 1

subset, _ = train_test_split(
    df,
    train_size=TRAIN_LIMIT,
    stratify=df["label"],
    random_state=RANDOM_STATE,
)

_, val_df = train_test_split(
    subset,
    test_size=0.15,
    stratify=subset["label"],
    random_state=RANDOM_STATE,
)

# Load your already-trained model; no retraining
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

predictions = []

texts = val_df["text"].tolist()
for start in range(0, len(texts), 16):
    batch = texts[start:start + 16]
    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
    predictions.extend(logits.argmax(dim=1).cpu().tolist())

results = val_df[["Title", "Description", "label"]].copy()
results["predicted_label"] = predictions
results["actual_category"] = results["label"].map(dict(enumerate(LABELS)))
results["predicted_category"] = results["predicted_label"].map(
    dict(enumerate(LABELS))
)

errors = results[results["label"] != results["predicted_label"]].copy()
errors.to_csv("misclassified_examples.csv", index=False)

print(f"Validation examples: {len(results)}")
print(f"Misclassified examples: {len(errors)}")
print(f"Error rate: {len(errors) / len(results):.2%}")

print("\nMost common actual -> predicted confusions:")
print(
    errors.groupby(["actual_category", "predicted_category"])
    .size()
    .sort_values(ascending=False)
    .head(10)
)

cm = confusion_matrix(
    results["label"],
    results["predicted_label"],
    labels=[0, 1, 2, 3],
)
display = ConfusionMatrixDisplay(cm, display_labels=LABELS)
display.plot(xticks_rotation=30)
plt.tight_layout()
plt.savefig("validation_confusion_matrix.png", dpi=150)
plt.show()

print("\nSaved misclassified_examples.csv and validation_confusion_matrix.png")

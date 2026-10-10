
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_DIR = Path("./final_agnews_model")
TEST_FILE = Path("./data/test.csv")
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Load official test dataset
df = pd.read_csv(TEST_FILE)
texts = (
    df["Title"].fillna("").astype(str)
    + " "
    + df["Description"].fillna("").astype(str)
).tolist()
true_labels = (df["Class Index"].astype(int) - 1).tolist()

# Load saved model (no training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

print(f"Device: {device}")
print(f"Evaluating {len(texts)} test articles...")

predictions = []
batch_size = 16

for start in range(0, len(texts), batch_size):
    batch = texts[start:start + batch_size]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)

    predictions.extend(preds.cpu().tolist())

    if (start // batch_size + 1) % 50 == 0:
        print(f"Processed {min(start + batch_size, len(texts))}/{len(texts)}")

print("\nFULL TEST RESULTS")
print("----------------------")
print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
print(f"Weighted F1: {f1_score(true_labels, predictions, average='weighted'):.4f}")

print("\nClassification Report:")
print(classification_report(
    true_labels,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=LABELS,
    digits=4
))

# Save predictions for future analysis
df["predicted_category"] = [LABELS[i] for i in predictions]
df.to_csv("full_test_predictions.csv", index=False)
print("Saved full_test_predictions.csv")

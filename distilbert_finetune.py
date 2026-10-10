
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
MODEL_NAME = "distilbert/distilbert-base-uncased"

DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = Path("./results")
MODEL_DIR = Path("./final_agnews_model")

# Start with a small CPU run to verify the pipeline.
# Set to None later to use the complete dataset.
TRAIN_LIMIT = 10000
TEST_LIMIT = 800
VALIDATION_SIZE = 0.15
EPOCHS = 2
MAX_LENGTH = 128

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ============================================================
# 3. Load and validate AG News data
# ============================================================

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

# Your downloaded CSV files have a header row.
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

required_columns = {"Class Index", "Title", "Description"}

for name, frame in [("training", train_df), ("test", test_df)]:
    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"{name} CSV is missing columns: {sorted(missing)}. "
            f"Found columns: {list(frame.columns)}"
        )

# Combine title and description into one text input.
for frame in (train_df, test_df):
    frame["text"] = (
        frame["Title"].fillna("").astype(str)
        + " "
        + frame["Description"].fillna("").astype(str)
    ).str.strip()

    # AG News labels are 1-4; PyTorch classification uses 0-3.
    frame["label"] = frame["Class Index"].astype(int) - 1

    if not frame["label"].between(0, 3).all():
        raise ValueError("Unexpected AG News labels. Expected classes 1-4.")

    if frame["text"].eq("").any():
        raise ValueError("Empty text examples found. Inspect the dataset.")

print(f"Full training rows: {len(train_df)}")
print(f"Official test rows: {len(test_df)}")
print("\nTraining class counts:")
print(train_df["label"].value_counts().sort_index())

# Optional smaller subset for the initial CPU smoke test.
if TRAIN_LIMIT is not None:
    train_df, _ = train_test_split(
        train_df,
        train_size=TRAIN_LIMIT,
        random_state=SEED,
        stratify=train_df["label"],
    )
    train_df = train_df.reset_index(drop=True)

if TEST_LIMIT is not None:
    test_df, _ = train_test_split(
        test_df,
        train_size=TEST_LIMIT,
        random_state=SEED,
        stratify=test_df["label"],
    )
    test_df = test_df.reset_index(drop=True)

print(f"\nTraining subset: {len(train_df)}")
print(f"Test subset: {len(test_df)}")


# ============================================================
# 4. Train-validation split
# ============================================================

train_part, val_part = train_test_split(
    train_df,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=train_df["label"],
)

train_part = train_part.reset_index(drop=True)
val_part = val_part.reset_index(drop=True)

print(f"Training examples: {len(train_part)}")
print(f"Validation examples: {len(val_part)}")


# ============================================================
# 5. Tokenization and PyTorch dataset
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = [int(label) for label in labels]

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[idx], dtype=torch.long
        )
        return item

    def __len__(self):
        return len(self.labels)


train_ds = NewsDataset(
    train_part["text"].tolist(),
    train_part["label"].tolist(),
    tokenizer,
    MAX_LENGTH,
)

val_ds = NewsDataset(
    val_part["text"].tolist(),
    val_part["label"].tolist(),
    tokenizer,
    MAX_LENGTH,
)

test_ds = NewsDataset(
    test_df["text"].tolist(),
    test_df["label"].tolist(),
    tokenizer,
    MAX_LENGTH,
)


# ============================================================
# 6. Load pretrained DistilBERT
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(CLASS_NAMES),
)

model.config.id2label = {
    i: name for i, name in enumerate(CLASS_NAMES)
}
model.config.label2id = {
    name: i for i, name in enumerate(CLASS_NAMES)
}


# ============================================================
# 7. Evaluation metrics
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


# ============================================================
# 8. Configure training
# ============================================================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_strategy="steps",
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
    dataloader_num_workers=0,
)


# ============================================================
# 9. Initialize Trainer and fine-tune
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2)
    ],
)

print("\nStarting DistilBERT fine-tuning...")
trainer.train()

print("\nValidation evaluation:")
validation_metrics = trainer.evaluate()
print(validation_metrics)


# ============================================================
# 10. Evaluate on the held-out test subset
# ============================================================

print("\n" + "=" * 50)
print("HELD-OUT TEST EVALUATION")
print("=" * 50)

test_output = trainer.predict(test_ds)

y_true = test_output.label_ids
y_pred = np.argmax(test_output.predictions, axis=-1)

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)

test_accuracy = accuracy_score(y_true, y_pred)
test_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test weighted F1: {test_f1:.4f}")


# ============================================================
# 11. Save the selected model and tokenizer
# ============================================================

trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

print(f"\nModel and tokenizer saved to: {MODEL_DIR.resolve()}")


# ============================================================
# 12. Plot and save the confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)

plt.title("AG News: DistilBERT Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("Confusion matrix saved to: confusion_matrix.png")
print("\nPipeline completed successfully.")


from transformers import pipeline

MODEL_PATH = "./final_agnews_model"

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

print("Loading trained DistilBERT model...")
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
)
print("Model loaded successfully!\n")

while True:
    title = input("Enter news headline (or type 'exit'): ").strip()

    if title.lower() == "exit":
        print("Goodbye!")
        break

    description = input("Enter news description: ").strip()
    text = f"{title} {description}"

    result = classifier(text, truncation=True, max_length=128)[0]

    label_id = int(result["label"].split("_")[-1]) \
        if result["label"].startswith("LABEL_") \
        else None

    if label_id is not None:
        category = LABELS[label_id]
    else:
        category = result["label"]

    print(f"\nPredicted category: {category}")
    print(f"Confidence: {result['score']:.2%}\n")

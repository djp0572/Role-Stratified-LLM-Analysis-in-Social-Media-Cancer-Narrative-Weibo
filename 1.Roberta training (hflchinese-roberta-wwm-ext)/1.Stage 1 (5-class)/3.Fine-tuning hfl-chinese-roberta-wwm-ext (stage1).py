# ======================
# 1. Install dependencies
# ======================
!pip install -q transformers datasets accelerate evaluate

# ======================
# 2. Imports
# ======================
from pathlib import Path

import torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ======================
# 3. Label mapping
# ======================
label2id = {
    "患者自述": 0,  # Patient self-narrative
    "亲属叙述": 1,  # Family member / caregiver narrative
    "朋友叙述": 2,  # Friend / other non-family social tie narrative
    "其他叙述": 3,  # Other (non-narrative / non-cancer-specific / noisy content)
    "无法判断": 4,  # Indeterminate / insufficient evidence
}
id2label = {v: k for k, v in label2id.items()}

# ======================
# 4. Model & tokenizer
# ======================
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# ======================
# 5. Dataset (local JSONL)
# ======================
DATA_DIR = Path("data")
train_path = DATA_DIR / "training_data_labeled.jsonl"
val_path = DATA_DIR / "testing_data_labeled.jsonl"

dataset = load_dataset(
    "json",
    data_files={"train": str(train_path), "validation": str(val_path)},
)

# ======================
# 6. Tokenization
# ======================
MAX_LEN = 256

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

tokenized = dataset.map(tokenize, batched=True)

# ======================
# 7. Training config
# ======================
OUTPUT_DIR = Path("outputs") / "roberta_cls_trained"
LOG_DIR = OUTPUT_DIR / "logs"

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=str(LOG_DIR),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

# ======================
# 8. Metrics
# ======================
accuracy = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=preds, references=labels)

# ======================
# 9. Trainer
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ======================
# 10. Train
# ======================
trainer.train()

# ======================
# 11. Save model
# ======================
final_model_dir = OUTPUT_DIR / "final_model"
final_model_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(final_model_dir))
tokenizer.save_pretrained(str(final_model_dir))
print(f"[OK] saved_model_dir={final_model_dir.as_posix()}")

# ======================
# 12. Quick inference test
# ======================
tokenizer = AutoTokenizer.from_pretrained(str(final_model_dir))
model = AutoModelForSequenceClassification.from_pretrained(str(final_model_dir))

test_text = "这是一条用于快速检查推理流程的测试文本。"
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    logits = model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=-1).item())

print(f"pred_id={pred_id} label={model.config.id2label[str(pred_id)]}")
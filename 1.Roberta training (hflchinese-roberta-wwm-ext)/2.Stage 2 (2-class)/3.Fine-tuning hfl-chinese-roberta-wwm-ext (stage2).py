!pip install -q transformers datasets accelerate evaluate

from pathlib import Path

import torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

label2id = {
    "Authentic": 0,       # Authentic cancer-related experiential narrative
    "Non-authentic": 1,   # Metaphor / joke / hypothetical / generic advice / news / promotion / noisy content
}
id2label = {v: k for k, v in label2id.items()}

model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

data_dir = Path("data")
dataset = load_dataset(
    "json",
    data_files={
        "train": str(data_dir / "training_data_authenticity.jsonl"),
        "validation": str(data_dir / "testing_data_authenticity.jsonl"),
    },
)

max_len = 256

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

tokenized = dataset.map(tokenize, batched=True)

output_dir = Path("outputs") / "roberta_auth_stage2"
training_args = TrainingArguments(
    output_dir=str(output_dir),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=str(output_dir / "logs"),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

accuracy = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.tensor(logits).argmax(dim=-1)
    return accuracy.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

final_model_dir = output_dir / "final_model"
final_model_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(final_model_dir))
tokenizer.save_pretrained(str(final_model_dir))
print(f"[OK] saved_model_dir={final_model_dir.as_posix()}")

tokenizer = AutoTokenizer.from_pretrained(str(final_model_dir))
model = AutoModelForSequenceClassification.from_pretrained(str(final_model_dir))

test_text = "这是一条用于快速检查推理流程的测试文本。"
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    pred_id = int(model(**inputs).logits.argmax(dim=-1).item())

print(f"pred_id={pred_id} label={model.config.id2label[str(pred_id)]}")
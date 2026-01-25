import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/roberta_cls_trained"
DATA_PATH = "data/raw/content.csv"
OUTPUT_PATH = "data/derived/content_preds.csv"

TEXT_COL = "content"
PRED_COL = "pred_label"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

df = pd.read_csv(DATA_PATH)

if os.path.exists(OUTPUT_PATH):
    df_out = pd.read_csv(OUTPUT_PATH)
    if PRED_COL not in df_out.columns:
        df_out[PRED_COL] = -1
    print("Resuming from existing output file.")
else:
    df_out = df.copy()
    df_out[PRED_COL] = -1

texts = df_out[TEXT_COL].astype(str).tolist()

batch_size = 128
save_every = 4000
total = len(df_out)

for i in range(0, total, batch_size):
    if (df_out.iloc[i:i + batch_size][PRED_COL] != -1).all():
        continue

    batch_texts = texts[i:i + batch_size]
    encodings = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        logits = model(**encodings).logits
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

    df_out.iloc[i:i + len(batch_preds), df_out.columns.get_loc(PRED_COL)] = batch_preds

    if ((i + len(batch_preds)) % save_every) < batch_size:
        df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"Saved progress: {i + len(batch_preds)}/{total}")

df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print("Done.")
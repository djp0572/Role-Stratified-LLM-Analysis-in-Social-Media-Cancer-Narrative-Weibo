import pandas as pd
from pathlib import Path

input_file = Path("data/raw/content_with_preds.csv")
output_file = Path("data/derived/samples/testing_sample_500.csv")

TEXT_COL = "content"
N = 20500
SEED = 42

if not input_file.exists():
    raise FileNotFoundError("Input file not found.")

output_file.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, usecols=[TEXT_COL], dtype={TEXT_COL: "string"})
df = df.dropna(subset=[TEXT_COL])
df[TEXT_COL] = df[TEXT_COL].str.strip()
df = df[df[TEXT_COL].str.len() > 0]

if len(df) < N:
    raise ValueError(f"Not enough rows after cleaning: need {N}, got {len(df)}")

sample_df = df.sample(n=N, random_state=SEED).reset_index(drop=False).rename(columns={"index": "id"})
sample_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"[OK] sampled_rows={len(sample_df)} seed={SEED}")
# Sentence Embeddings (BAAI/bge-large-zh-v1.5)

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ============================================================
MODEL_NAME = "BAAI/bge-large-zh-v1.5"

# Public version: use relative / generic paths to avoid leaking environment details
INPUT_PATH = "data/derived/clean_stage.csv"
OUTPUT_EMB = "data/derived/embeddings_bge_large_zh.npy"

TEXT_COL = "clean_content"  # output column from the cleaning step

# ============================================================
# Auto-select device (GPU/CPU)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] device={device}")

# Optional: set seeds for reproducibility (encoding is deterministic in practice, but this keeps the run stable)
torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed_all(42)

# ============================================================
# Load embedding model
# ============================================================
print(f"[INFO] loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device)
print("[INFO] model loaded")
print("[INFO] embedding_dim:", model.get_sentence_embedding_dimension())

# ============================================================
# Load texts
# ============================================================
df = pd.read_csv(INPUT_PATH)
assert TEXT_COL in df.columns, f"Missing column: {TEXT_COL}"

docs = df[TEXT_COL].astype(str).tolist()
print(f"[INFO] n_texts={len(docs)}")

# ============================================================
# Encode texts into sentence embeddings
# Notes for reviewers:
# - texts are embedded as complete units (no manual tokenization) to preserve narrative coherence
# - normalize_embeddings=True yields unit-length vectors, improving clustering stability
# ============================================================
BATCH_SIZE = 64  # adjust to 32/16 if out-of-memory

embeddings = model.encode(
    docs,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

# ============================================================
# Save embeddings
# ============================================================
np.save(OUTPUT_EMB, embeddings)
print("[INFO] embeddings saved")
print("[INFO] shape:", embeddings.shape)
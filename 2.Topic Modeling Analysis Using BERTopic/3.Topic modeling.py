# -*- coding: utf-8 -*-
"""
BERTopic topic modeling for Chinese narratives using precomputed bge-large-zh embeddings.

- Tokenization: jieba (ONLY for CountVectorizer / c-TF-IDF keyword extraction; NOT used for embeddings)
- Stopwords: resources/simple_stopwords.txt
"""

import os
import pandas as pd
import numpy as np
import jieba
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


# ==============================================================
# [1] Paths (public version: use generic relative paths)
#   - clean_stage1.csv should contain a column named "clean_content"
#   - embeddings .npy should be aligned row-wise with the CSV
# ==============================================================
BASE_DIR = "outputs/bertopic_run"

INPUT_PATH = os.path.join(BASE_DIR, "clean_stage1.csv")
EMB_PATH = os.path.join(BASE_DIR, "embeddings_bge_large_zh.npy")

OUTPUT_TOPIC = os.path.join(BASE_DIR, "bertopic_results.csv")
SUMMARY_PATH = os.path.join(BASE_DIR, "topic_summary.csv")
MODEL_SAVE = os.path.join(BASE_DIR, "bertopic_model")

STOPWORDS_PATH = "resources/simple_stopwords.txt"
TEXT_COL = "clean_content"


# ==============================================================
# [2] Load texts + embeddings
# ==============================================================
print("[INFO] loading texts and embeddings...")
df = pd.read_csv(INPUT_PATH)
embeddings = np.load(EMB_PATH)

if TEXT_COL not in df.columns:
    raise ValueError(f"Missing column: {TEXT_COL}")

docs_raw = df[TEXT_COL].astype(str).tolist()

if embeddings.shape[0] != len(df):
    raise ValueError(
        f"Row mismatch: n_docs={len(df)} but embeddings_rows={embeddings.shape[0]}. "
        "Make sure embeddings were generated from the same file in the same order."
    )

print(f"[INFO] n_docs={len(df)}")
print(f"[INFO] embeddings_shape={embeddings.shape}")


# ==============================================================
# [3] Load stopwords
# ==============================================================
print("[INFO] loading stopwords...")
with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    stopwords = [w.strip() for w in f if w.strip()]
print(f"[INFO] n_stopwords={len(stopwords)}")


# ==============================================================
# [4] jieba tokenizer for CountVectorizer (keywords only)
# Notes:
# - Embeddings are supplied externally (bge-large-zh); we do not use jieba for embeddings.
# - CountVectorizer here is used by BERTopic to produce interpretable topic words (c-TF-IDF).
# ==============================================================
def jieba_tokenizer(text: str):
    return [tok for tok in jieba.lcut(text) if tok.strip()]

vectorizer_model = CountVectorizer(
    tokenizer=jieba_tokenizer,
    stop_words=stopwords,
    ngram_range=(1, 2),
    min_df=5,
)


# ==============================================================
# [5] Build BERTopic model
# Notes for reviewers:
# - BERTopic performs UMAP-based dimensionality reduction and HDBSCAN clustering internally.
# - We provide embeddings to focus clustering on semantic similarity rather than bag-of-words.
# ==============================================================
topic_model = BERTopic(
    language="chinese",
    vectorizer_model=vectorizer_model,
    min_topic_size=40,   # typical tuning range: 30–50
    n_gram_range=(1, 2),
    verbose=True,
)

print("[INFO] fitting BERTopic...")
topics, probs = topic_model.fit_transform(docs_raw, embeddings)


# ==============================================================
# [6] Save results
# ==============================================================
os.makedirs(BASE_DIR, exist_ok=True)

df["topic"] = topics
df.to_csv(OUTPUT_TOPIC, index=False, encoding="utf-8-sig")

topic_info = topic_model.get_topic_info()
topic_info.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

topic_model.save(MODEL_SAVE)

print("[INFO] modeling finished")
print("[INFO] outputs saved")


# ==============================================================
# [7] Safe preview (NO raw text printing)
# Rationale:
# - To reduce re-identification risk, we do not print post text snippets.
# - Instead, we print top keywords and topic sizes.
# ==============================================================
print("\n[INFO] top-10 topics preview:")
for topic_id in topic_info["Topic"].head(10):
    if topic_id == -1:
        continue
    print(f"\n[TOPIC] {topic_id}")
    print(topic_model.get_topic(topic_id))
    print(f"[INFO] n_docs_in_topic={int((df['topic'] == topic_id).sum())}")


# ==============================================================
# [8] Save interactive visualization (HTML)
# ==============================================================
fig = topic_model.visualize_topics()
html_path = os.path.join(BASE_DIR, "topic_visualization.html")
fig.write_html(html_path)
print("[INFO] visualization saved")
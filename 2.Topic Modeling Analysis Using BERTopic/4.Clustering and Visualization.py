import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import umap
import seaborn as sns


# =========================
# Paths (public version)
# =========================
MODEL_PATH = "outputs/bertopic_run/bertopic_model"
RESULT_PATH = "outputs/bertopic_run/bertopic_results.csv"
SUMMARY_PATH = "outputs/bertopic_run/topic_summary.csv"
OUTPUT_DIR = "outputs/bertopic_run/topic_clustering"


# =========================
# Load model and outputs
# =========================
print("[INFO] loading BERTopic model...")
topic_model = BERTopic.load(MODEL_PATH)

print("[INFO] loading topic assignments and summary...")
df = pd.read_csv(RESULT_PATH)
summary = pd.read_csv(SUMMARY_PATH)

topic_embeddings = topic_model.topic_embeddings_
topic_ids = summary["Topic"].tolist()

# Prefer human-readable names if present; otherwise fall back to topic ids
if "Name" in summary.columns:
    topic_names = summary["Name"].astype(str).tolist()
else:
    topic_names = [f"Topic_{t}" for t in topic_ids]

print(f"[INFO] n_topics={len(topic_embeddings)}")


# ============================================================
# Step 1: Topic semantic distance matrix (cosine distance)
# ============================================================
print("[INFO] computing cosine distance matrix...")
dist_matrix = cosine_distances(topic_embeddings)


# ============================================================
# Step 2: Hierarchical clustering (Ward linkage)
# Notes:
# - We cluster topics (not documents) based on topic embeddings.
# - The number of clusters is controlled by TARGET_CLUSTER_NUM.
# ============================================================
TARGET_CLUSTER_NUM = 20

print(f"[INFO] hierarchical clustering (target_clusters={TARGET_CLUSTER_NUM})...")
Z = linkage(dist_matrix, method="ward")
cluster_ids = fcluster(Z, t=TARGET_CLUSTER_NUM, criterion="maxclust")

summary["Cluster"] = cluster_ids
os.makedirs(OUTPUT_DIR, exist_ok=True)
summary.to_csv(os.path.join(OUTPUT_DIR, "topic_clusters.csv"), index=False, encoding="utf-8-sig")
print("[INFO] saved: topic_clusters.csv")


# ============================================================
# Step 3: Dendrogram visualization
# Privacy note:
# - We only plot topic labels (names/ids), not raw document text.
# ============================================================
plt.figure(figsize=(15, 5))
dendrogram(Z, labels=topic_names, leaf_rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_topic_dendrogram.png"), dpi=300)
plt.close()
print("[INFO] saved: 1_topic_dendrogram.png")


# ============================================================
# Step 4: UMAP visualization in topic embedding space
# ============================================================
print("[INFO] running UMAP projection...")
u = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
umap_emb = u.fit_transform(topic_embeddings)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=umap_emb[:, 0],
    y=umap_emb[:, 1],
    hue=cluster_ids,
    palette="tab20",
    s=80,
    legend=False,
)

# Annotate points with topic index (safe: does not expose raw texts)
for i in range(len(topic_names)):
    plt.text(umap_emb[i, 0], umap_emb[i, 1], str(i), fontsize=7)

plt.title("Topic Clusters (UMAP)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_topic_umap.png"), dpi=300)
plt.close()
print("[INFO] saved: 2_topic_umap.png")


# ============================================================
# Step 5: Heatmap of topic semantic distances
# ============================================================
plt.figure(figsize=(12, 10))
sns.heatmap(dist_matrix, cmap="viridis")
plt.title("Topic Semantic Distance Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_topic_distance_heatmap.png"), dpi=300)
plt.close()
print("[INFO] saved: 3_topic_distance_heatmap.png")

print("[INFO] done")
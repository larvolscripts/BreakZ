import pandas as pd
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ================================
# 1. Input & Output
# ================================
IN_FILE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA.xlsx"
OUT_FILE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_clusters.xlsx"

print("📂 Loading:", IN_FILE)
df = pd.read_excel(IN_FILE, dtype=str)

if "Primary_MOA_all" not in df.columns:
    raise Exception("❌ ERROR: Primary_MOA_all column missing.")

# ================================
# 2. Clean MOA text
# ================================
def clean_moa(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9;, ]", " ", s)
    s = s.replace(";", ",")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

df["MOA_clean"] = df["Primary_MOA_all"].apply(clean_moa)

# ================================
# 3. TF-IDF vectorization
# ================================
tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
X = tfidf.fit_transform(df["MOA_clean"])

print("🔢 TF-IDF shape:", X.shape)

# ================================
# 4. Auto-select best K (clusters)
# ================================
print("🔍 Selecting best number of clusters...")
best_k = None
best_score = -1
scores = {}

for k in range(3, 12):  # try from 3 to 11 clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores[k] = score
    print(f"K={k}, silhouette={score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"\n🏆 Best K = {best_k} (silhouette={best_score:.4f})")

# ================================
# 5. Fit K-means with best K
# ================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X)

df["MOA_cluster"] = clusters

# ================================
# 6. Assign readable cluster names
# ================================
cluster_keywords = []

for c in range(best_k):
    idx = (df["MOA_cluster"] == c)
    text_sample = " ".join(df.loc[idx, "MOA_clean"].values)[:3000]

    # extract top TF-IDF keywords for the cluster
    if text_sample.strip() == "":
        cluster_keywords.append("Cluster_" + str(c))
    else:
        tfidf_cluster = TfidfVectorizer(stop_words="english").fit([text_sample])
        top_terms = sorted(
            tfidf_cluster.vocabulary_.items(),
            key=lambda x: x[1]
        )[:5]
        words = [w for w, _ in top_terms]
        cluster_keywords.append(", ".join(words))

df["MOA_cluster_label"] = df["MOA_cluster"].apply(lambda x: cluster_keywords[x])

# ================================
# 7. Save output
# ================================
df.to_excel(OUT_FILE, index=False)
print("\n✅ Saved clustered file to:", OUT_FILE)

print("\n📊 Cluster counts:")
print(df["MOA_cluster"].value_counts())

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
OUT_FILE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_ProductCluster.xlsx"

print("📂 Loading:", IN_FILE)
df = pd.read_excel(IN_FILE, dtype=str)

if "Product" not in df.columns:
    raise Exception("❌ ERROR: Product column missing.")

# ================================
# 2. Clean PRODUCT text
# ================================
def clean_prod(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9;,+/ ]", " ", s)
    s = s.replace("/", " ")
    s = s.replace("+", " ")
    s = s.replace(";", ",")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

df["Product_clean"] = df["Product"].apply(clean_prod)

# ================================
# 3. TF-IDF vectorization
# ================================
tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
X = tfidf.fit_transform(df["Product_clean"])

print("🔢 TF-IDF shape:", X.shape)

# ================================
# 4. Auto-select best K
# ================================
print("🔍 Selecting best number of clusters...")
best_k = None
best_score = -1

for k in range(3, 20):  # Products more diverse than MOA
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)

    print(f"K={k}, silhouette={score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"\n🏆 Best K = {best_k} (silhouette={best_score:.4f})")

# ================================
# 5. Fit K-means with best K
# ================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
df["Product_cluster"] = kmeans.fit_predict(X)

# ================================
# 6. Auto-generate human labels
# ================================
cluster_keywords = []

for c in range(best_k):
    idx = df["Product_cluster"] == c
    text_sample = " ".join(df.loc[idx, "Product_clean"].values)[:3000]

    if text_sample.strip() == "":
        cluster_keywords.append(f"Cluster_{c}")
    else:
        tfidf_cluster = TfidfVectorizer(stop_words="english").fit([text_sample])
        top_terms = sorted(tfidf_cluster.vocabulary_.items(), key=lambda x: x[1])[:5]
        words = [w for w, _ in top_terms]
        cluster_keywords.append(", ".join(words))

df["Product_cluster_label"] = df["Product_cluster"].apply(lambda c: cluster_keywords[c])

# ================================
# 7. Save
# ================================
df.to_excel(OUT_FILE, index=False)
print("\n✅ Saved PRODUCT clusters to:", OUT_FILE)

print("\n📊 Product Cluster counts:")
print(df["Product_cluster"].value_counts())

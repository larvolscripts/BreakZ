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
OUT_FILE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters.xlsx"

print("📂 Loading:", IN_FILE)
df = pd.read_excel(IN_FILE, dtype=str)

if "Precise_Area_Name" not in df.columns:
    raise Exception("❌ ERROR: Precise_Area_Name column missing.")

# ================================
# 2. Clean precise disease text
# ================================
def clean_precise(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9, ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

df["Precise_clean"] = df["Precise_Area_Name"].apply(clean_precise)

# ================================
# 3. TF-IDF
# ================================
tfidf = TfidfVectorizer(min_df=2)
X = tfidf.fit_transform(df["Precise_clean"])

print("🔢 TF-IDF shape:", X.shape)

# ================================
# 4. Auto-select best K
# ================================
print("🔍 Selecting best number of clusters...")
best_k = None
best_score = -1

for k in range(3, 15):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)

    print(f"K={k}, silhouette={score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"\n🏆 Best K = {best_k} (silhouette={best_score:.4f})")

# ================================
# 5. Fit
# ================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
df["Precise_cluster"] = kmeans.fit_predict(X)

# ================================
# 6. Auto Labels
# ================================
cluster_keywords = []

for c in range(best_k):
    idx = df["Precise_cluster"] == c
    text_sample = " ".join(df.loc[idx, "Precise_clean"].values)[:3000]

    if text_sample.strip() == "":
        cluster_keywords.append(f"Cluster_{c}")
    else:
        tfidf_cluster = TfidfVectorizer(stop_words="english").fit([text_sample])
        top_terms = sorted(tfidf_cluster.vocabulary_.items(), key=lambda x: x[1])[:5]
        words = [w for w, _ in top_terms]
        cluster_keywords.append(", ".join(words))

df["Precise_cluster_label"] = df["Precise_cluster"].apply(lambda c: cluster_keywords[c])

# ================================
# 7. Save
# ================================
df.to_excel(OUT_FILE, index=False)
print("\n✅ Saved PRECISE AREA clusters to:", OUT_FILE)

print("\n📊 Precise Cluster counts:")
print(df["Precise_cluster"].value_counts())

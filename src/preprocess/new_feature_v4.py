# feature_engineering_medianpfs.py
# Run: python feature_engineering_medianpfs.py

# add after precise_moa__

import os
import re
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# CONFIG — edit these paths
# -----------------------------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_V4.xlsx"
# If you have another file that contains min/max ORR / DoR info to merge, set path below.
# If not available, set to None.
OPTIONAL_MINMAX_PATH = None  # e.g. r"C:\path\to\orr_dor_minmax.xlsx"
OUTPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_FEATURES_READY_V4.1.xlsx"

# KMeans cluster range for TF-IDF clustering
KMIN, KMAX = 3, 12

# -----------------------------
# Small utilities
# -----------------------------
def safe_read_excel(path):
    print(f"Loading: {path}")
    df = pd.read_excel(path, dtype=str)
    print(f"Loaded shape: {df.shape}")
    return df

def to_float_safe(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def clean_text_for_tfidf(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s\-,/()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_combination_text(s):
    if pd.isna(s) or str(s).strip()=="":
        return 0
    text = str(s)
    # common delimiters / plus signs or a comma with multiple drug names
    if ("," in text) or ("+" in text) or ("/" in text):
        return 1
    return 0

def count_products_text(s):
    if pd.isna(s) or str(s).strip()=="":
        return 0
    # heuristics: split by comma, slash, plus, semicolon
    tokens = re.split(r"[,+;/]", str(s))
    tokens = [t.strip() for t in tokens if t.strip()!=""]
    return max(1, len(tokens))  # if non-empty, at least 1

# -----------------------------
# 1) Load data
# -----------------------------
df = safe_read_excel(INPUT_PATH)

# Make copy to avoid destructive operations
df_out = df.copy()

# Ensure numeric conversions for key numeric columns (if exist)
numeric_cols_guess = [
    "Arm N", "Objective Response Rate N", "Objective Response Rate Percentage",
    "Duration of Response Median", "Median PFS",
    "Objective_Response_Rate_Min", "Objective_Response_Rate_Max",
    "Duration_of_Response_Min", "Duration_of_Response_Max"
]

for col in numeric_cols_guess:
    if col in df_out.columns:
        df_out[col] = df_out[col].apply(to_float_safe)

# -----------------------------
# 2) Optional: merge min/max ORR & DoR if provided
#    (the external file should contain Trial ID / Arm ID or unique key to merge on)
# -----------------------------
if OPTIONAL_MINMAX_PATH:
    df_minmax = safe_read_excel(OPTIONAL_MINMAX_PATH)
    # Try to common-join on Trial ID + Arm ID if available
    join_cols = []
    for c in ["Trial ID", "Arm ID"]:
        if c in df_out.columns and c in df_minmax.columns:
            join_cols.append(c)
    if join_cols:
        print(f"Merging min/max file on columns: {join_cols}")
        df_out = df_out.merge(df_minmax, on=join_cols, how="left", suffixes=("", "_minmax"))
    else:
        print("No matching join columns for optional min/max file. Skipping merge.")

# -----------------------------
# 3) Basic engineered numeric features
# -----------------------------
# ArmSize_norm_ORR = ORR_N / Arm_N
if "Objective Response Rate N" in df_out.columns and "Arm N" in df_out.columns:
    df_out["ArmSize_norm_ORR"] = df_out.apply(
        lambda r: (to_float_safe(r.get("Objective Response Rate N")) / to_float_safe(r.get("Arm N")))
        if (to_float_safe(r.get("Arm N")) and not math.isnan(to_float_safe(r.get("Arm N")))) else np.nan,
        axis=1
    )
else:
    df_out["ArmSize_norm_ORR"] = np.nan

# ExposureAdj_ORR = (ORR% / 100) / Arm_N
if "Objective Response Rate Percentage" in df_out.columns and "Arm N" in df_out.columns:
    df_out["ExposureAdj_ORR"] = df_out.apply(
        lambda r: ((to_float_safe(r.get("Objective Response Rate Percentage"))/100.0) / to_float_safe(r.get("Arm N")))
        if (to_float_safe(r.get("Arm N")) and not math.isnan(to_float_safe(r.get("Arm N")))) else np.nan,
        axis=1
    )
else:
    df_out["ExposureAdj_ORR"] = np.nan

# ORR_x_DoR (percent*duration already exists as Response_Percentage_Duration maybe)
if "Response_Percentage_Duration" in df_out.columns:
    df_out["ORR_x_DoR"] = df_out["Response_Percentage_Duration"].apply(to_float_safe)
else:
    # fallback compute from % * median duration
    if "Objective Response Rate Percentage" in df_out.columns and "Duration of Response Median" in df_out.columns:
        df_out["ORR_x_DoR"] = df_out.apply(
            lambda r: (to_float_safe(r.get("Objective Response Rate Percentage"))/100.0) * to_float_safe(r.get("Duration of Response Median")),
            axis=1
        )
    else:
        df_out["ORR_x_DoR"] = np.nan

# log of ORR_x_DoR (use log1p)
df_out["log_ORR_x_DoR"] = df_out["ORR_x_DoR"].apply(lambda x: np.log1p(x) if pd.notna(x) else np.nan)

# Also create log_ORR_pct and log_DOR for modeling convenience
if "Objective Response Rate Percentage" in df_out.columns:
    df_out["log_ORR_pct"] = df_out["Objective Response Rate Percentage"].apply(lambda x: np.log1p(to_float_safe(x)) if pd.notna(x) else np.nan)
else:
    df_out["log_ORR_pct"] = np.nan

if "Duration of Response Median" in df_out.columns:
    df_out["log_DOR"] = df_out["Duration of Response Median"].apply(lambda x: np.log1p(to_float_safe(x)) if pd.notna(x) else np.nan)
else:
    df_out["log_DOR"] = np.nan

# -----------------------------
# 4) Text-derived features: Product_count, Is_Combination
# -----------------------------
if "Product" in df_out.columns:
    df_out["Is_Combination"] = df_out["Product"].apply(is_combination_text)
    df_out["Product_count"] = df_out["Product"].apply(count_products_text)
else:
    df_out["Is_Combination"] = 0
    df_out["Product_count"] = 0

# -----------------------------
# 5) Add publication era (5-year bins)
# -----------------------------
if "Publication Year" in df_out.columns:
    def era_fn(y):
        try:
            yy = int(float(y))
            return (yy // 5) * 5
        except Exception:
            return np.nan
    df_out["Publication_Era"] = df_out["Publication Year"].apply(era_fn)
else:
    df_out["Publication_Era"] = np.nan

# -----------------------------
# 6) Source normalization (simple)
# -----------------------------
if "Source Name" in df_out.columns:
    df_out["Source_Normalized"] = df_out["Source Name"].astype(str).str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.strip()
    # optional: collapse long names: take first two words
    df_out["Source_Normalized"] = df_out["Source_Normalized"].apply(lambda s: " ".join(s.split()[:2]) if isinstance(s, str) and s.strip()!="" else "")
else:
    df_out["Source_Normalized"] = ""

# -----------------------------
# 7) MOA_Class and MOA_Subtype (rule-based)
#    Uses 'Primary_MOA_all' column
# -----------------------------
def moa_class_and_subtype(moa_text):
    s = "" if pd.isna(moa_text) else str(moa_text).lower()
    # Subtype extraction: look for core tokens
    subtype = None
    for token in ["pd1", "pd-1", "pd 1", "pd-l1", "pd1 inhib", "pd-l1 inhib", "ctla4", "ctla", "egfr", "alk", "her2", "braf", "mek", "brc", "parp", "vegf", "c-met", "met", "trop-2", "cd20", "cd19", "btk"]:
        if token in s:
            subtype = token.replace(" ", "").replace("-", "")
            break
    # fallback some common names
    if subtype is None:
        if "pd1" in s or "pd-1" in s or "pd-l1" in s or "pdl1" in s:
            subtype = "pd1_pdl1"
        elif "ctla" in s:
            subtype = "ctla4"
        elif "egfr" in s:
            subtype = "egfr"
        elif "her2" in s:
            subtype = "her2"
        elif "alk" in s:
            subtype = "alk"
    # class mapping
    moa_class = "Other"
    if subtype is not None:
        if any(k in subtype for k in ["pd", "pdl1", "pd1", "ctla"]):
            moa_class = "Immunotherapy"
        elif any(k in subtype for k in ["egfr", "alk", "her2", "braf", "mek", "met", "parp", "vegf", "brc"]):
            moa_class = "Targeted"
        elif any(k in subtype for k in ["trop", "adc", "antibody", "mab", "cd"]):
            moa_class = "Antibody/ADC"
    else:
        # try to detect chemotherapy words
        if any(w in s for w in ["carboplatin", "cisplatin", "paclitaxel", "docetaxel", "etoposide", "fluorouracil", "irinotecan", "gemcitabine", "oxaliplatin"]):
            moa_class = "Chemotherapy"
    return moa_class, (subtype if subtype is not None else "")

df_out["MOA_Class"], df_out["MOA_Subtype"] = zip(*df_out.get("Primary_MOA_all", pd.Series([""]*len(df_out))).apply(moa_class_and_subtype))

# -----------------------------
# 8) TF-IDF + KMeans clustering for Product and Precise_Area_Name
#    We'll automatically choose k with silhouette between KMIN..KMAX
# -----------------------------
def create_text_clusters(series, col_prefix, kmin=3, kmax=12, min_df=2):
    docs = series.fillna("").astype(str).map(clean_text_for_tfidf).tolist()

    if all(d.strip()=="" for d in docs):
        return pd.Series([0]*len(series)), ["empty_cluster"], None, None

    tfidf = TfidfVectorizer(min_df=min_df, ngram_range=(1,2), stop_words="english")
    X = tfidf.fit_transform(docs)
    print(f"[{col_prefix}] TF-IDF shape: {X.shape}")

    if X.shape[0] < 10 or X.shape[1] < 2:
        return pd.Series([0]*len(series)), ["small_cluster"], tfidf, None

    # choose best k
    best_k = kmin
    best_score = -1
    scores = {}
    for k in range(kmin, min(kmax+1, X.shape[0])):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labs = km.fit_predict(X)
            if len(set(labs)) <= 1:
                continue
            score = silhouette_score(X, labs)
            scores[k] = score
            if score > best_score:
                best_score, best_k = score, k
        except:
            continue

    print(f"[{col_prefix}] silhouette scores: {scores}, selected K={best_k}")

    # final model
    kmodel = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = kmodel.fit_predict(X)
    labels_series = pd.Series(labels, index=series.index)

    # Add readable names
    terms = np.array(tfidf.get_feature_names_out())
    label_names = []
    for c in range(best_k):
        mask = (labels_series == c).values  # FIX HERE
        Xc = X[mask]                       # FIX HERE
        if Xc.shape[0] == 0:
            label_names.append(f"cluster_{c}")
            continue
        mean_tfidf = np.asarray(Xc.mean(axis=0)).ravel()
        top = mean_tfidf.argsort()[::-1][:5]
        words = [terms[i] for i in top if mean_tfidf[i] > 0]
        label_names.append(", ".join(words) if words else f"cluster_{c}")

    return labels_series, label_names, tfidf, kmodel


# Product clustering
prod_labels, prod_label_names, prod_tfidf, prod_kmodel = create_text_clusters(df_out.get("Product", pd.Series([""]*len(df_out))), "Product")
df_out["Product_cluster"] = prod_labels
# add readable product cluster name
df_out["Product_cluster_label"] = df_out["Product_cluster"].apply(lambda x: prod_label_names[int(x)] if prod_label_names and int(x) < len(prod_label_names) else f"prod_cluster_{x}")

# Precise area clustering
precise_labels, precise_label_names, precise_tfidf, precise_kmodel = create_text_clusters(df_out.get("Precise_Area_Name", pd.Series([""]*len(df_out))), "Precise_Area")
df_out["Precise_cluster"] = precise_labels
df_out["Precise_cluster_label"] = df_out["Precise_cluster"].apply(lambda x: precise_label_names[int(x)] if precise_label_names and int(x) < len(precise_label_names) else f"precise_cluster_{x}")

# MOA cluster label fallback: if MOA_cluster_labeL exists keep it; else add MOA_cluster (if present)
if "MOA_cluster_labeL" not in df_out.columns:
    # if MOA_cluster exists numeric, create string labels
    if "MOA_cluster" in df_out.columns:
        df_out["MOA_cluster_label"] = df_out["MOA_cluster"].apply(lambda x: f"moa_cluster_{x}")
    else:
        df_out["MOA_cluster_label"] = df_out.get("Primary_MOA_all", "").fillna("").apply(lambda s: str(s)[:40])

# -----------------------------
# 9) Add objective min/max if found from merged file or columns
# -----------------------------
# If the optional file provided columns named differently, try to locate candidate columns
def try_copy_col(src_cols):
    for c in src_cols:
        if c in df_out.columns:
            return df_out[c]
    return None

# Standardize column names for min/max if exist
orr_min_candidates = ["Objective_Response_Rate_Min", "ORR_Min", "Objective Response Rate Min", "ORR_min"]
orr_max_candidates = ["Objective_Response_Rate_Max", "ORR_Max", "Objective Response Rate Max", "ORR_max"]
dor_min_candidates = ["Duration_of_Response_Min", "DOR_Min", "Duration of Response Min", "DOR_min"]
dor_max_candidates = ["Duration_of_Response_Max", "DOR_Max", "Duration of Response Max", "DOR_max"]

orr_min_series = try_copy_col(orr_min_candidates)
orr_max_series = try_copy_col(orr_max_candidates)
dor_min_series = try_copy_col(dor_min_candidates)
dor_max_series = try_copy_col(dor_max_candidates)

if orr_min_series is not None:
    df_out["Objective_Response_Rate_Min"] = orr_min_series.apply(to_float_safe)
if orr_max_series is not None:
    df_out["Objective_Response_Rate_Max"] = orr_max_series.apply(to_float_safe)
if dor_min_series is not None:
    df_out["Duration_of_Response_Min"] = dor_min_series.apply(to_float_safe)
if dor_max_series is not None:
    df_out["Duration_of_Response_Max"] = dor_max_series.apply(to_float_safe)

# -----------------------------
# 10) Final touch & diagnostics
# -----------------------------
# Show counts for new columns
new_cols = [
    "ArmSize_norm_ORR","ExposureAdj_ORR","ORR_x_DoR","log_ORR_x_DoR",
    "log_ORR_pct","log_DOR","Is_Combination","Product_count",
    "MOA_Class","MOA_Subtype","Product_cluster","Product_cluster_label",
    "Precise_cluster","Precise_cluster_label","Publication_Era","Source_Normalized"
]
print("\nNew columns created (example values):\n")
print(df_out[new_cols].head(5))

# Cluster distribution diagnostics
if "Product_cluster" in df_out.columns:
    print("\nProduct_cluster counts:")
    print(df_out["Product_cluster"].value_counts().sort_index())
if "Precise_cluster" in df_out.columns:
    print("\nPrecise_cluster counts:")
    print(df_out["Precise_cluster"].value_counts().sort_index())

# -----------------------------
# 11) Save final file (non-destructive)
# -----------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_out.to_excel(OUTPUT_PATH, index=False)
print("\n✅ Saved feature-engineered file to:", OUTPUT_PATH)

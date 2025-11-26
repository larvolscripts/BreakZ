# improve_pfs_features_and_train.py
import os
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import shap

# ---------------- CONFIG - update paths ----------------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned.xlsx"   # <- cleaned input
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\improved_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Median PFS"
GROUP_COL = "Trial ID"   # used for group split to prevent leakage

RANDOM_STATE = 42

# ---------------- Helpers ----------------
def bucket_disease(text):
    if pd.isna(text): return "Other"
    s = str(text).lower()
    if any(x in s for x in ["non-small", "nsclc", "lung"]): return "Lung"
    if any(x in s for x in ["breast"]): return "Breast"
    if any(x in s for x in ["colorec", "crc", "colon", "rectal"]): return "Colorectal"
    if any(x in s for x in ["melanoma"]): return "Melanoma"
    if any(x in s for x in ["renal", "kidney", "rcc"]): return "RCC"
    if any(x in s for x in ["ovarian"]): return "Ovarian"
    if any(x in s for x in ["prostate"]): return "Prostate"
    if any(x in s for x in ["leukemia","lymphoma","myeloma","hemato","hematologic"]): return "Hematologic"
    if any(x in s for x in ["head and neck","hnscc"]): return "HeadNeck"
    return "Other"

def map_product_category(prod):
    # coarse categories derived from product string. Add rules for your corpus.
    if pd.isna(prod): return "Unknown"
    s = str(prod).lower()
    if any(x in s for x in ["pembrolizu","nivolumab","nivolum","atezoliz","durvalumab","avelumab","pembro","ipilimumab","tremelimumab","nivolu"]):
        return "Immunotherapy"
    if any(x in s for x in ["-tinib","nib","tki","erlot","gefit","osimert","sunitinib","crizotinib","axitinib","cabozantinib"]):
        return "TKI"
    if any(x in s for x in ["chemo","carbo","paclitaxel","docetaxel","cisplatin","oxaliplatin","fluorouracil","folfox"]):
        return "Chemotherapy"
    if any(x in s for x in ["trastuzumab","pertuzumab","t-dm1","trastuzumab deruxtecan","adc","-mab","mab"]):
        return "Antibody/ADC"
    if any(x in s for x in ["-ib","inib"]):  # generic targeted suffix
        return "Targeted"
    if any(x in s for x in ["combination", "+", "/"]):
        return "Combination"
    return "Other"

def is_combination(prod):
    if pd.isna(prod): return 0
    s = str(prod).lower()
    # treat '+' or '/' or comma-separated multiple drugs as combination
    if ("+" in s) or ("/" in s) or ("," in s) or (" and " in s):
        return 1
    # if there are multiple drug names heuristically (space-separated tokens > 3 and comma present)
    tokens = re.split(r"[\s,;/]+", s)
    uniq = [t for t in tokens if len(t)>2]
    return 1 if len(uniq) > 1 else 0

def safe_log1p(x):
    # return np.nan for invalid inputs, but log1p for >=0
    try:
        return np.log1p(float(x))
    except Exception:
        return np.nan

# ---------------- Load ----------------
print("Loading:", INPUT_PATH)
df = pd.read_excel(INPUT_PATH)
print("rows,cols:", df.shape)

# keep a copy
orig_df = df.copy()

# ---------------- Basic cleaning ----------------
# ensure numeric columns exist and coerced
for col in ["Objective Response Rate Percentage", "Duration of Response Median", "Median PFS"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# drop rows without target
df = df.dropna(subset=[TARGET])
print("After dropping missing target:", df.shape)

# ---------------- Feature engineering ----------------
# 1) disease buckets
if "Type" in df.columns:
    df["Disease_Bucket"] = df["Type"].fillna("").apply(bucket_disease)
else:
    df["Disease_Bucket"] = "Other"

# 2) product category & combination flag
df["Product_clean"] = df["Product"].astype(str).fillna("").str.replace(r"\s+", " ", regex=True)
df["Product_Category"] = df["Product_clean"].apply(map_product_category)
df["Is_Combination"] = df["Product_clean"].apply(is_combination).astype(int)

# 3) log transforms
df["log_ORR_pct"] = df["Objective Response Rate Percentage"].apply(safe_log1p)
df["log_DOR"] = df["Duration of Response Median"].apply(safe_log1p)
# ORR*DOR raw and log
df["ORR_x_DOR"] = df["Objective Response Rate Percentage"] * df["Duration of Response Median"]
df["log_ORR_x_DOR"] = df["ORR_x_DOR"].apply(safe_log1p)

# 4) Interaction terms (disease x log_ORR, disease x log_DOR)
# We'll one-hot encode Disease_Bucket and multiply by log features
disease_dummies = pd.get_dummies(df["Disease_Bucket"], prefix="Dis")
for col in disease_dummies.columns:
    df[col + "_x_logORR"] = disease_dummies[col] * df["log_ORR_pct"]
    df[col + "_x_logDOR"] = disease_dummies[col] * df["log_DOR"]

# 5) Keep a set of engineered numeric features
engineered_numeric = [
    "log_ORR_pct", "log_DOR", "ORR_x_DOR", "log_ORR_x_DOR", "Is_Combination"
]

# ---------------- Prepare feature matrix ----------------
# choose columns: numeric engineered + existing useful numerics + categorized features for LightGBM
numerics = ["Arm N", "Objective Response Rate N", "Objective Response Rate Percentage", "Duration of Response Median"]
numerics = [c for c in numerics if c in df.columns]

feature_cols = []
feature_cols += numerics
feature_cols += engineered_numeric

# categorical columns to include (coarse)
cats = []
if "Product_Category" in df.columns:
    cats.append("Product_Category")
if "Disease_Bucket" in df.columns:
    cats.append("Disease_Bucket")
if "Precise_Area_Name" in df.columns:
    cats.append("Precise_Area_Name")
if "Source_Name" in df.columns:
    cats.append("Source_Name")
# ensure exists
cats = [c for c in cats if c in df.columns]

feature_cols += cats

# add the disease×log interactions (numeric)
inter_cols = [c for c in df.columns if c.endswith("_x_logORR") or c.endswith("_x_logDOR")]
feature_cols += inter_cols

# drop duplicates
feature_cols = list(dict.fromkeys(feature_cols))

print("Using features count:", len(feature_cols))
print("Sample features:", feature_cols[:20])

# ---------------- Group split ----------------
if GROUP_COL in df.columns:
    groups = df[GROUP_COL].astype(str).fillna("nan")
    print("Using group split on:", GROUP_COL)
else:
    groups = np.arange(len(df))
    print("Group col not present; using random split fallback.")

X = df[feature_cols].copy()
y = df[TARGET].astype(float)

# LightGBM likes categorical dtype
for c in cats:
    if c in X.columns:
        X[c] = X[c].astype("category")

# Fill numeric nans with median
for col in X.select_dtypes(include=[np.number]).columns:
    med = X[col].median()
    X[col] = X[col].fillna(med)

for col in X.select_dtypes(include=["category", "object"]).columns:
    if X[col].dtype.name == "category":
        # Add empty string "" as a valid category
        if "" not in X[col].cat.categories:
            X[col] = X[col].cat.add_categories([""])
        X[col] = X[col].fillna("")
    else:
        # Object dtype: direct fill
        X[col] = X[col].fillna("")


gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, valid_idx = next(gss.split(X, y, groups=groups))
X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

# Drop trial/arm ids in feature sets if present
for idc in ["Trial ID","Arm ID"]:
    if idc in X_train.columns:
        X_train = X_train.drop(columns=idc)
        X_valid = X_valid.drop(columns=idc)

print("Train shape:", X_train.shape, "Valid shape:", X_valid.shape)

# ---------------- Train LightGBM ----------------
model = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=40,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=RANDOM_STATE,
    objective="regression"
)

print("Training LightGBM...")
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="l1",
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)]
)

# ---------------- Evaluate ----------------
y_pred = model.predict(X_valid)
r2 = r2_score(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print("\nRESULTS:")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# save model & columns
joblib.dump(model, os.path.join(OUTPUT_DIR, "lgbm_model_improved.pkl"))
pd.Series(X_train.columns).to_csv(os.path.join(OUTPUT_DIR, "feature_columns.csv"), index=False)

# ---------------- SHAP importance ----------------
print("Computing SHAP (approx)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_abs_mean = np.abs(shap_values).mean(axis=0)
fi = pd.DataFrame({
    "Feature": X_valid.columns,
    "SHAP_mean_abs": shap_abs_mean
}).sort_values("SHAP_mean_abs", ascending=False)

fi.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"), index=False)

plt.figure(figsize=(8,6))
sns.barplot(data=fi.head(20), y="Feature", x="SHAP_mean_abs")
plt.title("Top SHAP features (improved features)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar_improved.png"), bbox_inches="tight")
plt.close()

# ---------------- Save train/valid with preds for inspection ----------------
valid_out = X_valid.copy()
valid_out["MedianPFS_true"] = y_valid.values
valid_out["MedianPFS_pred"] = y_pred
valid_out.to_csv(os.path.join(OUTPUT_DIR, "valid_with_preds.csv"), index=False)

print("All outputs saved to", OUTPUT_DIR)

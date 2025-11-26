# train_final_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import joblib
import os
import re

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_product_features.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\final_lgbm_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Median PFS"
GROUP_COL = "Trial ID"

DROP_COLS = ["Trial ID", "Arm ID", "Date of Publication", "Dosage", "Product"] 
# Product removed because we now use engineered product features

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================
print(f"Loading {INPUT_PATH}")
df = pd.read_excel(INPUT_PATH)

df = df[df[GROUP_COL].notna()].copy()
df[GROUP_COL] = df[GROUP_COL].astype(str)

y = df[TARGET]


# ============================================================
# CATEGORICAL + NUMERIC CLEANING
# ============================================================
all_features = [c for c in df.columns if c != TARGET]

numeric_cols = df[all_features].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in all_features if c not in numeric_cols]

# remove cols we do not want as features
final_features = [c for c in all_features if c not in DROP_COLS]

# Clean string columns
for c in cat_cols:
    df[c] = df[c].astype(str).fillna("")


X = df[final_features]


# ============================================================
# GROUP SHUFFLE SPLIT
# ============================================================
groups = df[GROUP_COL]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, valid_idx = next(gss.split(X, y, groups))

X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

print("Train:", X_train.shape, "Valid:", X_valid.shape)


# LightGBM cannot ingest object strings → convert to category
for c in cat_cols:
    if c in X_train.columns:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype("category")


# ============================================================
# TRAIN FINAL LGBM MODEL
# (use your best Optuna parameters if available)
# ============================================================

model = lgb.LGBMRegressor(
    boosting_type="gbdt",
    objective="regression",
    metric="rmse",
    random_state=RANDOM_SEED,

    # Your best optuna parameters:
    learning_rate=0.015527206376765645,
    n_estimators=1811,
    num_leaves=64,
    max_depth=10,
    min_child_samples=70,
    subsample=0.7125461002991458,
    colsample_bytree=0.8120212437478006,
    reg_alpha=0.8289344618192059,
    reg_lambda=0.9065594659696742,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    # verbose=100
)


# ============================================================
# EVALUATION
# ============================================================
pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, pred)
rmse = mean_squared_error(y_valid, pred, squared=False)
r2 = r2_score(y_valid, pred)

print("\n=======================")
print(" FINAL MODEL RESULTS")
print("=======================")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))


# ============================================================
# SAVE MODEL + FEATURES
# ============================================================
model.booster_.save_model(os.path.join(OUTPUT_DIR, "model.txt"))
joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))
pd.Series(final_features).to_csv(os.path.join(OUTPUT_DIR, "final_features.csv"), index=False)

print(f"\nSaved final model to {OUTPUT_DIR}")

# shap_lgbm.py

import pandas as pd
import numpy as np
import shap
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

# ======================================================
# CONFIG
# ======================================================
DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_product_features.xlsx"
MODEL_PATH = r"C:\LARVOL_WORK\Median_PFS\outputs\final_lgbm_model\model.pkl"
FEATURE_PATH = r"C:\LARVOL_WORK\Median_PFS\outputs\final_lgbm_model\final_features.csv"

TARGET = "Median PFS"
GROUP_COL = "Trial ID"

# ======================================================
# LOAD MODEL + DATA
# ======================================================
print("Loading model + dataset...")
model = joblib.load(MODEL_PATH)

df = pd.read_excel(DATA_PATH)
features = pd.read_csv(FEATURE_PATH).iloc[:,0].tolist()

df = df[df[GROUP_COL].notna()].copy()
df[GROUP_COL] = df[GROUP_COL].astype(str)

X = df[features]
y = df[TARGET]

# LightGBM categorical fix
for c in X.select_dtypes(include="object").columns:
    X[c] = X[c].astype("category")

print(f"Dataset shape for SHAP: {X.shape}")

# ======================================================
# CREATE SHAP EXPLAINER
# ======================================================
print("Building SHAP explainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# ======================================================
# SHAP SUMMARY PLOT
# ======================================================
print("Generating SHAP summary plot...")
plt.figure(figsize=(12,8))
shap.summary_plot(shap_values.values, X, show=False)
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# ======================================================
# BAR PLOT (GLOBAL FEATURE IMPORTANCE)
# ======================================================
plt.figure(figsize=(12,8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig("shap_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n==========================")
print(" SHAP GENERATED SUCCESSFULLY")
print("==========================")
print("✔ shap_summary.png")
print("✔ shap_bar.png")
print("Check these two images to understand feature impact.")

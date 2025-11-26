"""
Master tuning script (Optuna) for:
 - XGBoost (XGBRegressor)
 - CatBoost (CatBoostRegressor)
 - Build StackingRegressor using best XGB + best CAT + user LGBM (meta)
 
Notes:
 - Group-aware split by 'Trial ID' to avoid leakage.
 - Converts categorical columns to safe formats for each library.
 - Saves best params + fitted models to OUTPUT_DIR.
"""

import os
import json
import joblib
from time import time
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import OrdinalEncoder
import re
import optuna

# Models
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor

# ----------------------------
# CONFIG - change these paths if needed
# ----------------------------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\tuning_master"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If you already have tuned LGBM params, paste them here to use as meta-learner
USER_TUNED_LGBM_PARAMS = {
    # Example (replace with your actual tuned params). If None, will use defaults.
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "random_state": 42,
    "objective": "regression"
}

TARGET = "Median PFS"
GROUP_COL = "Trial ID"
DROP_COLS = ["Trial ID", "Arm ID", "Date of Publication", "Dosage"]  # dropped after grouping

RANDOM_SEED = 42
N_TRIALS = 40   # change as you like (longer → better search)
TIMEOUT = None  # optuna timeout in seconds, or None

# ----------------------------
# UTILITIES
# ----------------------------
def safe_median_fill(df, cols):
    for c in cols:
        if c in df.columns:
            med = float(df[c].median(skipna=True))
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)
    return df

def prepare_data(df, target_col=TARGET, group_col=GROUP_COL, drop_cols=DROP_COLS):
    # Remove rows without target
    df = df[df[target_col].notna()].copy()

    # Ensure group_col exists
    if group_col not in df.columns:
        df[group_col] = np.arange(len(df)).astype(str)

    # Keep a copy
    df = df.reset_index(drop=True)

    # Extract X/y
    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])

    # Group split
    groups = df[group_col].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, valid_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_valid = X.iloc[valid_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_valid = y.iloc[valid_idx].copy()

    # Drop ID columns AFTER split
    for c in drop_cols:
        if c in X_train.columns:
            X_train = X_train.drop(columns=c)
        if c in X_valid.columns:
            X_valid = X_valid.drop(columns=c)

    return X_train.reset_index(drop=True), X_valid.reset_index(drop=True), y_train.reset_index(drop=True), y_valid.reset_index(drop=True)

def identify_categoricals(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # also treat low-cardinal numerics as categorical? optional
    return cat_cols

# ----------------------------
# LOAD DATA
# ----------------------------
print("Loading data:", INPUT_PATH)
df = pd.read_excel(INPUT_PATH)
print("Raw shape:", df.shape)

# Make sure Trial ID is string and strip
if "Trial ID" in df.columns:
    df["Trial ID"] = df["Trial ID"].astype(str).str.strip()

# ---------- prepare data split ----------
X_train, X_valid, y_train, y_valid = prepare_data(df, TARGET, GROUP_COL, DROP_COLS)
print("Train shape:", X_train.shape, "Valid shape:", X_valid.shape)

# ----------------------------
# SIMPLE PREPROCESS FOR BOTH: numeric median fill + categorical fill
# ----------------------------
# numeric columns
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
# but some numeric-like columns might be object typed; convert candidates:
for col in X_train.columns:
    if col not in numeric_cols:
        # if looks numeric, coerce
        sample = X_train[col].dropna().astype(str).head(20).tolist()
        if len(sample) and all(re.match(r"^-?\d+(\.\d+)?$", s) for s in sample):
            numeric_cols.append(col)

# fill numeric NaNs with median on both train/valid
X_train = safe_median_fill(X_train, numeric_cols)
X_valid = safe_median_fill(X_valid, numeric_cols)

# identify categorical columns (explicit)
cat_cols = identify_categoricals(X_train)
print("Categorical columns detected:", cat_cols)

# Convert categories to strings for CatBoost-safe behaviour
for c in cat_cols:
    X_train[c] = X_train[c].astype(str).fillna("").replace("nan", "")
    X_valid[c] = X_valid[c].astype(str).fillna("").replace("nan", "")

# Also ensure no leftover NaNs anywhere
X_train = X_train.fillna("")
X_valid = X_valid.fillna("")

# Save a copy of the cleaned feature names
CLEANED_COLS = X_train.columns.tolist()

# ----------------------------
# Prepare versions for each model:
# - CatBoost: needs categorical columns as strings + indices
# - XGBoost: numeric features only -> encode categoricals to integer codes
# ----------------------------
# 1) For CatBoost -> keep textual categories as-is (strings), compute cat_feature indices
cat_feature_indices = [i for i, col in enumerate(CLEANED_COLS) if col in cat_cols]
print("CatBoost cat_feature indices:", cat_feature_indices)

X_train_cat = X_train.copy()
X_valid_cat = X_valid.copy()

# 2) For XGBoost -> encode categorical columns to integer codes (Ordinal)
X_train_xgb = X_train.copy()
X_valid_xgb = X_valid.copy()
if len(cat_cols) > 0:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_xgb[cat_cols] = enc.fit_transform(X_train_xgb[cat_cols])
    X_valid_xgb[cat_cols] = enc.transform(X_valid_xgb[cat_cols])

# Ensure numeric dtype for XGB
X_train_xgb = X_train_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)
X_valid_xgb = X_valid_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)

# ----------------------------
# Save preprocessed copies (optional)
pd.DataFrame(X_train_cat).to_excel(os.path.join(OUTPUT_DIR, "X_train_cat_preprocessed.xlsx"), index=False)
pd.DataFrame(X_valid_cat).to_excel(os.path.join(OUTPUT_DIR, "X_valid_cat_preprocessed.xlsx"), index=False)

# ----------------------------
# OBJECTIVES (Optuna)
# ----------------------------
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "random_state": RANDOM_SEED,
        "verbosity": 0,
        "tree_method": "hist",  # fast
    }

    model = XGBRegressor(**params)
    model.fit(X_train_xgb, y_train)
    pred = model.predict(X_valid_xgb)
    r2 = r2_score(y_valid, pred)
    return r2

def objective_cat(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 900),
"depth": trial.suggest_int("depth", 4, 8),
"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
"l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 5.0, log=True),
"border_count": trial.suggest_int("border_count", 32, 128),

    }

    # CatBoost wants a Pool; we pass it directly (strings are OK)
    train_pool = Pool(X_train_cat, y_train, cat_features=cat_feature_indices)
    val_pool = Pool(X_valid_cat, y_valid, cat_features=cat_feature_indices)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=False)
    pred = model.predict(X_valid_cat)
    r2 = r2_score(y_valid, pred)
    return r2

# ----------------------------
# RUN OPTUNA for XGB & CAT
# ----------------------------
study_xgb = optuna.create_study(direction="maximize", study_name="XGB_tuning")
print("Start tuning XGBoost... (trials:", N_TRIALS, ")")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, timeout=TIMEOUT)
best_xgb = study_xgb.best_params
best_xgb["n_estimators"] = int(best_xgb["n_estimators"])
print("Best XGB params:")
pprint(best_xgb)

with open(os.path.join(OUTPUT_DIR, "best_xgb_params.json"), "w") as f:
    json.dump(best_xgb, f, indent=2)

study_cat = optuna.create_study(direction="maximize", study_name="CAT_tuning")
print("Start tuning CatBoost... (trials:", N_TRIALS, ")")
study_cat.optimize(objective_cat, n_trials=N_TRIALS, timeout=TIMEOUT)
best_cat = study_cat.best_params
print("Best CatBoost params:")
pprint(best_cat)

# Cat params names slightly different - map to CatBoost constructor args
cat_params_mapped = {
    "iterations": int(best_cat["iterations"]),
    "depth": int(best_cat["depth"]),
    "learning_rate": float(best_cat["learning_rate"]),
    "l2_leaf_reg": float(best_cat["l2_leaf_reg"]),
    "border_count": int(best_cat["border_count"]),
    "random_seed": RANDOM_SEED,
    "loss_function": "RMSE",
    "verbose": False
}
with open(os.path.join(OUTPUT_DIR, "best_cat_params.json"), "w") as f:
    json.dump(cat_params_mapped, f, indent=2)

# ----------------------------
# Fit final models with best params
# ----------------------------
print("\nTraining final XGB with best params...")
final_xgb = XGBRegressor(**best_xgb, tree_method="hist", verbosity=0, random_state=RANDOM_SEED)
final_xgb.fit(X_train_xgb, y_train)
pred_xgb = final_xgb.predict(X_valid_xgb)
print("XGB R2:", r2_score(y_valid, pred_xgb))

print("\nTraining final CatBoost with best params...")
final_cat = CatBoostRegressor(**cat_params_mapped)
train_pool = Pool(X_train_cat, y_train, cat_features=cat_feature_indices)
val_pool = Pool(X_valid_cat, y_valid, cat_features=cat_feature_indices)
final_cat.fit(train_pool, eval_set=val_pool, use_best_model=False)
pred_cat = final_cat.predict(X_valid_cat)
print("CatBoost R2:", r2_score(y_valid, pred_cat))

# Save the models
joblib.dump(final_xgb, os.path.join(OUTPUT_DIR, "final_xgb.joblib"))
final_cat.save_model(os.path.join(OUTPUT_DIR, "final_cat.cbm"))
print("Saved final XGB & CatBoost models.")

# ----------------------------
# Stacking: use the two fitted base models + user LGBM as final estimator
# We need scikit-learn compatible estimators: wrap CatBoost with silent sklearn API
# ----------------------------
print("\nBuilding stacking regressor...")

# Prepare X versions for stacking inputs: XGB expects numeric-coded cat version, Cat expects string version.
# We'll train stacking with numeric features (encoded) but include a CatBoost wrapper that accepts encoded numeric as well.
# Simpler approach: retrain fresh instances of XGB & CAT inside the stacking pipeline using best params.

estimators = [
    ("xgb", XGBRegressor(**best_xgb, tree_method="hist", verbosity=0, random_state=RANDOM_SEED)),
    ("cat", CatBoostRegressor(**cat_params_mapped)),
]

# Meta-learner: use user-tuned LGBM params if provided
meta_params = USER_TUNED_LGBM_PARAMS.copy()
meta = LGBMRegressor(**meta_params)

# For sklearn Stacking, we must pass X as numeric matrix. We'll use X_train_xgb / X_valid_xgb (categoricals encoded)
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=meta,
    passthrough=False,
    n_jobs=-1
)

# Fit stack: For CatBoost inside stacking, scikit-learn wrapper expects numeric/categorical handling;
# to avoid complexities we convert all categorical columns to ordinal integers for the stacking fit (already in X_train_xgb).
stack.fit(X_train_xgb, y_train)

pred_stack = stack.predict(X_valid_xgb)
r2_stack = r2_score(y_valid, pred_stack)
mae_stack = mean_absolute_error(y_valid, pred_stack)
rmse_stack = mean_squared_error(y_valid, pred_stack)

print("\nSTACKING RESULTS:")
print("R2    :", r2_stack)
print("MAE   :", mae_stack)
print("RMSE  :", rmse_stack)

# Save stacking model
joblib.dump(stack, os.path.join(OUTPUT_DIR, "stacking_model.joblib"))

# Save evaluation summary
summary = {
    "XGB_val_r2": float(r2_score(y_valid, pred_xgb)),
    "CAT_val_r2": float(r2_score(y_valid, pred_cat)),
    "STACK_val_r2": float(r2_stack),
    "STACK_mae": float(mae_stack),
    "STACK_rmse": float(rmse_stack),
    "best_xgb_params": best_xgb,
    "best_cat_params": cat_params_mapped,
    "meta_lgbm_params": meta_params
}

with open(os.path.join(OUTPUT_DIR, "tuning_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll done. Outputs saved to:", OUTPUT_DIR)
pprint(summary)

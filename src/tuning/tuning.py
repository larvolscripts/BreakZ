import optuna
import pandas as pd
import numpy as np
import re
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb

# ======================================================
# Load data
# ======================================================
# df = pd.read_excel(r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned.xlsx")
# df=pd.read_excel(r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters.xlsx")---0.74_STABLED CORRCT

df=pd.read_excel(r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_FEATURES_READY_V4.1.xlsx")

# # tuning for kmeans
# df = pd.read_excel(r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_clusters.xlsx")
# 📊 Evaluation Results:
# MAE  : 1.6720
# MSE  : 5.9933
# RMSE : 2.4481
# R²   : 0.7410

# TUNING FOR RULE PMOA
# df=pd.read_excel(r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_categories.xlsx")

# Remove outliers
df = df[df["Median PFS"] <= 40]

target = "Median PFS"
drop_cols = ["Trial ID", "Arm ID", "Date of Publication", "Dosage"]

# Features BEFORE dropping
features = [c for c in df.columns if c != target]
X = df[features].copy()
y = df[target]

groups = df["Trial ID"]  # for no leakage

# ======================================================
# Drop unwanted columns (same as your training script)
# ======================================================
for c in drop_cols:
    if c in X.columns:
        X = X.drop(columns=c)

# ======================================================
# Encode categoricals (same logic as your code)
# ======================================================
def clean_feature_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X[cat_cols] = X[cat_cols].fillna("Unknown")
for col in cat_cols:
    X[col] = X[col].astype("category")

X.columns = [clean_feature_name(c) for c in X.columns]

# ======================================================
# Optuna Objective Function
# ======================================================
def objective(trial):

    params = {
        "objective": "regression",
        "metric": "l2",
        "random_state": 42,

        # Search space:
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 400, 2000),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }

    # Group split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(gss.split(X, y, groups=groups))

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # ===============================
# SAFE categorical detection
# ===============================
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# 🔥 ensure no dropped/missing cols passed to LGBM
    cat_cols = [c for c in cat_cols if c in X_train.columns]

# convert to category
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")


    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
        categorical_feature=cat_cols
    )

    preds = model.predict(X_valid)
    return r2_score(y_valid, preds)

# ======================================================
# Run Optuna
# ======================================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("\n🎯 Best Parameters:")
print(study.best_params)
print(f"🏆 Best R²: {study.best_value:.4f}")

study.trials_dataframe().to_csv(
    r"C:\LARVOL_WORK\Median_PFS\outputs\optuna_lgbm_trials.csv",
    index=False
)

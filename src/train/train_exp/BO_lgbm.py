import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
INPUT = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
TARGET = "Median PFS"
GROUP_COL = "Trial ID"

DROP_COLS = ["Arm ID", "Date of Publication"]   # DO NOT drop Trial ID here

# ======================================================
# LOAD
# ======================================================
df = pd.read_excel(INPUT)
df = df[df[TARGET].notna()].copy()

# drop invalid rows
df = df[df[GROUP_COL].notna()].copy()

# ======================================================
# DEFINE FEATURES
# ======================================================
features = [c for c in df.columns if c != TARGET and c not in DROP_COLS]

X = df[features].copy()
y = df[TARGET].values
groups = df[GROUP_COL].astype(str).values

# identify categorical features
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# FIX: convert all categorical columns → category
for col in cat_features:
    X[col] = X[col].fillna("").astype("category")

# re-check: MUST show no object dtypes
print("\nCHECK DTYPES BEFORE SPLIT:")
print(X.dtypes)

# ======================================================
# GROUP SPLIT
# ======================================================
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, valid_idx = next(gss.split(X, y, groups))

X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
y_train, y_valid = y[train_idx], y[valid_idx]

# ======================================================
# Convert categorical again after slicing
# ======================================================
for col in cat_features:
    X_train[col] = X_train[col].astype("category")
    X_valid[col] = X_valid[col].astype("category")

# get numeric + categorical index-based features
cat_index = [X_train.columns.get_loc(c) for c in cat_features]

print("\nCategorical index positions:", cat_index)

# ======================================================
# OPTUNA OBJECTIVE
# ======================================================
def objective(trial):

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("lr", 0.005, 0.15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 120),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000)
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        # early_stopping_rounds=150,
        categorical_feature=cat_index,
        # verbose=False
    )

    pred = model.predict(X_valid)
    return r2_score(y_valid, pred)


# ======================================================
# RUN OPTUNA
# ======================================================
print("\n===== RUNNING BAYESIAN OPTIMIZATION =====")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("\nBest R2:", study.best_value)
print("Best Params:", study.best_params)


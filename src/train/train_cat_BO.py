import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
TARGET = "Median PFS"
GROUP_COL = "Trial ID"
DROP_COLS = ["Trial ID", "Arm ID", "Date of Publication", "Dosage"]

df = pd.read_excel(INPUT_PATH)
df = df[df[GROUP_COL].notna()].copy()
df[GROUP_COL] = df[GROUP_COL].astype(str)

y = df[TARGET]
X = df.drop(columns=[TARGET])

for c in DROP_COLS:
    if c in X: X = X.drop(columns=c)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
X[cat_cols] = X[cat_cols].astype(str).fillna("")

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, valid_idx = next(gss.split(X, y, df[GROUP_COL]))

X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)

# ============================
# FAST Bayesian Optimization
# ============================
def objective(trial):

    params = {
        "loss_function": "RMSE",
        "iterations": trial.suggest_int("iterations", 400, 900),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 5.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "random_seed": 42,
        "task_type": "CPU",
        "verbose": False
    }

    model = CatBoostRegressor(**params)

    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=50,
        verbose=False
    )

    preds = model.predict(X_valid)
    return r2_score(y_valid, preds)


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5)
)

study.optimize(objective, n_trials=25)

print("\n===== FAST OPTUNA DONE =====")
print("Best R2:", study.best_value)
print("Best Params:", study.best_params)

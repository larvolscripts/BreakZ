import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import re

# -----------------------------------------------------------
# 📌 Load input dataset
# -----------------------------------------------------------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
df = pd.read_excel(INPUT_PATH)

TARGET = "Median PFS"
drop_cols = ["Trial ID", "Arm ID", "Date of Publication", "Dosage"]

# -----------------------------------------------------------
# 📌 Prepare data
# -----------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

groups = df["Trial ID"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(gss.split(X, y, groups))

X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

# Drop ID fields AFTER split
for c in drop_cols:
    if c in X_train:
        X_train = X_train.drop(columns=c)
        X_valid = X_valid.drop(columns=c)

# -----------------------------------------------------------
# 📌 Encode categoricals using OrdinalEncoder (safe for all models)
# -----------------------------------------------------------
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_valid[cat_cols] = encoder.transform(X_valid[cat_cols])

# -----------------------------------------------------------
# 📌 Base Model 1: CatBoost (use tuned params)
# -----------------------------------------------------------
cat_model = CatBoostRegressor(
    iterations=606,
    depth=8,
    learning_rate=0.01066,
    l2_leaf_reg=3.88,
    border_count=74,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

cat_model.fit(X_train, y_train)
cat_train_pred = cat_model.predict(X_train)
cat_valid_pred = cat_model.predict(X_valid)

# -----------------------------------------------------------
# 📌 Base Model 2: XGBoost (use tuned params)
# -----------------------------------------------------------
xgb_model = XGBRegressor(
    n_estimators=617,
    learning_rate=0.00505,
    max_depth=9,
    min_child_weight=13,
    subsample=0.6605,
    colsample_bytree=0.5659,
    reg_alpha=0.0246,
    reg_lambda=0.0018,
    tree_method="hist",
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_train_pred = xgb_model.predict(X_train)
xgb_valid_pred = xgb_model.predict(X_valid)

# -----------------------------------------------------------
# 📌 Create meta-features
# -----------------------------------------------------------
stack_train = pd.DataFrame({
    "cat_pred": cat_train_pred,
    "xgb_pred": xgb_train_pred
})

stack_valid = pd.DataFrame({
    "cat_pred": cat_valid_pred,
    "xgb_pred": xgb_valid_pred
})

# -----------------------------------------------------------
# 📌 Meta Model: LightGBM (your tuned params)
# -----------------------------------------------------------
meta_model = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=42
)

meta_model.fit(stack_train, y_train)
meta_pred = meta_model.predict(stack_valid)

# -----------------------------------------------------------
# 📌 Final Evaluation
# -----------------------------------------------------------
print("\n==============================")
print(" STACKING MODEL PERFORMANCE ")
print("==============================")
print("MAE  :", mean_absolute_error(y_valid, meta_pred))
print("RMSE :", mean_squared_error(y_valid, meta_pred, squared=False))
print("R²   :", r2_score(y_valid, meta_pred))

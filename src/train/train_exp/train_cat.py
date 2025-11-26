import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re

# =====================================
# Load Data
# =====================================
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"

df = pd.read_excel(INPUT_PATH)

TARGET = "Median PFS"
drop_cols = ["Trial ID", "Arm ID", "Date of Publication", "Dosage"]

# =====================================
# Features & Target
# =====================================
X = df.drop(columns=[TARGET])
y = df[TARGET]

groups = df["Trial ID"]

# =====================================
# Group-based split (no leakage)
# =====================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(gss.split(X, y, groups))

X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

# Drop unwanted columns AFTER splitting
for c in drop_cols:
    if c in X_train:
        X_train = X_train.drop(columns=c)
        X_valid = X_valid.drop(columns=c)

# =====================================
# Categorical Columns (fixed)
# =====================================
cat_cols = [
    "Type", "Product", "Source Name", "Primary_MOA_all",
    "MOA_cluster_labeL", "Precise_Area_Name",
    "Precise_cluster_label", "Product_Category"
]

# Convert categoricals → safe strings (CatBoost requirement)
for col in cat_cols:
    X_train[col] = X_train[col].astype(str).replace("nan", "").replace("NaN", "")
    X_valid[col] = X_valid[col].astype(str).replace("nan", "").replace("NaN", "")

# Also replace true NaN with empty string
X_train[cat_cols] = X_train[cat_cols].fillna("")
X_valid[cat_cols] = X_valid[cat_cols].fillna("")

# =====================================
# Train CatBoost
# =====================================
model = CatBoostRegressor(
    iterations= 410,
 depth= 8,
 learning_rate= 0.09528587217040241,
 l2_leaf_reg=0.22948683681130552,
 border_count= 49,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_cols,
    use_best_model=True
)

# =====================================
# Evaluate
# =====================================
pred = model.predict(X_valid)

print("\n📊 CatBoost Results")
print("MAE  :", mean_absolute_error(y_valid, pred))
print("RMSE :", mean_squared_error(y_valid, pred)**0.5)
print("R²   :", r2_score(y_valid, pred))

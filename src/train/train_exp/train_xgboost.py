import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import re

INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"

df = pd.read_excel(INPUT_PATH)

TARGET = "Median PFS"
drop_cols = ["Trial ID", "Arm ID", "Date of Publication", "Dosage"]

X = df.drop(columns=[TARGET])
y = df[TARGET]

groups = df["Trial ID"]

# ======================================================
# Group-based split
# ======================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(gss.split(X, y, groups))

X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx].copy()

# ======================================================
# Drop ID columns AFTER split
# ======================================================
for c in drop_cols:
    if c in X_train:
        X_train = X_train.drop(columns=c)
        X_valid = X_valid.drop(columns=c)

# ======================================================
# Identify categorical columns
# ======================================================
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

# ======================================================
# Label Encoding (XGBoost requirement)
# ======================================================
for col in cat_cols:
    le = LabelEncoder()
    
    combined_values = pd.concat([X_train[col], X_valid[col]], axis=0).astype(str)
    le.fit(combined_values)
    
    X_train[col] = le.transform(X_train[col].astype(str))
    X_valid[col] = le.transform(X_valid[col].astype(str))

# ======================================================
# Ensure all columns numeric
# ======================================================
X_train = X_train.apply(pd.to_numeric, errors="ignore")
X_valid = X_valid.apply(pd.to_numeric, errors="ignore")

# ======================================================
# Clean feature names for safety
# ======================================================
def clean_name(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

X_train.columns = [clean_name(c) for c in X_train.columns]
X_valid.columns = [clean_name(c) for c in X_valid.columns]

# ======================================================
# Train XGBoost
# ======================================================
model = XGBRegressor(
    n_estimators=617,
    learning_rate=0.0050,
    max_depth=9,
    min_child_samples=13,     # LightGBM equivalent of min_child_weight
    subsample=0.6605,
    colsample_bytree=0.5659,
    reg_alpha=0.024,
    reg_lambda=0.0018,
    random_state=42
)


model.fit(X_train, y_train)

# ======================================================
# Evaluate
# ======================================================
pred = model.predict(X_valid)

print("MAE :", mean_absolute_error(y_valid, pred))
print("MSE :", mean_squared_error(y_valid, pred))
print("RMSE:", mean_squared_error(y_valid, pred) ** 0.5)
print("R²  :", r2_score(y_valid, pred))

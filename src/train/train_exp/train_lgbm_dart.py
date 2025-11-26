import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ==========================================
# 🔧 CONFIG
# ==========================================
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\lgbm_dart"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Median PFS"
GROUP_COL = "Trial ID"

# Only drop if column exists
SAFE_DROP_COLS = ["Trial ID", "Date of Publication"]

RANDOM_SEED = 42

# ==========================================
# 📥 LOAD DATA
# ==========================================
print(f"Loading data: {INPUT_PATH}")
df = pd.read_excel(INPUT_PATH)

if TARGET not in df.columns:
    raise ValueError(f"❌ Target '{TARGET}' not found in file")

# ensure groups exist
df = df[df[GROUP_COL].notna()].copy()
df[GROUP_COL] = df[GROUP_COL].astype(str)

# ==========================================
# 🧹 CLEANING
# ==========================================
all_features = [c for c in df.columns if c != TARGET]
numeric_cols = df[all_features].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in all_features if c not in numeric_cols and c != GROUP_COL]

print("Numeric:", numeric_cols)
print("Categorical:", cat_cols)

# numeric fill
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# categorical fill
for col in cat_cols:
    df[col] = df[col].fillna("").astype(str)

# ==========================================
# SAFE DROP
# ==========================================
features = [c for c in all_features if c not in SAFE_DROP_COLS or c not in df.columns]
features = [c for c in all_features if c in df.columns and c not in SAFE_DROP_COLS]

print("Final features:", features)
print(f"Feature count: {len(features)}")

# ==========================================
# 🔀 GROUP SHUFFLE SPLIT
# ==========================================
X = df[features].copy()
y = df[TARGET].values
groups = df[GROUP_COL].values

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
train_idx, valid_idx = next(gss.split(X, y, groups))

X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
y_train, y_valid = y[train_idx], y[valid_idx]

# convert categoricals for LGBM
for col in cat_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")

# ==========================================
# SCALE NUMERICS
# ==========================================
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_valid[numeric_cols] = scaler.transform(X_valid[numeric_cols])

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# ==========================================
# 🧠 LIGHTGBM DART
# ==========================================
model = lgb.LGBMRegressor(
    boosting_type="dart",
    objective="regression",
    metric="rmse",
    random_state=RANDOM_SEED,

    learning_rate=0.03,
    num_leaves=64,
    max_depth=10,
    n_estimators=1500,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.4,

    drop_rate=0.1,
    skip_drop=0.5,
)

print("\nTraining LGBM Dart...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    
    categorical_feature=[c for c in cat_cols if c in X_train.columns]
)

# ==========================================
# 📈 PERFORMANCE
# ==========================================
pred = model.predict(X_valid)

print("\n==============================")
print(" LGBM DART PERFORMANCE")
print("==============================")
print("MAE  :", mean_absolute_error(y_valid, pred))
print("RMSE :", np.sqrt(mean_squared_error(y_valid, pred)))
print("R²   :", r2_score(y_valid, pred))

# ==========================================
# SAVE ARTIFACTS
# ==========================================
model.booster_.save_model(os.path.join(OUTPUT_DIR, "lgbm_dart_model.txt"))
joblib.dump(model, os.path.join(OUTPUT_DIR, "lgbm_dart_model.pkl"))

pd.Series(features).to_csv(os.path.join(OUTPUT_DIR, "features.csv"), index=False)

print("\nModel saved to:", OUTPUT_DIR)

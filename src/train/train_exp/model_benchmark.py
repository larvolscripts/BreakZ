import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

# ==========================================
# ⚙️ Config
# ==========================================
LOG_TRANSFORM = True
DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 📂 Load & Clean Data
# ==========================================
print("📂 Loading dataset...")
df = pd.read_excel(DATA_PATH)
df = df.replace(["-", " ", ""], pd.NA)

numeric_cols = [
    "Arm N",
    "Objective Response Rate N",
    "Objective Response Rate Percentage",
    "Duration of Response Median",
    "Median PFS",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Date of Publication"] = pd.to_datetime(df["Date of Publication"], errors="coerce")
df["Publication Year"] = df["Date of Publication"].dt.year
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

cat_cols = ["Type", "Product", "Dosage"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

df = df.dropna(subset=["Median PFS"])

target = "Median PFS"
features = [
    "Type",
    "Product",
    "Dosage",
    "Arm N",
    "Objective Response Rate N",
    "Objective Response Rate Percentage",
    "Duration of Response Median",
    "Publication Year",
]

X = df[features]
y = df[target]
if LOG_TRANSFORM:
    y = np.log1p(y)

cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# ==========================================
# ✂️ Split Data
# ==========================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")

# ==========================================
# 🔠 Encode Categoricals for LightGBM/XGBoost
# ==========================================
X_train_enc = X_train.copy()
X_valid_enc = X_valid.copy()
for col in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([X_train[col], X_valid[col]]).astype(str)
    le.fit(all_vals)
    X_train_enc[col] = le.transform(X_train[col].astype(str))
    X_valid_enc[col] = le.transform(X_valid[col].astype(str))

# ==========================================
# 🧠 Model Definitions
# ==========================================
models = {}

models["LightGBM"] = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    objective="regression",
)

models["CatBoost"] = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    cat_features=cat_features,
    random_seed=42,
    od_type="Iter",
    od_wait=50,
    use_best_model=True,
    verbose=False,
)

models["XGBoost"] = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
)

models["RandomForest"] = RandomForestRegressor(
    n_estimators=500, random_state=42, n_jobs=-1
)

models["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# ==========================================
# 🚀 Training & Evaluation
# ==========================================
results = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    if name in ["LightGBM", "XGBoost"]:
        model.fit(X_train_enc, y_train)
        y_pred = model.predict(X_valid_enc)
    elif name == "CatBoost":
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        y_pred = model.predict(X_valid)
    else:
        model.fit(X_train_enc, y_train)
        y_pred = model.predict(X_valid_enc)

    if LOG_TRANSFORM:
        y_pred = np.expm1(y_pred)
        y_valid_eval = np.expm1(y_valid)
    else:
        y_valid_eval = y_valid

    mae = mean_absolute_error(y_valid_eval, y_pred)
    r2 = r2_score(y_valid_eval, y_pred)
    results.append({"Model": name, "MAE": mae, "R2": r2})

    print(f"📊 {name} - MAE: {mae:.3f} | R²: {r2:.3f}")

# ==========================================
# 💾 Results Summary
# ==========================================
results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n🏁 Final Model Comparison:")
print(results_df)

results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

# ==========================================
# 📊 Feature Importances
# ==========================================
for name, model in models.items():
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        imp.to_csv(os.path.join(OUTPUT_DIR, f"{name}_feature_importance.csv"), index=False)

        plt.figure(figsize=(8,5))
        plt.barh(imp["Feature"], imp["Importance"])
        plt.title(f"{name} Feature Importance")
        plt.xlabel("Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_feature_importance.png"))
        plt.close()

# ==========================================
# 📈 Model Performance Plot
# ==========================================
plt.figure(figsize=(8,5))
plt.bar(results_df["Model"], results_df["R2"], color="teal")
plt.title("Model Comparison - R² Score")
plt.ylabel("R² Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_performance_r2.png"))
plt.close()

plt.figure(figsize=(8,5))
plt.bar(results_df["Model"], results_df["MAE"], color="orange")
plt.title("Model Comparison - Mean Absolute Error (Lower is Better)")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_performance_mae.png"))
plt.close()

print(f"\n✅ All results and plots saved in: {OUTPUT_DIR}")

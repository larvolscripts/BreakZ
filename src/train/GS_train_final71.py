# -----------------correct--data leakage proof------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
import re

# ==========================================
# ⚙️ Config
# ==========================================
LOG_TRANSFORM = False
CLEANED_DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\outputs\MedianPFS_cleaned.xlsx"
# CLEANED_DATA_PATH=r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned1.xlsx"
# CLEANED_DATA_PATH=r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 📂 Load Cleaned Dataset
# ==========================================
print("📂 Loading cleaned dataset...")
df = pd.read_excel(CLEANED_DATA_PATH)
print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns from cleaned file")

# ==========================================
# 🚫 Remove Outliers (Median PFS > 40)
# # ==========================================
initial_rows = df.shape[0]
df = df[df["Median PFS"] <=40]
removed = initial_rows - df.shape[0]
print(f"✅ Removed {removed} rows with Median PFS > 40. Remaining: {df.shape[0]}")



# ==========================================
# 📊 Correlation Heatmap
# ==========================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# ==========================================
# 🧩 Features & Target
# ==========================================
target = "Median PFS"
drop_cols = ["Trial ID", "Arm ID",'Date of Publication']  # will be dropped after grouping
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# ==========================================
# 🧬 Group-based Split (No Data Leakage)
# ==========================================
if "Trial ID" in df.columns:
    groups = df["Trial ID"]
    print("✅ Using Trial ID for group-based split to prevent leakage.")
else:
    groups = np.arange(len(df))
    print("⚠️ Trial ID not found — using random split fallback.")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(gss.split(X, y, groups=groups))

X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

if "Trial ID" in X_train.columns:
    train_trials = set(X_train["Trial ID"])
    valid_trials = set(X_valid["Trial ID"])
    overlap = train_trials.intersection(valid_trials)
    print(f"🧬 Unique Trial IDs → Train: {len(train_trials)}, Valid: {len(valid_trials)}, Overlap: {len(overlap)}")

# Drop ID columns AFTER grouping
for id_col in ["Trial ID", "Arm ID"]:
    if id_col in X_train.columns:
        X_train = X_train.drop(columns=id_col)
        X_valid = X_valid.drop(columns=id_col)
        print(f"🧹 Dropped ID column: {id_col}")

print(f"📊 Final shapes → Train: {X_train.shape}, Valid: {X_valid.shape}")

# ==========================================
# 🔠 Encode Categoricals for LightGBM
# ==========================================
cat_cols = [col for col in X_train.select_dtypes(include=["object", "category"]).columns]
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_valid[col] = X_valid[col].astype("category")


def clean_feature_name(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name))

X_train.columns = [clean_feature_name(c) for c in X_train.columns]
X_valid.columns = [clean_feature_name(c) for c in X_valid.columns]

# ==========================================
# 🧠 Train LightGBM (Tuned) 0.71
# ==========================================
# model = LGBMRegressor(
#     n_estimators=1200,
#     learning_rate=0.03,
#     num_leaves=40,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     reg_alpha=0.1,
#     reg_lambda=0.3,
#     random_state=42,
#     objective="regression"
# )

model = LGBMRegressor(
    learning_rate=0.015527206376765645,
    n_estimators=1811,
    num_leaves=64,
    max_depth=10,
    min_child_samples=70,
    subsample=0.7125461002991458,
    colsample_bytree=0.8120212437478006,
    reg_alpha=0.8289344618192059,
    reg_lambda=0.9065594659696742,
    random_state=42,
    objective="regression"
)

print("\n🚀 Training LightGBM model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="l1",
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(period=100),
    ]
)

# ==========================================
# 📈 Evaluate Model Performance
# ==========================================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_valid)

# Metrics
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, y_pred)

print("\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# ==========================================
# 📉 Actual vs Predicted Plot
# ==========================================
plt.figure(figsize=(6, 6))
plt.scatter(y_valid, y_pred, alpha=0.7)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')

plt.xlabel("Actual Median PFS")
plt.ylabel("Predicted Median PFS")
plt.title("Actual vs Predicted (LightGBM, Grouped Split)")

# Annotate metrics
plt.text(
    0.05, 0.95,
    f"R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted_grouped.png"))
plt.close()

# ==========================================
# 💾 Save Model
# ==========================================
model_path = os.path.join(OUTPUT_DIR, "lightgbm_grouped_pfs.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")

# ==========================================
# 🔍 Feature Importance
# ==========================================
imp = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(imp["Feature"], imp["Importance"])
plt.title("LightGBM Feature Importance (Grouped Split)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_grouped_pfs.png"))
plt.close()

# ==========================================
# 💡 SHAP Analysis
# ==========================================
print("\n⚡ Calculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)

# LightGBM may return a list → select first element
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Bar summary
shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar_grouped_pfs.png"),
            bbox_inches="tight")
plt.close()

# Beeswarm summary
shap.summary_plot(shap_values, X_valid, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm_grouped_pfs.png"),
            bbox_inches="tight")
plt.close()

print(f"\n✅ All outputs saved in: {OUTPUT_DIR}")


# # ==========================================
# # 📈 Evaluate
# # ==========================================
# y_pred = model.predict(X_valid)
# mae = mean_absolute_error(y_valid, y_pred)
# r2 = r2_score(y_valid, y_pred)

# print(f"\n📊 Model Performance:")
# print(f"Mean Absolute Error: {mae:.3f}")
# print(f"R² Score: {r2:.3f}")

# plt.figure(figsize=(6, 6))
# plt.scatter(y_valid, y_pred, alpha=0.7)
# plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
# plt.xlabel("Actual Median PFS")
# plt.ylabel("Predicted Median PFS")
# plt.title("Actual vs Predicted (LightGBM, Grouped Split)")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted_grouped.png"))
# plt.close()

# # ==========================================
# # 💾 Save Model
# # ==========================================
# model_path = os.path.join(OUTPUT_DIR, "lightgbm_grouped_pfs.pkl")
# joblib.dump(model, model_path)
# print(f"✅ Model saved to: {model_path}")

# # ==========================================
# # 🔍 Feature Importance
# # ==========================================
# imp = pd.DataFrame({"Feature": X_train.columns, "Importance": model.feature_importances_})
# imp = imp.sort_values("Importance", ascending=False)
# plt.figure(figsize=(10, 6))
# plt.barh(imp["Feature"], imp["Importance"])
# plt.title("LightGBM Feature Importance (Grouped Split)")
# plt.xlabel("Importance")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_grouped_pfs.png"))
# plt.close()

# # ==========================================
# # 📈 Evaluate
# # ==========================================
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np

# y_pred = model.predict(X_valid)

# # Metrics
# mae = mean_absolute_error(y_valid, y_pred)
# mse = mean_squared_error(y_valid, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_valid, y_pred)

# print(f"\n📊 Model Performance:")
# print(f"Mean Absolute Error (MAE): {mae:.3f}")
# print(f"Mean Squared Error (MSE): {mse:.3f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
# print(f"R² Score: {r2:.3f}")

# # ==========================================
# # 📉 Actual vs Predicted Plot with R²
# # ==========================================
# plt.figure(figsize=(6, 6))
# plt.scatter(y_valid, y_pred, alpha=0.7)
# plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')

# plt.xlabel("Actual Median PFS")
# plt.ylabel("Predicted Median PFS")
# plt.title("Actual vs Predicted (LightGBM, Grouped Split) PM")

# # 🧾 Annotate R² and metrics on the plot
# plt.text(
#     0.05, 0.95,
#     f"R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
#     transform=plt.gca().transAxes,
#     fontsize=10,
#     verticalalignment='top',
#     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
# )

# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted_grouped_PM.png"))
# plt.close()


# # ==========================================
# # 💡 SHAP Analysis
# # ==========================================
# print("\n⚡ Calculating SHAP values...")
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_valid)
# if isinstance(shap_values, list):
#     shap_values = shap_values[0]

# shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
# plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar_grouped_pfs_PM.png"), bbox_inches="tight")
# plt.close()

# shap.summary_plot(shap_values, X_valid, show=False)
# plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm_grouped_pfs_PM.png"), bbox_inches="tight")
# plt.close()

# print(f"\n✅ All outputs saved in: {OUTPUT_DIR}")

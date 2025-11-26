import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# ⚙️ Config
# ==========================================
LOG_TRANSFORM = False
CLEANED_DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\outputs\MedianPFS_cleaned.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 📂 Load Cleaned Dataset
# ==========================================
print("📂 Loading cleaned dataset...")
df = pd.read_excel(CLEANED_DATA_PATH)
print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns from cleaned file")

# ==========================================
# 🕒 Handle Date of Publication → Extract Year
# ==========================================
if "Date of Publication" in df.columns:
    # Convert to datetime safely
    df["Date of Publication"] = pd.to_datetime(df["Date of Publication"], errors="coerce")
    
    # Extract only the year
    df["Publication Year"] = df["Date of Publication"].dt.year
    
    # Drop the full date column
    df = df.drop(columns=["Date of Publication"])
    
    print("✅ Extracted 'Publication Year' and removed 'Date of Publication' column.")


# ==========================================
# 🚫 Remove Outliers (Median PFS > 60)
# ==========================================
initial_rows = df.shape[0]
df = df[df["Median PFS"] <= 60]
removed = initial_rows - df.shape[0]
print(f"✅ Removed {removed} rows with Median PFS > 50. Remaining: {df.shape[0]}")

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

# Drop non-predictive ID columns if present
drop_cols = ["Trial ID", "Arm ID"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# Remove any datetime columns
datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns
if len(datetime_cols) > 0:
    print(f"🕒 Removing datetime columns (not supported by LightGBM): {list(datetime_cols)}")
    X = X.drop(columns=datetime_cols)

# ✅ Let LightGBM handle categoricals natively
cat_cols = [col for col in X.select_dtypes(include=["object", "category"]).columns]
for col in cat_cols:
    X[col] = X[col].astype("category")

# ==========================================
# ✂️ Split Data
# ==========================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"📊 Train: {X_train.shape}, Valid: {X_valid.shape}")

# ==========================================
# 🧠 Train LightGBM (Tuned)
# ==========================================
model = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=40,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.3,
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
# 📈 Evaluate
# ==========================================
y_pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f"\n📊 Model Performance:")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_valid, y_pred, alpha=0.7)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.xlabel("Actual Median PFS")
plt.ylabel("Predicted Median PFS")
plt.title("Actual vs Predicted (LightGBM)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted_exp.png"))
plt.close()

# ==========================================
# 💾 Save Model
# ==========================================
model_path = os.path.join(OUTPUT_DIR, "lightgbm__exp.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")

# ==========================================
# 🔍 Feature Importance
# ==========================================
imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
imp = imp.sort_values("Importance", ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(imp["Feature"], imp["Importance"])
plt.title("LightGBM Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_exp.png"))
plt.close()

# ==========================================
# 💡 SHAP Analysis
# ==========================================
print("\n⚡ Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar_exp.png"), bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_valid, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.exp.png"), bbox_inches="tight")
plt.close()

print(f"\n✅ All outputs saved in: {OUTPUT_DIR}")


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import shap
# import joblib
# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score

# # ==========================================
# # ⚙️ Config
# # ==========================================
# LOG_TRANSFORM = False
# CLEANED_DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\outputs\MedianPFS_cleaned.xlsx"
# OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ==========================================
# # 📂 Load Cleaned Dataset
# # ==========================================
# print("📂 Loading cleaned dataset...")
# df = pd.read_excel(CLEANED_DATA_PATH)
# print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns from cleaned file")

# # ==========================================
# # 🕒 Handle Date + ID Columns
# # ==========================================
# # Convert date to year and drop original date column
# if "Date of Publication" in df.columns:
#     df["Date of Publication"] = pd.to_datetime(df["Date of Publication"], errors="coerce")
#     df["Publication Year"] = df["Date of Publication"].dt.year
#     df = df.drop(columns=["Date of Publication"])

# # Drop non-predictive identifier columns if present
# for col in ["Trial ID", "Arm ID"]:
#     if col in df.columns:
#         df = df.drop(columns=[col])
# print("✅ Dropped ID and datetime columns if present.")

# # ==========================================
# # 🚫 Remove Outliers (Median PFS > 50)
# # ==========================================
# initial_rows = df.shape[0]
# df = df[df["Median PFS"] <= 60]
# removed = initial_rows - df.shape[0]
# print(f"✅ Removed {removed} rows with Median PFS > 50. Remaining: {df.shape[0]}")

# # ==========================================
# # 📊 Correlation Heatmap
# # ==========================================
# numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# plt.figure(figsize=(10, 8))
# sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap (Numeric Features)")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
# plt.close()

# # ==========================================
# # 🧩 Features & Target
# # ==========================================
# target = "Median PFS"
# features = [col for col in df.columns if col != target]

# X = df[features]
# y = df[target]

# # ✅ Let LightGBM handle categoricals natively
# cat_cols = [col for col in X.select_dtypes(include=["object", "category"]).columns]
# for col in cat_cols:
#     X[col] = X[col].astype("category")

# # ==========================================
# # ✂️ Split Data
# # ==========================================
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# print(f"📊 Train: {X_train.shape}, Valid: {X_valid.shape}")

# # ==========================================
# # 🧠 Train LightGBM (Tuned)
# # ==========================================
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

# print("\n🚀 Training LightGBM model...")
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_valid, y_valid)],
#     eval_metric="l1",
#     callbacks=[
#         early_stopping(stopping_rounds=100),
#         log_evaluation(period=100),
#     ]
# )

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
# plt.title("Actual vs Predicted (LightGBM)")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted_exp.png"))
# plt.close()

# # ==========================================
# # 💾 Save Model
# # ==========================================
# model_path = os.path.join(OUTPUT_DIR, "lightgbm__exp.pkl")
# joblib.dump(model, model_path)
# print(f"✅ Model saved to: {model_path}")

# # ==========================================
# # 🔍 Feature Importance
# # ==========================================
# imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
# imp = imp.sort_values("Importance", ascending=False)
# plt.figure(figsize=(10, 6))
# plt.barh(imp["Feature"], imp["Importance"])
# plt.title("LightGBM Feature Importance")
# plt.xlabel("Importance")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_exp.png"))
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
# plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar_exp.png"), bbox_inches="tight")
# plt.close()

# shap.summary_plot(shap_values, X_valid, show=False)
# plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.exp.png"), bbox_inches="tight")
# plt.close()

# print(f"\n✅ All outputs saved in: {OUTPUT_DIR}")

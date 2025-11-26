import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import re

pd.options.mode.chained_assignment = None  # disable pandas SettingWithCopyWarning

# ==========================================
# ⚙️ Config
# ==========================================
# CLEANED_DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned.xlsx"

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# testing on primary_moa_category
# CLEANED_DATA_PATH=r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_categories.xlsx"
# 📊  Evaluation Results:
# MAE  : 1.7569
# MSE  : 6.2719
# RMSE : 2.5044
# R²   : 0.7290
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# testing on MOA--------------------
# CLEANED_DATA_PATH=r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_clusters.xlsx"
# 📊 Evaluation Results:
# MAE  : 1.6720
# MSE  : 5.9933
# RMSE : 2.4481
# R²   : 0.7410
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # testing on prod cat_kmeans
# CLEANED_DATA_PATH=r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_ProductCluster.xlsx"
# # 📊 Evaluation Results:
# MAE  : 1.7359
# MSE  : 6.5185
# RMSE : 2.5531
# R²   : 0.7183

# testing precise_kmeans_cat_best performance: 0.7413 stable model
CLEANED_DATA_PATH=r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"


# 📊 Evaluation Results:
# MAE  : 1.6832
# MSE  : 5.9859
# RMSE : 2.4466
# R²   : 0.7413

# V4
# CLEANED_DATA_PATH=r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_FEATURES_READY_V4.1.xlsx"
# --------------------------------------------------------------------
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 📂 Load Cleaned Dataset
# ==========================================
print("📂 Loading cleaned dataset...")
df = pd.read_excel(CLEANED_DATA_PATH)
print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns from cleaned file")

# ==========================================
# 🏷️ Add Product Category (does NOT remove Product)
# ==========================================

def categorize_product(prod):
    """Assign broad product categories based on biological mechanism keywords."""
    if pd.isna(prod) or str(prod).strip() == "":
        return "Other"


    p = str(prod).lower()

    # Immunotherapy (IO)
    if any(k in p for k in [
        "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
        "avelumab", "ipilimumab", "pd-1", "pdl1", "ctla"
    ]):
        return "Immunotherapy"

    # Targeted therapies (TKIs, inhibitors)
    if any(k in p for k in [
        "inhibitor", "tinib", "met", "vegf", "her2", "egfr", 
        "alk", "ros1", "braf", "mek"
    ]):
        return "Targeted"

    # Classic chemotherapy
    if any(k in p for k in [
        "carboplatin", "cisplatin", "oxaliplatin",
        "paclitaxel", "docetaxel",
        "fluorouracil", "5-fu",
        "cyclophosphamide", "etoposide", "irinotecan"
    ]):
        return "Chemotherapy"

    # Hormonal
    if any(k in p for k in [
        "letrozole", "tamoxifen", "anastrozole", "enzalutamide"
    ]):
        return "Hormonal"

    # ADCs, monoclonal antibodies
    if any(k in p for k in ["adc", "conjugate", "mab"]):
        return "Antibody/ADC"

    return "Other"


# Create the new feature
df["Product_Category"] = df["Product"].apply(categorize_product)

print("\n🎯 Added Product_Category")
print(df["Product_Category"].value_counts())

# ==========================================
# 💾 UPDATE ORIGINAL CLEANED FILE WITH NEW COLUMN
# ==========================================
df.to_excel(CLEANED_DATA_PATH, index=False)
print(f"✅ Updated cleaned dataset saved back to: {CLEANED_DATA_PATH}")

# extra--------experiment--------

# ==========================================
# # 🧮 Number of Products
# # ==========================================
# def count_products(prod):
#     if pd.isna(prod) or str(prod).strip() == "":
#         return 0
#     # split by comma (because you already cleaned delimiters)
#     return len([x for x in str(prod).split(",") if x.strip() != ""])

# df["Num_Products"] = df["Product"].apply(count_products)

# # ==========================================
# # 🔢 Is Combination Treatment?
# # ==========================================
# df["Is_Combination"] = (df["Num_Products"] > 1).astype(int)

# df.to_excel(CLEANED_DATA_PATH, index=False)
# print("✅ Updated cleaned dataset saved with new engineered features!")



# df.to_excel(CLEANED_DATA_PATH, index=False)
# print("✅ Updated cleaned dataset saved with new engineered features!")

# -------------------------------------------------------------------
# ==========================================
# 🚫 Remove Outliers (Median PFS > 40)
# ==========================================
initial_rows = df.shape[0]
df = df[df["Median PFS"] <= 40]
removed = initial_rows - df.shape[0]
print(f"✅ Removed {removed} rows with Median PFS > 40. Remaining: {df.shape[0]}")

# ==========================================
# 📊 Correlation Heatmap
# ==========================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# ==========================================
# 🧩 Features & Target
# ==========================================
target = "Median PFS"
drop_cols = ["Trial ID", "Arm ID", "Date of Publication", "Dosage"]

features = [col for col in df.columns if col != target]

X = df[features].copy()
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

# Check leakage
if "Trial ID" in X_train.columns:
    train_trials = set(X_train["Trial ID"])
    valid_trials = set(X_valid["Trial ID"])
    overlap = train_trials.intersection(valid_trials)
    print(f"🧬 Unique Trial IDs → Train: {len(train_trials)}, Valid: {len(valid_trials)}, Overlap: {len(overlap)}")

# ==========================================
# 🧹 Drop ID + unwanted columns AFTER split
# ==========================================
for col in drop_cols:
    if col in X_train.columns:
        X_train = X_train.drop(columns=col)
    if col in X_valid.columns:
        X_valid = X_valid.drop(columns=col)

print(f"📊 Final shapes → Train: {X_train.shape}, Valid: {X_valid.shape}")

# ==========================================
# 🔠 Encode Categoricals for LightGBM
# ==========================================
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# Fill NaN to avoid category errors
X_train[cat_cols] = X_train[cat_cols].fillna("Unknown")
X_valid[cat_cols] = X_valid[cat_cols].fillna("Unknown")

# Convert to category dtype
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_valid[col] = X_valid[col].astype("category")

# ==========================================
# 🧹 Clean feature names (keep same content)
# ==========================================
def clean_feature_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))

X_train.columns = [clean_feature_name(c) for c in X_train.columns]
X_valid.columns = [clean_feature_name(c) for c in X_valid.columns]

# ==========================================
# 🧠 Train LightGBM
# =================CORRECT=========================
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
#     objective="regression",
# )

#  Tuned parameters___correct_prod_category

# model = LGBMRegressor(
#     learning_rate=0.015527206376765645,
#     n_estimators=1811,
#     num_leaves=64,
#     max_depth=10,
#     min_child_samples=70,
#     subsample=0.7125461002991458,
#     colsample_bytree=0.8120212437478006,
#     reg_alpha=0.8289344618192059,
#     reg_lambda=0.9065594659696742,
#     random_state=42,
#     objective="regression"
# )

# tuned with prod category

# model = LGBMRegressor(
#     learning_rate=0.09584385330632615,
#     n_estimators=504,
#     num_leaves=52,
#     max_depth=3,
#     min_child_samples=52,
#     subsample=0.6689314143327704,
#     colsample_bytree=0.9094850151192724,
#     reg_alpha=0.6986499874127731,
#     reg_lambda=0.61959447305784,
#     random_state=42,
#     objective='regression'
# )

# tuned kmeans prod

# 📊 Evaluation Results:
# MAE  : 1.6832
# MSE  : 5.9859
# RMSE : 2.4466
# # R²   : 0.7413
model = LGBMRegressor(
    learning_rate=0.036665670589045796,
    n_estimators=755,
    num_leaves=59,
    max_depth=7,
    min_child_samples=41,
    subsample=0.9682755282162213,
    colsample_bytree=0.8114053624407164,
    reg_alpha=0.6484128017656786,
    reg_lambda=0.1703169991567584,
    random_state=42,
    objective='regression')

# --------------------------------------------corect---0.7413------------------------
# # tuned params kmeans testing

# model = LGBMRegressor(
#     learning_rate=0.06835005185428475,
#     n_estimators=1996,
#     num_leaves=73,
#     max_depth=10,
#     min_child_samples=28,
#     subsample=0.7134575258380971,
#     colsample_bytree=0.7840572877685461,
#     reg_alpha=0.18116299374653042,
#     reg_lambda=0.5771346788085655,
#     objective="regression",
#     random_state=42
# )

# # tuned paramess_rule based

# model = LGBMRegressor(
#     learning_rate=0.03369140194146386,
#     n_estimators=708,
#     num_leaves=57,
#     max_depth=10,
#     min_child_samples=98,
#     subsample=0.9713654430848173,
#     colsample_bytree=0.9499248652757907,
#     reg_alpha=0.534231250678175,
#     reg_lambda=0.9987707314311589,
#     objective="regression",
#     random_state=42
# )

# # TUNE_PRECISE_KMESNA
# model = LGBMRegressor(
#     learning_rate=0.06279440733648266,
#     n_estimators=1676,
#     num_leaves=71,
#     max_depth=10,
#     min_child_samples=49,
#     subsample=0.6542195680379892,
#     colsample_bytree=0.5024160100775839,
#     reg_alpha=0.6508835663953706,
#     reg_lambda=0.60551458978122,
#     random_state=42
# )

# V4
# model = LGBMRegressor(
#     learning_rate=0.048507922232488114,
#     n_estimators=1566,
#     num_leaves=62,
#     max_depth=10,
#     min_child_samples=98,
#     subsample=0.6375285446229005,
#     colsample_bytree=0.6699952075968806,
#     reg_alpha=0.9971493879673339,
#     reg_lambda=0.4731904199348796,
#     random_state=42,
#     objective='regression'
# )

# model = LGBMRegressor(
#     boosting_type="dart",       # required for drop_rate & skip_drop
#     learning_rate=0.1290067253288081,
#     n_estimators=2531,
#     num_leaves=67,
#     max_depth=5,
#     min_child_samples=59,
#     subsample=0.5825453457757226,
#     colsample_bytree=0.7148538589793427,
#     reg_alpha=0.8638900372842315,
#     reg_lambda=0.5824582803960838,
#     drop_rate=0.30592644736118974,
#     skip_drop=0.13949386065204183,
#     random_state=42
# )

# LGBM- BO OPTIMISED PARAMS----
# model = LGBMRegressor(
#     learning_rate=0.011371624499242713,
#     num_leaves=49,
#     max_depth=8,
#     min_child_samples=71,
#     subsample=0.6653465017096492,
#     colsample_bytree=0.9032519426445318,
#     reg_alpha=0.99003283954799,
#     reg_lambda=0.08534313593307497,
#     n_estimators=1367,
#     random_state=42
# )
print("\n🚀 Training LightGBM model...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="l1",
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(period=100),
    ],
)

# ==========================================
# 📈 Evaluation
# ==========================================
y_pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, y_pred)

print("\n📊 Evaluation Results:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ==========================================
# 📈 1. Prediction Plot (Actual vs Predicted)
# ==========================================
plt.figure(figsize=(7, 7))
plt.scatter(y_valid, y_pred, alpha=0.6, edgecolors="k")

# diagonal reference line
plt.plot([y_valid.min(), y_valid.max()],
         [y_valid.min(), y_valid.max()],
         "r--", linewidth=2)

plt.xlabel("Actual Median PFS", fontsize=12)
plt.ylabel("Predicted Median PFS", fontsize=12)
plt.title("Actual vs Predicted (LightGBM)", fontsize=14)

# Display metrics on plot
text = f"R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}"
plt.text(0.05, 0.95, text,
         transform=plt.gca().transAxes,
         fontsize=11,
         verticalalignment="top",
         bbox=dict(facecolor="white", alpha=0.8))

plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_vs_actual.png"), dpi=300)
plt.close()

# ==========================================
# 📊 2. Feature Importance Plot
# ==========================================
importances = model.feature_importances_
feature_names = X_train.columns

fi = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(
    data=fi,
    y="Feature",
    x="Importance",
    hue="Feature",
    palette="viridis",
    dodge=False,
    legend=False
)
plt.title("Feature Importance (LightGBM)", fontsize=14)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=300)
plt.close()

# ==========================================
# 🔍 3. SHAP Summary Plot
# ==========================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_valid, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_plot.png"), dpi=300)
plt.close()

# ==========================================
# 📊 4. SHAP Bar Plot (Mean |SHAP| Importance)
# ==========================================
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, 
    X_valid, 
    plot_type="bar", 
    show=False
)

plt.title("SHAP Feature Importance (Mean |SHAP| Values)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar_plot.png"), dpi=300)
plt.close()



# ==========================================
# 📉 High-Quality Actual vs Predicted Plot for PPT
# ==========================================
plt.figure(figsize=(7, 7), dpi=600)   # EXTRA high DPI for PPT

# High-quality scatter
plt.scatter(
    y_valid,
    y_pred,
    alpha=0.7,
    s=50,
    edgecolor="white",
    linewidth=0.5
)

# 1:1 diagonal
plt.plot(
    [y_valid.min(), y_valid.max()],
    [y_valid.min(), y_valid.max()],
    'r--',
    linewidth=2,
    alpha=0.9
)

# Labels
plt.xlabel("Actual Median PFS", fontsize=16)
plt.ylabel("Predicted Median PFS", fontsize=16)
plt.title("Actual vs Predicted (LightGBM, Grouped Split)", fontsize=18)

# Metrics box
plt.text(
    0.05, 0.95,
    f"R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.85)
)

plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
plt.tight_layout()

# ================================
# 🖼 Save in 3 formats for PPT
# ================================
png_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted_grouped_PFS.png")
svg_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted_grouped_PFS.svg")
pdf_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted_grouped_PFS.pdf")

plt.savefig(png_path, dpi=600, bbox_inches="tight")  # Best for PPT
plt.savefig(svg_path, bbox_inches="tight")           # Infinite resolution
plt.savefig(pdf_path, bbox_inches="tight")           # Publication/PPT
plt.close()

print("📁 Saved for PPT:")
print("   • PNG (super crisp):", png_path)
print("   • SVG (infinite resolution):", svg_path)
print("   • PDF (vector quality):", pdf_path)

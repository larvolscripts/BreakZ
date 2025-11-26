import pandas as pd
import numpy as np
import os
import re
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -------------------------------
# CONFIG
# -------------------------------
CLEANED_DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"

OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\cv_final_clean_script"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Median PFS"
DROP_COLS = ["Trial ID", "Arm ID", "Date of Publication", "Dosage"]

# -------------------------------
# TUNED PARAMETERS
# -------------------------------
LGB_PARAMS = {
    "learning_rate": 0.036665670589045796,
    "n_estimators": 755,
    "num_leaves": 59,
    "max_depth": 7,
    "min_child_samples": 41,
    "subsample": 0.9682755282162213,
    "colsample_bytree": 0.8114053624407164,
    "reg_alpha": 0.6484128017656786,
    "reg_lambda": 0.1703169991567584,
    "random_state": 42,
    "objective": "regression"
}

# -------------------------------
# Load data
# -------------------------------
print("Loading cleaned dataset...")
df = pd.read_excel(CLEANED_DATA_PATH)
print("Loaded:", df.shape)

# -------------------------------
# Remove outliers (Median PFS > 40)
# -------------------------------
initial = df.shape[0]
df = df[df["Median PFS"] <= 40]
print(f"Removed {initial - df.shape[0]} outliers. Remaining: {df.shape[0]}")

# -------------------------------
# Product Category engineering
# (same logic as training script)
# -------------------------------
def categorize_product(prod):
    if pd.isna(prod) or str(prod).strip() == "":
        return "Other"

    p = str(prod).lower()

    if any(k in p for k in [
        "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
        "avelumab", "ipilimumab", "pd-1", "pdl1", "ctla"
    ]):
        return "Immunotherapy"

    if any(k in p for k in [
        "inhibitor", "tinib", "met", "vegf", "her2","egfr",
        "alk","ros1","braf","mek"
    ]):
        return "Targeted"

    if any(k in p for k in [
        "carboplatin","cisplatin","oxaliplatin","paclitaxel",
        "docetaxel","fluorouracil","5-fu","cyclophosphamide",
        "etoposide","irinotecan"
    ]):
        return "Chemotherapy"

    if any(k in p for k in [
        "letrozole","tamoxifen","anastrozole","enzalutamide"
    ]):
        return "Hormonal"

    if any(k in p for k in ["adc", "conjugate", "mab"]):
        return "Antibody/ADC"

    return "Other"

df["Product_Category"] = df["Product"].apply(categorize_product)

# -------------------------------
# Features & target
# -------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

groups = df["Trial ID"]

# -------------------------------
# Clean feature names
# -------------------------------
def clean_feature_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))

X.columns = [clean_feature_name(c) for c in X.columns]

# -------------------------------
# 5-FOLD GROUP K-FOLD
# -------------------------------
gkf = GroupKFold(n_splits=5)

results = []
fold = 0

for train_idx, valid_idx in gkf.split(X, y, groups=groups):
    fold += 1
    print(f"\n===== FOLD {fold} =====")

    X_train = X.iloc[train_idx].copy()
    X_valid = X.iloc[valid_idx].copy()
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Drop unwanted cols AFTER splitting
    for c in DROP_COLS:
        if c in X_train.columns:
            X_train = X_train.drop(columns=c)
        if c in X_valid.columns:
            X_valid = X_valid.drop(columns=c)

    # categorical encoding
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train[cat_cols] = X_train[cat_cols].fillna("Unknown")
    X_valid[cat_cols] = X_valid[cat_cols].fillna("Unknown")

    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype("category")

    # LightGBM model
    model = LGBMRegressor(**LGB_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=100),
        ],
    )

    # Predictions
    pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, pred)
    mse = mean_squared_error(y_valid, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_valid, pred)

    print(f"Fold {fold} R² = {r2:.4f}")

    results.append([fold, mae, rmse, r2])

    # -----------------------------------
    # Save predictions Excel
    # -----------------------------------
    out = df.iloc[valid_idx][["Trial ID","Arm ID"]].copy()
    out["Actual"] = y_valid
    out["Pred"] = pred
    out.to_excel(f"{OUTPUT_DIR}/predictions_fold{fold}.xlsx", index=False)

    # -----------------------------------
    # Plot Actual vs Predicted
    # -----------------------------------
    plt.figure(figsize=(7,7))
    plt.scatter(y_valid, pred, alpha=0.6, edgecolor="k")
    plt.plot([y_valid.min(), y_valid.max()],
             [y_valid.min(), y_valid.max()],
             "r--", linewidth=2)
    plt.xlabel("Actual Median PFS")
    plt.ylabel("Predicted Median PFS")
    plt.title(f"Fold {fold} - Actual vs Predicted\nR²={r2:.3f}")
    plt.savefig(f"{OUTPUT_DIR}/fold{fold}_actual_vs_pred.png", dpi=300)
    plt.close()

    # -----------------------------------
    # Feature Importance
    # -----------------------------------
    fi = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fi.to_excel(f"{OUTPUT_DIR}/feature_importance_fold{fold}.xlsx", index=False)

    plt.figure(figsize=(8,10))
    sns.barplot(data=fi.head(30), x="Importance", y="Feature")
    plt.title(f"Fold {fold} Feature Importance (Top 30)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fold{fold}_feature_importance.png", dpi=300)
    plt.close()

    # -----------------------------------
    # SHAP
    # -----------------------------------
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_valid)

        plt.figure(figsize=(10,8))
        shap.summary_plot(shap_values, X_valid, show=False)
        plt.savefig(f"{OUTPUT_DIR}/fold{fold}_shap_summary.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10,8))
        shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
        plt.savefig(f"{OUTPUT_DIR}/fold{fold}_shap_bar.png", dpi=300)
        plt.close()

    except Exception as e:
        print(f"SHAP failed in fold {fold}:", e)

# -------------------------------
# Save final summary
# -------------------------------
res_df = pd.DataFrame(results, columns=["Fold","MAE","RMSE","R2"])
res_df.loc["Mean"] = res_df.mean(numeric_only=True)

print("\n===== FINAL 5-FOLD SUMMARY =====")
print(res_df)

res_df.to_excel(f"{OUTPUT_DIR}/cv_summary.xlsx", index=False)

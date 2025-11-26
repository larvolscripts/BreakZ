# train_with_embeddings.py
"""
Load original dataset + disease & drug embedding parquet files, merge, optionally PCA compress embeddings,
train LightGBM (use tunable params), save model, metrics, SHAP summary + bar plot.
"""

# FINAL METRICS
# MAE : 2.2679182919085576
# RMSE: 2.9001926263297926
# R2  : 0.7775125544638781

import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# ---------- CONFIG ----------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
DISEASE_PARQUET = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3_with_disease_emb.parquet"
DRUG_PARQUET = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3_with_drug_emb.parquet"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\with_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Median PFS"
GROUP_COL = "Trial ID"
DROP_COLS = ["Trial ID", "Arm ID", "Date of Publication", "Dosage", "Product"]
PCA_N_COMPONENTS = 50   # reduce 768 dims -> 50 (tune as you like)
RANDOM_SEED = 42

# Replace with your best optuna/tuned params (example from earlier)...0.77
# LGB_PARAMS = {
#     "boosting_type": "gbdt",
#     "objective": "regression",
#     "metric": "rmse",
#     "random_state": RANDOM_SEED,
#     "learning_rate": 0.03369140194146386,
#     "n_estimators": 708,
#     "num_leaves": 57,
#     "max_depth": 10,
#     "min_child_samples": 98,
#     "subsample": 0.9713654430848173,
#     "colsample_bytree": 0.9499248652757907,
#     "reg_alpha": 0.534231250678175,
#     "reg_lambda": 0.9987707314311589,
# }

# TUNED PARAMS------------
# FINAL METRICS
# MAE : 2.057927288857675
# RMSE: 2.7780920672010425
# R2  : 0.7958520184165069

LGB_PARAMS={
    "boosting_type": "gbdt",
  "learning_rate": 0.0037396367131530017,
  "n_estimators": 1351,
  "num_leaves": 227,
  "max_depth": 6,
  "min_child_samples": 198,
  "subsample": 0.7403684989264331,
  "colsample_bytree": 0.49227493824693724,
  "reg_alpha": 1.0975389761073688,
  "reg_lambda": 0.2247974008377543,
  "random_state": 42,
  "objective": "regression"
}
# ----------------------------

def load_and_merge():
    print("Loading base:", INPUT_PATH)
    df_base = pd.read_excel(INPUT_PATH, dtype=str)
    # read parquet of embeddings (if exist)
    print("Merging disease embed:", DISEASE_PARQUET)
    df_d = pd.read_parquet(DISEASE_PARQUET)
    print("Merging drug embed:", DRUG_PARQUET)
    df_dr = pd.read_parquet(DRUG_PARQUET)

    # prefer using same index/order approach: merge on index/Trial ID + Arm ID if present
    # Simplest: concat columns by index when input files have same ordering / same rows
    # safer: merge on Trial ID + Arm ID if present
    key_cols = []
    if "Trial ID" in df_base.columns and "Arm ID" in df_base.columns:
        key_cols = ["Trial ID", "Arm ID"]
        df_merged = df_base.merge(df_d[key_cols + [c for c in df_d.columns if c.startswith("disease_emb_")]],
                                  on=key_cols, how="left")
        df_merged = df_merged.merge(df_dr[key_cols + [c for c in df_dr.columns if c.startswith("drug_emb_")]],
                                    on=key_cols, how="left")
    else:
        # fallback: concat side-by-side (assumes same row order)
        df_merged = pd.concat([df_base, df_d[[c for c in df_d.columns if c.startswith("disease_emb_")]],
                               df_dr[[c for c in df_dr.columns if c.startswith("drug_emb_")]]], axis=1)

    # ---------------------------------------------------
    # 💾 SAVE FINAL MERGED DATASET (for training/debugging)
    # ---------------------------------------------------
    SAVE_MERGED = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_training_merged_with_embeddings.xlsx"
    print(f"\nSaving merged dataset to:\n{SAVE_MERGED}")
    df_merged.to_excel(SAVE_MERGED, index=False)
    print("✅ Saved merged dataset!")
    
    return df_merged

def compress_embeddings(df, prefix, n_components=PCA_N_COMPONENTS):
    emb_cols = [c for c in df.columns if c.startswith(prefix)]
    if len(emb_cols) == 0:
        print("No columns found for", prefix)
        return df, []
    X_emb = df[emb_cols].fillna(0).values.astype(float)
    print(f"PCA compressing {len(emb_cols)} -> {n_components}")
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(X_emb)
    new_cols = [f"{prefix}pc_{i}" for i in range(n_components)]
    df_pca = pd.DataFrame(Xp, columns=new_cols, index=df.index)
    df = pd.concat([df.drop(columns=emb_cols), df_pca], axis=1)
    # save pca
    joblib.dump(pca, os.path.join(OUTPUT_DIR, f"pca_{prefix}.pkl"))
    return df, new_cols

def main():
    df = load_and_merge()
    print("Shape after merge:", df.shape)

    # Convert numeric columns
    # safe list of numeric cols you had earlier:
    numeric_like = ["Arm N", "Objective Response Rate N", "Objective Response Rate Percentage",
                    "Duration of Response Median", "Response_Count_Duration", "Response_Percentage_Duration",
                    "Median PFS"]
    for c in numeric_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # PCA compress disease + drug embeddings
    df, disease_pc_cols = compress_embeddings(df, "disease_emb_", n_components=PCA_N_COMPONENTS)
    df, drug_pc_cols = compress_embeddings(df, "drug_emb_", n_components=PCA_N_COMPONENTS)

    # Build features list
    all_features = [c for c in df.columns if c != TARGET]
    # drop unwanted
    final_features = [c for c in all_features if c not in DROP_COLS]
    # keep PCA columns + other features
    print("Final feature count:", len(final_features))

    # drop rows with target missing
    df = df.dropna(subset=[TARGET])
    X = df[final_features].copy()
    y = df[TARGET].astype(float).values
    groups = df[GROUP_COL].astype(str).values if GROUP_COL in df.columns else None

    # split by groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    if groups is None:
        train_idx, valid_idx = next(gss.split(X, y))
    else:
        train_idx, valid_idx = next(gss.split(X, y, groups=groups))

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    # convert object columns to category for LGBM
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    print("Categorical columns:", cat_cols)
    for c in cat_cols:
        X_train[c] = X_train[c].fillna("").astype("category")
        X_valid[c] = X_valid[c].fillna("").astype("category")

    # Ensure numeric columns are numeric
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    print("Numeric columns:", len(num_cols))

    # Train LightGBM
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    print("Training LGBM ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        categorical_feature=cat_cols,
        # early_stopping_rounds=200,
        # verbose=100
    )

    # Evaluate
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    rmse = mean_squared_error(y_valid, pred, squared=False)
    r2 = r2_score(y_valid, pred)

    print("FINAL METRICS")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    # SHAP summary + bar
    print("Computing SHAP values (TreeExplainer) ...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_valid)
        # summary plot
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X_valid, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), bbox_inches="tight", dpi=250)
        plt.close()

        # bar plot of mean(|shap|)
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), bbox_inches="tight", dpi=250)
        plt.close()

        # save shap values & data (optional)
        np.save(os.path.join(OUTPUT_DIR, "shap_values.npy"), shap_values)
    except Exception as e:
        print("SHAP failed:", e)

    # Save model & features
    joblib.dump(model, os.path.join(OUTPUT_DIR, "lgbm_model.pkl"))
    pd.Series(final_features).to_csv(os.path.join(OUTPUT_DIR, "final_feature_list.csv"), index=False)

    print("Saved outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

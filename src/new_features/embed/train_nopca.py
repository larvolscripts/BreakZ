# train_with_embeddings_no_pca.py----------------
"""
Train LightGBM using full 768-dimensional disease + drug embeddings.
NO PCA.
Saves SHAP summary + bar plot + model.
"""

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- CONFIG -----------------------
INPUT_BASE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
DISEASE_PARQUET = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3_with_disease_emb.parquet"
DRUG_PARQUET = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3_with_drug_emb.parquet"

OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\with_embeddings_nopca"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Median PFS"
GROUP_COL = "Trial ID"

DROP_COLS = ["Trial ID", "Arm ID", "Date of Publication", "Dosage", "Product"]

RANDOM_SEED = 42

# Your best tuned LightGBM params
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "random_state": RANDOM_SEED,
    "learning_rate": 0.03369140194146386,
    "n_estimators": 708,
    "num_leaves": 57,
    "max_depth": 10,
    "min_child_samples": 98,
    "subsample": 0.9713654430848173,
    "colsample_bytree": 0.9499248652757907,
    "reg_alpha": 0.534231250678175,
    "reg_lambda": 0.9987707314311589,
}
# ------------------------------------------------


def load_merged():
    print("Loading base data…")
    df_base = pd.read_excel(INPUT_BASE)

    print("Loading disease embedding…")
    df_d = pd.read_parquet(DISEASE_PARQUET)

    print("Loading drug embedding…")
    df_dr = pd.read_parquet(DRUG_PARQUET)

    # -----------------------------
    # 🔧 FIX MERGE KEY TYPE MISMATCH
    # -----------------------------
    key_cols = ["Trial ID", "Arm ID"]

    for df_tmp in [df_base, df_d, df_dr]:
        for col in key_cols:
            if col in df_tmp.columns:
                df_tmp[col] = df_tmp[col].astype(str).str.strip()

    # -----------------------------
    # MERGE
    # -----------------------------
    df = df_base.merge(
        df_d[key_cols + [c for c in df_d.columns if c.startswith("disease_emb_")]],
        on=key_cols,
        how="left"
    )

    df = df.merge(
        df_dr[key_cols + [c for c in df_dr.columns if c.startswith("drug_emb_")]],
        on=key_cols,
        how="left"
    )

    print("Merged shape:", df.shape)
    return df






def main():
    df = load_merged()

    # convert numeric
    numeric_cols = ["Arm N","Objective Response Rate N","Objective Response Rate Percentage",
                    "Duration of Response Median","Response_Count_Duration","Response_Percentage_Duration",
                    "Median PFS"]

    for c in numeric_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[TARGET])

    # feature list
    all_features = [c for c in df.columns if c != TARGET]
    final_features = [c for c in all_features if c not in DROP_COLS]

    print("Final feature count:", len(final_features))

    X = df[final_features].copy()
    y = df[TARGET].values
    groups = df[GROUP_COL].astype(str).values

    # split
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
    train_idx, valid_idx = next(gss.split(X, y, groups))

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    # categorical → convert
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")

    print("Training LightGBM with full embeddings…")
    model = lgb.LGBMRegressor(**LGB_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        categorical_feature=cat_cols,
        # early_stopping_rounds=200,
        # verbose=100
    )

    # evaluation
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    rmse = mean_squared_error(y_valid, pred, squared=False)
    r2 = r2_score(y_valid, pred)

    print("\n==== FINAL RESULTS ====")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)

    # SHAP
    print("Computing SHAP…")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_valid)

    shap.summary_plot(shap_values, X_valid, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))
    pd.Series(final_features).to_csv(os.path.join(OUTPUT_DIR, "features.csv"), index=False)

    print("Saved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()


# ========================above is correct-----0.77


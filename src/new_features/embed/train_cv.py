# train_cv_with_embeddings_final.py
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb


# ======================
# CONFIG
# ======================
INPUT_MERGED = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_training_merged_with_embeddings.xlsx"
OUTPUT_DIR   = r"C:\LARVOL_WORK\Median_PFS\outputs\cv_with_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Median PFS"
GROUP_COL = "Trial ID"
DROP_COLS = ["Trial ID", "Arm ID", "Date of Publication", "Dosage", "Product"]

PCA_N_COMPONENTS = 80
RANDOM_SEED = 42

# -------------------------
# TUNED LIGHTGBM PARAMS
# -------------------------
LGB_PARAMS = {
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


# ============================================================
# 1) LOAD MERGED DATASET
# ============================================================
def load_data():
    print("\nLoading merged dataset...")
    df = pd.read_excel(INPUT_MERGED)
    print("Shape:", df.shape)

    # Convert target to numeric
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Standardize keys
    df["Trial ID"] = df["Trial ID"].astype(str).str.strip()
    df["Arm ID"]   = df["Arm ID"].astype(str).str.strip()

    # Fix embedding columns
    emb_cols = [c for c in df.columns if c.startswith("disease_emb_") or c.startswith("drug_emb_")]
    for c in emb_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


# ============================================================
# 2) PCA COMPRESSION (same as training)
# ============================================================
def compress_embeddings(df, prefix):
    emb_cols = [c for c in df.columns if c.startswith(prefix)]
    if len(emb_cols) == 0:
        print(f"No embedding columns found for {prefix}")
        return df, []

    X_emb = df[emb_cols].fillna(0).values.astype(float)

    print(f"PCA compressing {prefix}: {len(emb_cols)} → {PCA_N_COMPONENTS}")
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
    pcs = pca.fit_transform(X_emb)

    new_cols = [f"{prefix}pc_{i}" for i in range(PCA_N_COMPONENTS)]
    df_pca = pd.DataFrame(pcs, columns=new_cols, index=df.index)

    df = pd.concat([df.drop(columns=emb_cols), df_pca], axis=1)

    # Save the PCA model
    joblib.dump(pca, os.path.join(OUTPUT_DIR, f"pca_{prefix}.pkl"))

    return df, new_cols



# ============================================================
# MAIN CV FUNCTION
# ============================================================
def main():
    df = load_data()
    print("\n→ Running PCA compression...")
    df, _ = compress_embeddings(df, "disease_emb_")
    df, _ = compress_embeddings(df, "drug_emb_")

    # Final features
    all_features = [c for c in df.columns if c != TARGET]
    final_features = [c for c in all_features if c not in DROP_COLS]

    print("Final feature count:", len(final_features))

    # Drop missing target rows
    df = df.dropna(subset=[TARGET])

    X = df[final_features].copy()
    y = df[TARGET].astype(float).values
    groups = df[GROUP_COL].astype(str).values

    # Fix categorical dtype
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    # 5-fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    results = []

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups), start=1):

        print(f"\n========== Fold {fold} ==========")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        # Ensure category dtype
        for c in cat_cols:
            X_train[c] = X_train[c].astype("category")
            X_valid[c] = X_valid[c].astype("category")

        # ------------------------
        # Train Model
        # ------------------------
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train)

        pred = model.predict(X_valid)

        mae = mean_absolute_error(y_valid, pred)
        rmse = mean_squared_error(y_valid, pred, squared=False)
        r2 = r2_score(y_valid, pred)

        print(f"Fold {fold} → R² = {r2:.4f}")

        results.append([fold, mae, rmse, r2])

        # Save fold model
        joblib.dump(model, f"{OUTPUT_DIR}/model_fold{fold}.pkl")

        # Save predictions
        out = df.loc[valid_idx, ["Trial ID", "Arm ID"]].copy()
        out["Actual"] = y_valid
        out["Pred"] = pred
        out.to_excel(f"{OUTPUT_DIR}/preds_fold{fold}.xlsx", index=False)

        # ------------------------
        # SHAP (optional)
        # ------------------------
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_valid)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_valid, show=False)
            plt.savefig(f"{OUTPUT_DIR}/shap_summary_fold{fold}.png", dpi=250, bbox_inches="tight")
            plt.close()

        except Exception as e:
            print(f"SHAP failed on fold {fold}: {e}")

    # =======================
    # Results Summary
    # =======================
    res_df = pd.DataFrame(results, columns=["Fold", "MAE", "RMSE", "R2"])
    res_df.loc["Mean"] = res_df.mean(numeric_only=True)

    print("\n===== FINAL 5-FOLD SUMMARY =====")
    print(res_df)

    res_df.to_excel(f"{OUTPUT_DIR}/cv_summary.xlsx", index=False)
    print("\nSaved all CV outputs to:", OUTPUT_DIR)



if __name__ == "__main__":
    main()


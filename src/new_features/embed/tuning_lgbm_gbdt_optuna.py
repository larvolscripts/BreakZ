"""
tune_lgbm_gbdt_optuna.py

Optuna tuning for LightGBM (GBDT) with Group-aware splitting and embeddings merged.

Outputs saved to OUTPUT_DIR:
 - merged_input_used_for_tuning.xlsx
 - optuna_study.pkl
 - best_params.json
 - trials_summary.csv
 - best_model.pkl
"""

import os
import json
import joblib
from pathlib import Path
from typing import List

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------
# CONFIG: edit if needed
# -----------------------
INPUT_BASE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
DISEASE_PARQUET = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3_with_disease_emb.parquet"
DRUG_PARQUET = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3_with_drug_emb.parquet"

OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\tune_lgbm_gbdt_optuna"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Median PFS"
GROUP_COL = "Trial ID"
RANDOM_SEED = 42
N_TRIALS = 40       # change as desired (50-100 typical)
N_SPLITS = 3        # number of GroupShuffleSplit repeats used inside objective (averaged)
EARLY_STOPPING_ROUNDS = 100
VERBOSE = 100

# -----------------------
# Helper: load & merge
# -----------------------
def load_and_merge(base_path: str, disease_parquet: str, drug_parquet: str) -> pd.DataFrame:
    # Load base (as strings for keys)
    print("Loading base:", base_path)
    df_base = pd.read_excel(base_path, dtype=str)

    # If parquet files don't exist, skip or raise
    print("Loading disease embeddings:", disease_parquet)
    df_d = pd.read_parquet(disease_parquet)
    print("Loading drug embeddings:", drug_parquet)
    df_dr = pd.read_parquet(drug_parquet)

    # Normalize key column types & whitespace
    key_cols = []
    if "Trial ID" in df_base.columns and "Arm ID" in df_base.columns:
        key_cols = ["Trial ID", "Arm ID"]
        for df_tmp in (df_base, df_d, df_dr):
            for col in key_cols:
                if col in df_tmp.columns:
                    df_tmp[col] = df_tmp[col].astype(str).str.strip()
    else:
        raise ValueError("Base file must have 'Trial ID' and 'Arm ID' for safe merge.")

    # quick duplicate check on base
    dup = df_base.duplicated(subset=key_cols, keep=False).sum()
    if dup > 0:
        print(f"⚠️ Found {dup} duplicate rows (Trial ID + Arm ID) in base file. Keep duplicates if they are real arms; merge will proceed.")
    # merge
    cols_d = [c for c in df_d.columns if c.startswith("disease_emb_")]
    cols_dr = [c for c in df_dr.columns if c.startswith("drug_emb_")]
    # Keep only key_cols + embed columns from parquet
    df_merged = df_base.merge(df_d[key_cols + cols_d], on=key_cols, how="left")
    df_merged = df_merged.merge(df_dr[key_cols + cols_dr], on=key_cols, how="left")

    print("Merged shape:", df_merged.shape)
    return df_merged

# -----------------------
# Preprocessing for a trial
# -----------------------
def prepare_features(df: pd.DataFrame, target: str, drop_cols: List[str]=None):
    if drop_cols is None:
        drop_cols = []

    # Ensure target numeric
    df[target] = pd.to_numeric(df[target], errors="coerce")

    # drop rows without target
    df = df[df[target].notna()].copy()

    # Identify embedding columns (numeric) and numeric cols
    emb_cols = [c for c in df.columns if c.startswith("disease_emb_") or c.startswith("drug_emb_")]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove target from numeric list
    if target in numeric_cols:
        numeric_cols.remove(target)

    # Categorical candidate cols: object / string types excluding group and keys
    cat_cols = [c for c in df.columns
                if df[c].dtype == "object" and c not in [GROUP_COL, target] and c not in emb_cols]

    # Replace NaNs in categorical with empty string (no "Unknown")
    for c in cat_cols:
        df[c] = df[c].fillna("").astype(str)

    # Fill numeric embeddings NaN with 0 (or column median)
    for c in emb_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # fill other numeric columns with median
    for c in numeric_cols:
        if c not in emb_cols:
            df[c] = df[c].fillna(df[c].median())

    # final features: exclude group and any drop_cols
    all_features = [c for c in df.columns if c != target and c not in drop_cols]
    # ensure group col not in features
    if GROUP_COL in all_features:
        all_features.remove(GROUP_COL)

    return df, all_features, numeric_cols, cat_cols, emb_cols

# -----------------------
# Objective for Optuna
# -----------------------
def objective(trial: optuna.Trial, df: pd.DataFrame, features: list, numeric_cols: list, cat_cols: list, groups: pd.Series, target_series: pd.Series):
    # Suggest hyperparams
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": RANDOM_SEED,
        "objective": "regression",
        "verbosity": -1,
    }

    # We'll average R2 across N_SPLITS GroupShuffleSplit runs
    r2_scores = []
    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=RANDOM_SEED)

    for train_idx, valid_idx in gss.split(df[features], target_series, groups=groups):
        X_train = df.iloc[train_idx][features].copy()
        X_valid = df.iloc[valid_idx][features].copy()
        y_train = target_series.iloc[train_idx].astype(float)
        y_valid = target_series.iloc[valid_idx].astype(float)

        # scale numeric if you want (embedding columns remain numeric)
        scaler = StandardScaler()
        cols_to_scale = [c for c in numeric_cols if c in X_train.columns]
        if cols_to_scale:
            X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
            X_valid[cols_to_scale] = scaler.transform(X_valid[cols_to_scale])

        # ensure categorical columns cast to 'category'
        cat_present = [c for c in cat_cols if c in X_train.columns]
        for c in cat_present:
            X_train[c] = X_train[c].astype("category")
            X_valid[c] = X_valid[c].astype("category")

        model = LGBMRegressor(**params)

        # fit with early stopping on the validation set
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse",
                # early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                categorical_feature=cat_present,
                # verbose=False
            )
        except Exception as e:
            # if LightGBM complains about categorical_feature type, retry without categorical_feature arg
            # but cast to category should usually be fine
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="rmse")

        y_pred = model.predict(X_valid)
        r2 = r2_score(y_valid, y_pred)
        r2_scores.append(r2)

    # return average r2
    avg_r2 = float(np.mean(r2_scores))
    # report intermediate result (for optuna visualization)
    trial.set_user_attr("r2_scores", r2_scores)
    return avg_r2

# -----------------------
# Main: load, prepare, tune
# -----------------------
def main():
    # 1) load & merge
    df = load_and_merge(INPUT_BASE, DISEASE_PARQUET, DRUG_PARQUET)

    # 2) prepare features: drop original Product if you want, default keep everything except group col
    drop_cols = []  # you can add "Product" here if you prefer engineered product features instead
    df_prepared, all_features, numeric_cols, cat_cols, emb_cols = prepare_features(df, TARGET, drop_cols=drop_cols)

    # save merged dataset used for tuning
    merged_save = os.path.join(OUTPUT_DIR, "merged_input_used_for_tuning.xlsx")
    df_prepared.to_excel(merged_save, index=False)
    print("Saved merged dataset →", merged_save)

    # Ensure group vector and target
    groups = df_prepared[GROUP_COL].astype(str)
    target_series = df_prepared[TARGET].astype(float)

    print("Candidate feature count:", len(all_features))
    print("Numeric (sample):", numeric_cols[:10])
    print("Categorical (sample):", cat_cols[:10])
    print("Embeddings (sample):", emb_cols[:6])

    # 3) Optuna study
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    func = lambda trial: objective(trial, df_prepared, all_features, numeric_cols, cat_cols, groups, target_series)

    print(f"Starting optimization: {N_TRIALS} trials (this can take a while)...")
    study.optimize(func, n_trials=N_TRIALS, show_progress_bar=True)

    # 4) Save study & best model (retrain on full training set using best params)
    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED
    best_params["objective"] = "regression"

    # Save best params and trials
    joblib.dump(study, os.path.join(OUTPUT_DIR, "optuna_study.pkl"))
    with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(OUTPUT_DIR, "trials_summary.csv"), index=False)
    print("Best R²:", study.best_value)
    print("Best params saved to:", os.path.join(OUTPUT_DIR, "best_params.json"))
    print("Trials saved to:", os.path.join(OUTPUT_DIR, "trials_summary.csv"))

    # 5) Retrain best model on all data (no split) OR retrain on full data and save
    final_model = LGBMRegressor(**best_params)
    # cast categories and scale numeric
    df_all = df_prepared.copy()
    cols_to_scale = [c for c in numeric_cols if c in df_all.columns]
    scaler = StandardScaler()
    if cols_to_scale:
        df_all[cols_to_scale] = scaler.fit_transform(df_all[cols_to_scale])

    for c in cat_cols:
        if c in df_all.columns:
            df_all[c] = df_all[c].astype("category")

    final_model.fit(df_all[all_features], df_all[TARGET])
    # save model, scaler, and feature list
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, "best_lgbm_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_used_for_final.pkl"))
    pd.Series(all_features).to_csv(os.path.join(OUTPUT_DIR, "features_used.csv"), index=False)

    print("Saved final model and artifacts to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

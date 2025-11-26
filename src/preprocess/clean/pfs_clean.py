
# #  old features------catboost------------
# import os
# import pandas as pd
# import numpy as np
# from catboost import CatBoostRegressor, Pool
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score
# import matplotlib.pyplot as plt

# # ==========================================
# # ⚙️ Config
# # ==========================================
# LOG_TRANSFORM = True
# DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS.xlsx"
# OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# print("📂 Loading dataset...")
# df = pd.read_excel(DATA_PATH)
# print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# # ==========================================
# # 🧹 Data Cleaning
# # ==========================================
# print("🧹 Cleaning data...")

# # Replace "-" and blanks with NaN
# df = df.replace(["-", " ", ""], pd.NA)

# # Convert numeric columns
# numeric_cols = [
#     "Arm N",
#     "Objective Response Rate N",
#     "Objective Response Rate Percentage",
#     "Duration of Response Median",
#     "Median PFS"
# ]
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# # Convert date to year
# df["Date of Publication"] = pd.to_datetime(df["Date of Publication"], errors="coerce")
# df["Publication Year"] = df["Date of Publication"].dt.year

# # ✅ Fill missing numeric values (with mean)
# df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# # ✅ Fill missing categorical values (with "Unknown")
# cat_cols = ["Type", "Product", "Dosage"]
# for col in cat_cols:
#     if col in df.columns:
#         df[col] = df[col].fillna("Unknown")

# # Drop rows without target
# df = df.dropna(subset=["Median PFS"])

# print("✅ Missing values filled.")
# print(df.isna().sum())

# # ==========================================
# # 🧩 Features & Target
# # ==========================================
# target = "Median PFS"
# features = [
#     "Type",
#     "Product",
#     "Dosage",
#     "Arm N",
#     "Objective Response Rate N",
#     "Objective Response Rate Percentage",
#     "Duration of Response Median",
#     "Publication Year"
# ]

# X = df[features]
# y = df[target]

# if LOG_TRANSFORM:
#     y = np.log1p(y)

# cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# # ==========================================
# # ✂️ Split
# # ==========================================
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# print(f"\n📊 Train shape: {X_train.shape}, Validation shape: {X_valid.shape}")

# # ==========================================
# # 🧠 Train CatBoost
# # ==========================================
# print("\n🚀 Training CatBoost model...")
# # model = CatBoostRegressor(
# #     iterations=300,
# #     learning_rate=0.05,
# #     depth=6,
# #     loss_function="RMSE",
# #     cat_features=cat_features,
# #     eval_metric="RMSE",
# #     random_seed=42
# # )

# # model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

# model = CatBoostRegressor(
#     iterations=1000,               # match LightGBM n_estimators
#     learning_rate=0.05,
#     depth=6,
#     loss_function="RMSE",
#     cat_features=cat_features,
#     eval_metric="RMSE",
#     random_seed=42,
#     od_type="Iter",                # enables early stopping
#     od_wait=50,                    # stop if no improvement after 50 rounds
#     verbose=100,                   # log every 100 iterations (safer)
#     use_best_model=True            # keep the best model automatically
# )

# model.fit(
#     X_train,
#     y_train,
#     eval_set=(X_valid, y_valid)
# )

# # ==========================================
# # 📈 Evaluate
# # ==========================================
# y_pred = model.predict(X_valid)
# if LOG_TRANSFORM:
#     y_pred = np.expm1(y_pred)
#     y_valid = np.expm1(y_valid)

# mae = mean_absolute_error(y_valid, y_pred)
# r2 = r2_score(y_valid, y_pred)

# print(f"\n📊 Model Evaluation:")
# print(f"Mean Absolute Error: {mae:.3f}")
# print(f"R² Score: {r2:.3f}")

# # ==========================================
# # 💾 Save model
# # ==========================================
# model_path = os.path.join(OUTPUT_DIR, "model.cbm")
# model.save_model(model_path)
# print(f"✅ Model saved to: {model_path}")

# # ==========================================
# # 🔍 Feature Importance (standard)
# # ==========================================
# print("\n📊 Generating feature importance plot...")
# importances = model.get_feature_importance(prettified=True)
# plt.figure(figsize=(8, 5))
# plt.barh(importances["Feature Id"], importances["Importances"])
# plt.title("CatBoost Feature Importance")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
# plt.show()

# # ==========================================
# # 💡 Built-in SHAP (CatBoost only)
# # ==========================================
# print("\n⚡ Calculating CatBoost SHAP values...")
# pool_valid = Pool(X_valid, label=y_valid, cat_features=cat_features)
# shap_values = model.get_feature_importance(pool_valid, type="ShapValues")

# # shap_values: (n_samples, n_features + 1)
# shap_contrib = shap_values[:, :-1]  # drop bias term

# mean_abs_shap = np.abs(shap_contrib).mean(axis=0)
# shap_df = pd.DataFrame({
#     "Feature": X_valid.columns,
#     "MeanAbsSHAP": mean_abs_shap
# }).sort_values("MeanAbsSHAP", ascending=False)

# print("\n🔍 SHAP-like Feature Contribution:")
# print(shap_df)

# plt.figure(figsize=(8, 5))
# plt.barh(shap_df["Feature"], shap_df["MeanAbsSHAP"])
# plt.title("Feature Impact (CatBoost SHAP)")
# plt.xlabel("Mean |SHAP value|")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "catboost_shap_bar.png"))
# plt.show()

# print(f"\n✅ All outputs saved in: {OUTPUT_DIR}")


# ---------------------------lighgbm--r2 0.52
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from lightgbm import LGBMRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score
# # import shap

# # ==========================================
# # ⚙️ Config
# # ==========================================
# LOG_TRANSFORM = True
# DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS.xlsx"
# OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ==========================================
# # 📂 Load & Clean Data
# # ==========================================
# df = pd.read_excel(DATA_PATH)
# df = df.replace(["-", " ", ""], pd.NA)

# numeric_cols = [
#     "Arm N",
#     "Objective Response Rate N",
#     "Objective Response Rate Percentage",
#     "Duration of Response Median",
#     "Median PFS"
# ]
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# df["Date of Publication"] = pd.to_datetime(df["Date of Publication"], errors="coerce")
# df["Publication Year"] = df["Date of Publication"].dt.year
# df = df.dropna(subset=["Median PFS"])

# # ==========================================
# # 🧩 Features & Target
# # ==========================================
# target = "Median PFS"
# features = [
#     "Type",
#     "Product",
#     "Dosage",
#     "Arm N",
#     "Objective Response Rate N",
#     "Objective Response Rate Percentage",
#     "Duration of Response Median",
#     "Publication Year"
# ]

# X = df[features]
# y = df[target]
# if LOG_TRANSFORM:
#     y = np.log1p(y)

# # convert categoricals to category dtype (LightGBM handles that)
# for col in X.select_dtypes(include="object").columns:
#     X[col] = X[col].astype("category")

# # ==========================================
# # ✂️ Split
# # ==========================================
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ==========================================
# # 🧠 Train LightGBM
# # ==========================================
# model = LGBMRegressor(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=6,
#     random_state=42,
# )
# model.fit(X_train, y_train,
#           eval_set=[(X_valid, y_valid)],
#           eval_metric="l2")

# # ==========================================
# # 📈 Evaluate
# # ==========================================
# y_pred = model.predict(X_valid)
# if LOG_TRANSFORM:
#     y_pred = np.expm1(y_pred)
#     y_valid = np.expm1(y_valid)

# mae = mean_absolute_error(y_valid, y_pred)
# r2 = r2_score(y_valid, y_pred)
# print(f"\n📊 Model Evaluation:")
# print(f"Mean Absolute Error: {mae:.3f}")
# print(f"R² Score: {r2:.3f}")

# # ==========================================
# # 💾 Save model
# # ==========================================
# model.booster_.save_model(os.path.join(OUTPUT_DIR, "model_lgbm.txt"))
# print(f"✅ Model saved to: {OUTPUT_DIR}\\model_lgbm.txt")

# # ==========================================
# # 🔍 Feature Importance
# # ==========================================
# importances = model.booster_.feature_importance(importance_type="gain")
# feat_imp = pd.DataFrame({
#     "Feature": X.columns,
#     "Importance": importances
# }).sort_values("Importance", ascending=False)

# plt.figure(figsize=(8, 5))
# plt.barh(feat_imp["Feature"], feat_imp["Importance"])
# plt.title("LightGBM Feature Importance (gain)")
# plt.xlabel("Importance")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
# plt.show()

# # ==========================================
# # # 💡 SHAP Feature Contributions
# # # ==========================================
# # print("\n⚡ Calculating SHAP values...")
# # explainer = shap.Explainer(model, X_train)
# # shap_values = explainer(X_valid)

# # # Summary plot (bar)
# # shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
# # plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar.png"), bbox_inches="tight")

# # # Beeswarm plot
# # shap.summary_plot(shap_values, X_valid, show=False)
# # plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), bbox_inches="tight")

# print(f"\n✅ All outputs saved in: {OUTPUT_DIR}")

# # +++++++++++++++++++++++++++++++correct code clean

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import shap
# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.preprocessing import LabelEncoder
# import joblib

# pd.options.mode.chained_assignment = None  # disable pandas SettingWithCopyWarning

# # ==========================================
# # ⚙️ Config
# # ==========================================
# LOG_TRANSFORM = False
# DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_precise_moa.xlsx"
# OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\data"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ==========================================
# # 📂 Load & Clean Data
# # ==========================================
# print("📂 Loading dataset...")
# df = pd.read_excel(DATA_PATH).replace(["-", " ", ""], pd.NA)
# print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# # ==========================================
# # 🧹 Clean Product Column (remove company names, slashes, etc.)
# # ==========================================
# import re

# def clean_product(text):
#     if pd.isna(text):
#         return text

#     # Lowercase for consistency
#     text = text.lower()

#     # Replace separators with commas
#     text = (
#         text.replace("/", ",")
#             .replace("+", ",")
#             .replace(";", ",")
#             .replace(" and ", ",")
#     )

#     # Remove known company/manufacturer names
#     companies = [
#         "bms", "merck", "msd", "astrazeneca", "eli lilly", "roche", "generic mfg",
#         "ono pharma", "savara", "novartis", "pfizer", "amgen", "abbvie", "bayer","roche", "merck", "msd", "bms", "ono", "pharma", "astrazeneca",
#     "generic", "eli", "lilly", "pfizer", "janssen", "jnj", "otsuka",
#     "biogen", "astellas", "bayer", "abbvie", "novartis", "sanofi",
#     "bristol", "boehringer", "amgen", "sirtex", "celldex", "menarini",
#     "nordic", "bausch", "health", "medical", "mfg", "manufacturer", "company","neotx","novartis","folfoxye","tegafur","group",'gsk','servier','novo','beigene','incyte','regeneron','gilead','takeda',
#     # Big Pharma & common entities
#     "roche","genentech","merck","msd","bms","bristol","myers","squibb","ono","pharma","pharmaceutical",
#     "astrazeneca","az","pfizer","eli","lilly","janssen","jnj","johnson","otsuka","biogen","astellas",
#     "bayer","abbvie","novartis","sanofi","boehringer","ingelheim","amgen","takeda","servier","gsk",
#     "glaxosmithkline","novo","nordisk","bei","beigene","regeneron","incyte","gilead","menarini","sirtex",
#     "celldex","nordic","teva","sun","sunpharma","bausch","health","sri","international","neotx",
#     # Generics, contract & biotech terms
#     "generic","mfg","manufacturer","biopharma","biosciences","labs","laboratories","therapeutics","biotech",
#     "pharmaceutics","pharmaceuticals","bio","corp","inc","limited","ltd","sa","plc","company","partner",
#     "co","medical","research","technologies","science","gmbh",
#     # Asian / regional / collaborations
#     "luye","chugai","kyowa","daiichi","sumitomo","asahi","taiho","hanmi","biocon","cipla","dr","reddy",
#     "intas","zydus","torrent","hansoh","hisun","simcere","beijing","shionogi","lupin","sandoz",
#     # Trial text & misc words
#     "group","collaborator","partnered","sponsor","partnering","tegafur","folfoxye","high","low","dose",
#     "for","the","is"
# ]
    
#     for c in companies:
#         text = re.sub(rf"\b{c}\b", "", text)

#     # Remove placebo and control arms
#     text = re.sub(r"\bplacebo\b|\bstandard of care\b", "", text)

#     # Clean extra commas/spaces
#     text = re.sub(r",+", ",", text)
#     text = re.sub(r"\s+", " ", text)
#     text = text.strip().strip(",")

#     return text

# if "Product" in df.columns:
#     df["Product"] = df["Product"].apply(clean_product)
#     print("\n✅ Cleaned Product column examples:")
#     print(df["Product"].head(20))

# numeric_cols = [
#     "Arm N", "Objective Response Rate N", "Objective Response Rate Percentage",
#     "Duration of Response Median", "Median PFS"
# ]
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# # df["Date of Publication"] = pd.to_datetime(df["Date of Publication"], errors="coerce")
# # df["Publication Year"] = df["Date of Publication"].dt.year

# # ==========================================
# # 🕒 Handle Date of Publication → Extract Year
# # ==========================================
# if "Date of Publication" in df.columns:
#     df["Date of Publication"] = pd.to_datetime(df["Date of Publication"], errors="coerce")
#     df["Publication Year"] = df["Date of Publication"].dt.year
#     df = df.drop(columns=["Date of Publication"])
#     print("✅ Extracted 'Publication Year' and removed 'Date of Publication' column.")

# # ==========================================
# # 🧮 Feature Engineering
# # ==========================================
# df["Response_Count_Duration"] = (
#     df["Objective Response Rate N"] * df["Duration of Response Median"]
# )
# df["Response_Percentage_Duration"] = (
#     (df["Objective Response Rate Percentage"] / 100) * df["Duration of Response Median"]
# )

# numeric_cols += ["Response_Count_Duration", "Response_Percentage_Duration"]
# df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# for col in ["Type", "Product", "Dosage"]:
#     if col in df.columns:
#         df[col] = df[col].fillna("Unknown")

# # Drop rows without target
# df = df.dropna(subset=["Median PFS"])

# # # ==========================================
# # # 🚫 Remove Outliers from Target (Median PFS > 50)
# # # # ==========================================
# # initial_rows = df.shape[0]
# # df = df[df["Median PFS"] <= 50]
# # removed_rows = initial_rows - df.shape[0]

# # print(f"✅ Removed {removed_rows} rows where Median PFS > 50 (kept {df.shape[0]} rows).")



# # Q1 = df["Median PFS"].quantile(0.25)
# # Q3 = df["Median PFS"].quantile(0.75)
# # IQR = Q3 - Q1
# # upper_limit = Q3 + 1.5 * IQR
# # lower_limit = Q1 - 1.5 * IQR

# # print("\n🔎 IQR-based Outlier Analysis:")
# # print(f"  • Q1 (25th percentile): {Q1:.2f}")
# # print(f"  • Q3 (75th percentile): {Q3:.2f}")
# # print(f"  • IQR (Q3 - Q1): {IQR:.2f}")
# # print(f"  • Upper limit (Q3 + 1.5×IQR): {upper_limit:.2f}")
# # print(f"  • Lower limit (Q1 - 1.5×IQR): {lower_limit:.2f}")

# # # Optional: show how many points lie beyond cutoff
# # n_outliers = (df["Median PFS"] > upper_limit).sum()
# # print(f"  ⚠️ Trials with Median PFS above {upper_limit:.1f}: {n_outliers} ({(n_outliers/len(df)*100):.1f}%)")

# # ==========================================
# # 💾 Save Cleaned Data
# # ==========================================
# cleaned_data_path = os.path.join(OUTPUT_DIR, "MedianPFS_cleaned.xlsx")
# df.to_excel(cleaned_data_path, index=False)
# print(f"✅ Cleaned dataset saved to: {cleaned_data_path}")

# # ==========================================
# # 📈 Correlation Heatmap
# # ==========================================
# plt.figure(figsize=(10, 8))
# sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap (Numeric Features)")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_PFS.png"))
# plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 0.659
import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# ⚙️ CONFIG
# ============================================================
DATA_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_precise_moa.xlsx"
OUTPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned1.xlsx"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ============================================================
# 📌 1. LOAD DATA (WITH MOA ALREADY ADDED)
# ============================================================
print("📂 Loading MOA-mapped PFS dataset...")
df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()

# Handle NA replacements
df = df.replace(["-", "--", "—", "", " ", "NA", "N/A", "nan"], np.nan)
print(f"✔ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# 📌 2. CLEAN PRODUCT COLUMN
# ============================================================
company_words = [
    "bms","merck","msd","astrazeneca","eli","lilly","roche","pfizer","amgen",
    "abbvie","bayer","janssen","sanofi","novartis","gsk","regeneron","incyte",
    "gilead","takeda","bei","beigene","servier","novo","nordisk","pharma",
    "pharmaceutical","biotech","therapeutics","labs","inc","corp","company",
    "generic","manufacturing","partner","group"
    # Big Pharma & common entities
    "roche","genentech","merck","msd","bms","bristol","myers","squibb","ono","pharma","pharmaceutical",
    "astrazeneca","az","pfizer","eli","lilly","janssen","jnj","johnson","otsuka","biogen","astellas",
    "bayer","abbvie","novartis","sanofi","boehringer","ingelheim","amgen","takeda","servier","gsk",
    "glaxosmithkline","novo","nordisk","bei","beigene","regeneron","incyte","gilead","menarini","sirtex",
    "celldex","nordic","teva","sun","sunpharma","bausch","health","sri","international","neotx",
    # Generics, contract & biotech terms
    "generic","mfg","manufacturer","biopharma","biosciences","labs","laboratories","therapeutics","biotech",
    "pharmaceutics","pharmaceuticals","bio","corp","inc","limited","ltd","sa","plc","company","partner",
    "co","medical","research","technologies","science","gmbh",
    # Asian / regional / collaborations
    "luye","chugai","kyowa","daiichi","sumitomo","asahi","taiho","hanmi","biocon","cipla","dr","reddy",
    "intas","zydus","torrent","hansoh","hisun","simcere","beijing","shionogi","lupin","sandoz",
    # Trial text & misc words
    "group","collaborator","partnered","sponsor","partnering","tegafur","folfoxye","high","low","dose",
    "for","the","is"
]

def clean_product(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()

    # Unify delimiters
    text = re.sub(r"[+/;]", ",", text)
    text = text.replace(" and ", ",")
    text = re.sub(r",+", ",", text)

    # Remove company terms
    for w in company_words:
        text = re.sub(rf"\b{w}\b", "", text)

    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,")
    return text or ""

df["Product"] = df["Product"].apply(clean_product)

# ============================================================
# 📌 3. NUMERIC COLUMNS
# ============================================================
numeric_cols = [
    "Arm N", "Objective Response Rate N", "Objective Response Rate Percentage",
    "Duration of Response Median", "Median PFS"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Median PFS"])

# ============================================================
# 📌 4. PUBLICATION YEAR (IF PRESENT)
# ============================================================
if "Date of Publication" in df.columns:
    df["Date of Publication"] = pd.to_datetime(df["Date of Publication"], errors="coerce")
    df["Publication Year"] = df["Date of Publication"].dt.year
    df.drop(columns=["Date of Publication"], inplace=True)

# ============================================================
# 📌 5. RESPONSE FEATURES
# ============================================================
df["Response_Count_Duration"] = np.log1p(
    df["Objective Response Rate N"] * df["Duration of Response Median"]
)

df["Response_Percentage_Duration"] = np.log1p(
    (df["Objective Response Rate Percentage"] / 100) * df["Duration of Response Median"]
)

# Fill numeric NA
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


# ============================================================
# 📌 6. CLEAN CATEGORICAL COLUMNS
# ============================================================
cat_cols = df.select_dtypes(include="object").columns

def clean_cat(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text or ""

for col in cat_cols:
    df[col] = df[col].apply(clean_cat)

# ============================================================
# 📌 8. SAVE CLEAN FINAL FILE
# ============================================================
df.to_excel(OUTPUT_PATH, index=False)
print(f"🎉 CLEAN DATA SAVED → {OUTPUT_PATH}")


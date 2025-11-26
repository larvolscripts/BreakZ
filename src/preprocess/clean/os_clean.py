
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

pd.options.mode.chained_assignment = None  # disable pandas SettingWithCopyWarning

# ==========================================
# ⚙️ Config
# ==========================================
LOG_TRANSFORM = False
DATA_PATH=r"C:\LARVOL_WORK\Median_PFS\data\MedianOS.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 📂 Load & Clean Data
# ==========================================
print("📂 Loading dataset...")
df = pd.read_excel(DATA_PATH).replace(["-", " ", ""], pd.NA)
print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ==========================================
# 🧹 Clean Product Column (remove company names, slashes, etc.)
# ==========================================
import re

def clean_product(text):
    if pd.isna(text):
        return text

    # Lowercase for consistency
    text = text.lower()

    # Replace separators with commas
    text = (
        text.replace("/", ",")
            .replace("+", ",")
            .replace(";", ",")
            .replace(" and ", ",")
    )

    # Remove known company/manufacturer names
    companies = [
        "bms", "merck", "msd", "astrazeneca", "eli lilly", "roche", "generic mfg",
        "ono pharma", "savara", "novartis", "pfizer", "amgen", "abbvie", "bayer","roche", "merck", "msd", "bms", "ono", "pharma", "astrazeneca",
    "generic", "eli", "lilly", "pfizer", "janssen", "jnj", "otsuka",
    "biogen", "astellas", "bayer", "abbvie", "novartis", "sanofi",
    "bristol", "boehringer", "amgen", "sirtex", "celldex", "menarini",
    "nordic", "bausch", "health", "medical", "mfg", "manufacturer", "company","neotx","novartis","folfoxye","tegafur","group",'gsk','servier','novo','beigene','incyte','regeneron','gilead','takeda'
]
    
    for c in companies:
        text = re.sub(rf"\b{c}\b", "", text)

    # Remove placebo and control arms
    text = re.sub(r"\bplacebo\b|\bstandard of care\b", "", text)

    # Clean extra commas/spaces
    text = re.sub(r",+", ",", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().strip(",")

    return text

if "Product" in df.columns:
    df["Product"] = df["Product"].apply(clean_product)
    print("\n✅ Cleaned Product column examples:")
    print(df["Product"].head(20))

numeric_cols = [
    "Arm N", "Objective Response Rate N", "Objective Response Rate Percentage",
    "Duration of Response Median", "Median OS"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================
# 🕒 Handle Date of Publication → Extract Year
# ==========================================
if "Date of Publication" in df.columns:
    df["Date of Publication"] = pd.to_datetime(df["Date of Publication"], errors="coerce")
    df["Publication Year"] = df["Date of Publication"].dt.year
    df = df.drop(columns=["Date of Publication"])
    print("✅ Extracted 'Publication Year' and removed 'Date of Publication' column.")

# ==========================================
# 🧮 Feature Engineering
# ==========================================
df["Response_Count_Duration"] = (
    df["Objective Response Rate N"] * df["Duration of Response Median"]
)
df["Response_Percentage_Duration"] = (
    (df["Objective Response Rate Percentage"] / 100) * df["Duration of Response Median"]
)

numeric_cols += ["Response_Count_Duration", "Response_Percentage_Duration"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

for col in ["Type", "Product", "Dosage"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Drop rows without target
df = df.dropna(subset=["Median OS"])

# ==========================================
# 🚫 Remove Outliers from Target (Median PFS > 50)
# ==========================================
# initial_rows = df.shape[0]
# df = df[df["Median PFS"] <= 50]
# removed_rows = initial_rows - df.shape[0]

# print(f"✅ Removed {removed_rows} rows where Median PFS > 50 (kept {df.shape[0]} rows).")

# ==========================================
# 💾 Save Cleaned Data
# ==========================================
cleaned_data_path = os.path.join(OUTPUT_DIR, "MedianOS_cleaned.xlsx")
df.to_excel(cleaned_data_path, index=False)
print(f"✅ Cleaned dataset saved to: {cleaned_data_path}")

# ==========================================
# 📈 Correlation Heatmap
# ==========================================
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()
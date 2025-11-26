# ============================================================
# 📈 SCATTER PLOTS: Median OS
# Description: ORR vs OS, DoR vs OS, Product vs OS
# ============================================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# -------------------------------
# ⚙️ Config
# -------------------------------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianOS_cleaned.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\scatter_OS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 📂 Load Dataset
# -------------------------------
df = pd.read_excel(INPUT_PATH)
df = df[df["Median OS"].notna() & (df["Median OS"] > 0)]

# -------------------------------
# 🎨 Plot Style
# -------------------------------
sns.set_theme(style="whitegrid", palette="muted")

# Helper: Safe save
def savefig_safe(path):
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()

# -------------------------------
# 1️⃣ ORR vs Median OS
# -------------------------------
if "Objective Response Rate Percentage" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.regplot(
        data=df,
        x="Objective Response Rate Percentage",
        y="Median OS",
        scatter_kws={"alpha": 0.7, "color": "purple"},
        line_kws={"color": "red"},
    )
    plt.title("Objective Response Rate (%) vs Median OS", fontsize=13)
    plt.xlabel("Objective Response Rate (%)")
    plt.ylabel("Median OS (months)")
    savefig_safe(os.path.join(OUTPUT_DIR, "1_ORR_vs_MedianOS.png"))

# -------------------------------
# 2️⃣ DoR vs Median OS
# -------------------------------
if "Duration of Response Median" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.regplot(
        data=df,
        x="Duration of Response Median",
        y="Median OS",
        scatter_kws={"alpha": 0.7, "color": "darkorange"},
        line_kws={"color": "red"},
    )
    plt.title("Duration of Response vs Median OS", fontsize=13)
    plt.xlabel("Duration of Response (months)")
    plt.ylabel("Median OS (months)")
    savefig_safe(os.path.join(OUTPUT_DIR, "2_DoR_vs_MedianOS.png"))

# -------------------------------
# 3️⃣ Product vs Median OS
# -------------------------------
if "Product" in df.columns:
    top_products = df["Product"].value_counts().head(15).index
    subset = df[df["Product"].isin(top_products)]

    plt.figure(figsize=(14, 6))
    ax = sns.stripplot(
        data=subset,
        x="Product",
        y="Median OS",
        jitter=True,
        alpha=0.7,
        color="mediumvioletred"
    )
    labels = [textwrap.fill(label.get_text(), 12) for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    plt.title("Median OS by Top 15 Products", fontsize=13)
    plt.ylabel("Median OS (months)")
    plt.xlabel("Product")
    savefig_safe(os.path.join(OUTPUT_DIR, "3_Product_vs_MedianOS.png"))

print(f"✅ All Median OS scatter plots saved to: {OUTPUT_DIR}")

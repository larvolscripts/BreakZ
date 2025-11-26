# ============================================================
# 📈 SCATTER PLOTS: Median PFS
# Description: ORR vs PFS, DoR vs PFS, Product vs PFS
# ============================================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# -------------------------------
# ⚙️ Config
# -------------------------------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\scatter_PFS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 📂 Load Dataset
# -------------------------------
df = pd.read_excel(INPUT_PATH)
df = df[df["Median PFS"].notna() & (df["Median PFS"] > 0)]

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
# 1️⃣ ORR vs Median PFS
# -------------------------------
if "Objective Response Rate Percentage" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.regplot(
        data=df,
        x="Objective Response Rate Percentage",
        y="Median PFS",
        scatter_kws={"alpha": 0.7, "color": "teal"},
        line_kws={"color": "red"},
    )
    plt.title("Objective Response Rate (%) vs Median PFS", fontsize=13)
    plt.xlabel("Objective Response Rate (%)")
    plt.ylabel("Median PFS (months)")
    savefig_safe(os.path.join(OUTPUT_DIR, "1_ORR_vs_MedianPFS.png"))

# -------------------------------
# 2️⃣ DoR vs Median PFS
# -------------------------------
if "Duration of Response Median" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.regplot(
        data=df,
        x="Duration of Response Median",
        y="Median PFS",
        scatter_kws={"alpha": 0.7, "color": "seagreen"},
        line_kws={"color": "red"},
    )
    plt.title("Duration of Response vs Median PFS", fontsize=13)
    plt.xlabel("Duration of Response (months)")
    plt.ylabel("Median PFS (months)")
    savefig_safe(os.path.join(OUTPUT_DIR, "2_DoR_vs_MedianPFS.png"))

# -------------------------------
# 3️⃣ Product vs Median PFS
# -------------------------------
if "Product" in df.columns:
    top_products = df["Product"].value_counts().head(15).index
    subset = df[df["Product"].isin(top_products)]

    plt.figure(figsize=(14, 6))
    ax = sns.stripplot(
        data=subset,
        x="Product",
        y="Median PFS",
        jitter=True,
        alpha=0.7,
        color="royalblue"
    )
    labels = [textwrap.fill(label.get_text(), 12) for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    plt.title("Median PFS by Top 15 Products", fontsize=13)
    plt.ylabel("Median PFS (months)")
    plt.xlabel("Product")
    savefig_safe(os.path.join(OUTPUT_DIR, "3_Product_vs_MedianPFS.png"))

print(f"✅ All Median PFS scatter plots saved to: {OUTPUT_DIR}")

# ============================================================
# 📈 MEDIAN OS CATPLOT DASHBOARD (Readable Version)
# Description: Explore how Median OS varies across key features
# ============================================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import textwrap

# -------------------------------
# ⚙️ Config
# -------------------------------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianOS_cleaned.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\catplots_OS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 📂 Load Dataset
# -------------------------------
df = pd.read_excel(INPUT_PATH)
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------
# 🧹 Clean Values
# -------------------------------
df = df[df["Median OS"].notna()]
df = df[df["Median OS"] > 0]

# -------------------------------
# 🎨 Seaborn Theme
# -------------------------------
sns.set_theme(style="whitegrid", palette="Set2")

# -------------------------------
# 🧩 Simplify Arm Type (Experimental, Control, etc.)
# -------------------------------
def simplify_type(text):
    text = str(text).lower()
    if "experimental" in text and "control" in text:
        return "Mixed Arm"
    elif "experimental" in text:
        return "Experimental"
    elif "active" in text:
        return "Active Comparator"
    elif "placebo" in text:
        return "Placebo Comparator"
    elif "control" in text:
        return "Control"
    else:
        return "Other"

if "Type" in df.columns:
    df["Type_Simplified"] = df["Type"].apply(simplify_type)

# -------------------------------
# Helper: Wrap Long Labels
# -------------------------------
def wrap_labels(ax, width=12):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width))
    ax.set_xticklabels(labels, rotation=30, ha="right")

# -------------------------------
# 1️⃣ Median OS by Arm Type
# -------------------------------
if "Type_Simplified" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(
        data=df, x="Type_Simplified", y="Median OS",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"}
    )
    plt.title("Median OS by Arm Type (Simplified)", fontsize=13)
    plt.ylabel("Median OS (months)")
    plt.xlabel("Arm Type")
    wrap_labels(plt.gca(), width=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_os_by_type_simplified.png"), bbox_inches="tight")
    plt.close()

# Horizontal version (better readability)
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df, y="Type_Simplified", x="Median OS", orient="h",
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"}
)
plt.title("Median OS by Arm Type (Horizontal View)", fontsize=13)
plt.xlabel("Median OS (months)")
plt.ylabel("Arm Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1b_os_by_type_horizontal.png"), bbox_inches="tight")
plt.close()

# -------------------------------
# 2️⃣ Median OS by Product (Top 15)
# -------------------------------# -------------------------------
# 2️⃣ Median OS by Product (Top 15, Better Readable)# -------------------------------
# 2️⃣ Median OS by Product (Top 15, Clean + Wrapped + Large Boxes)
# -------------------------------
import textwrap
from matplotlib.ticker import FixedLocator

if "Product" in df.columns:
    top_products = df["Product"].value_counts().head(15).index
    subset = df[df["Product"].isin(top_products)]
    order = subset.groupby("Product")["Median OS"].median().sort_values(ascending=False).index

    plt.figure(figsize=(18, 8))  # 🔹 wider figure
    ax = sns.boxplot(
        data=subset,
        x="Product", y="Median OS",
        order=order,
        width=0.6,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
    )

    # ✅ Wrap long product names
    labels = [textwrap.fill(label.get_text(), 15) for label in ax.get_xticklabels()]
    ax.xaxis.set_major_locator(FixedLocator(range(len(labels))))
    ax.set_xticklabels(labels, rotation=0, ha="center")

    plt.title("Median OS by Top 15 Products", fontsize=14)
    plt.ylabel("Median OS (months)")
    plt.xlabel("Product")
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # gives extra bottom margin
    plt.savefig(os.path.join(OUTPUT_DIR, "2_os_by_top_products.png"), bbox_inches="tight")
    plt.close()

# -------------------------------
# 3️⃣ Median OS by Dosage (Top 20)
# -------------------------------
# -------------------------------
# 3️⃣ Median OS by Dosage (Readable, Wrapped, Warning-Free)
# -------------------------------
import textwrap
from matplotlib.ticker import FixedLocator

if "Dosage" in df.columns:
    # Use top N dosages (avoid overcrowding)
    top_dosages = df["Dosage"].value_counts().head(20).index
    subset = df[df["Dosage"].isin(top_dosages)]

    order = (
        subset.groupby("Dosage")["Median OS"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    plt.figure(figsize=(18, 8))  # ✅ Bigger canvas
    ax = sns.boxplot(
        data=subset,
        x="Dosage",
        y="Median OS",
        order=order,
        width=0.6,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "red",
            "markeredgecolor": "black",
        },
    )

    # ✅ Wrap long dosage labels for readability
    labels = [textwrap.fill(label.get_text(), 12) for label in ax.get_xticklabels()]
    ax.xaxis.set_major_locator(FixedLocator(range(len(labels))))
    ax.set_xticklabels(labels, rotation=0, ha="center")

    plt.title("Median OS by Top 20 Dosages", fontsize=14)
    plt.ylabel("Median OS (months)")
    plt.xlabel("Dosage")
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # extra bottom margin
    plt.savefig(
        os.path.join(OUTPUT_DIR, "3_os_by_top_dosages_readable.png"),
        bbox_inches="tight",
    )
    plt.close()

# -------------------------------
# 4️⃣ Trend Over Publication Year
# -------------------------------
if "Publication Year" in df.columns:
    g = sns.catplot(
        data=df, x="Publication Year", y="Median OS",
        kind="box", height=5, aspect=1.6
    )
    sns.lineplot(
        data=df.groupby("Publication Year")["Median OS"].mean().reset_index(),
        x="Publication Year", y="Median OS", color="red", marker="o", label="Mean Trend"
    )
    plt.title("Median OS by Publication Year", fontsize=13)
    plt.legend()
    plt.tight_layout()
    g.figure.savefig(os.path.join(OUTPUT_DIR, "4_os_by_year.png"), bbox_inches="tight")
    plt.close(g.figure)

# -------------------------------
# 5️⃣ Objective Response Rate (%) vs Median OS
# -------------------------------
if "Objective Response Rate Percentage" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.regplot(
        data=df, x="Objective Response Rate Percentage", y="Median OS",
        scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}
    )
    plt.title("Objective Response Rate (%) vs Median OS", fontsize=13)
    plt.xlabel("Objective Response Rate (%)")
    plt.ylabel("Median OS (months)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_os_vs_response_rate.png"), bbox_inches="tight")
    plt.close()

# -------------------------------
# 6️⃣ Duration of Response vs Median OS
# -------------------------------
if "Duration of Response Median" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.regplot(
        data=df, x="Duration of Response Median", y="Median OS",
        scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}
    )
    plt.title("Duration of Response vs Median OS", fontsize=13)
    plt.xlabel("Duration of Response (months)")
    plt.ylabel("Median OS (months)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_os_vs_duration.png"), bbox_inches="tight")
    plt.close()

# -------------------------------
# 7️⃣ Correlation Heatmap
# -------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "7_corr_heatmap.png"), bbox_inches="tight")
    plt.close()


# -------------------------------
# 8️⃣ ORR × DoR vs Median OS
# -------------------------------
if "Objective Response Rate Percentage" in df.columns and "Duration of Response Median" in df.columns:
    df["ORR_DoR"] = df["Objective Response Rate Percentage"] * df["Duration of Response Median"]

    plt.figure(figsize=(7, 5))
    sns.regplot(
        data=df,
        x="ORR_DoR",
        y="Median OS",
        scatter_kws={"alpha": 0.6, "color": "steelblue"},
        line_kws={"color": "red"}
    )
    plt.title("ORR × DoR vs Median OS", fontsize=13)
    plt.xlabel("Objective Response Rate × Duration of Response")
    plt.ylabel("Median OS (months)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "8_orrxdor_vs_median_os.png"), bbox_inches="tight")
    plt.close()


print(f"✅ All Median OS catplots and visuals saved to: {OUTPUT_DIR}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Paths
DATA_PFS = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned.xlsx"
DATA_OS = r"C:\LARVOL_WORK\Median_PFS\data\MedianOS_cleaned.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs\log_scatterplots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
pfs = pd.read_excel(DATA_PFS)
os_ = pd.read_excel(DATA_OS)

# Compute ORR×DoR and log(ORR×DoR)
for df in [pfs, os_]:
    df["ORR×DoR"] = (
        df["Objective Response Rate Percentage"]
        * df["Duration of Response Median"]
    )
    df["log_ORR×DoR"] = np.log1p(df["ORR×DoR"])  # log(1+x) to avoid log(0)

# Plot function
def plot_log_scatter(df, x, y, color, title, file_name):
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=df, x=x, y=y,
        scatter_kws={"alpha": 0.6, "color": color},
        line_kws={"color": "red", "lw": 2}
    )
    plt.title(title, fontsize=14)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, file_name), bbox_inches="tight", dpi=300)
    plt.close()

# 1️⃣ log(ORR×DoR) vs Median PFS
plot_log_scatter(
    pfs, "log_ORR×DoR", "Median PFS",
    "#1f77b4", "log(ORR×DoR) vs Median PFS", "log_ORR_DoR_vs_MedianPFS.png"
)

# 2️⃣ log(ORR×DoR) vs Median OS
plot_log_scatter(
    os_, "log_ORR×DoR", "Median OS",
    "#ff7f0e", "log(ORR×DoR) vs Median OS", "log_ORR_DoR_vs_MedianOS.png"
)

# Optional – raw versions (for comparison)
plot_log_scatter(
    pfs, "ORR×DoR", "Median PFS",
    "#2ca02c", "ORR×DoR vs Median PFS", "ORR_DoR_vs_MedianPFS.png"
)
plot_log_scatter(
    os_, "ORR×DoR", "Median OS",
    "#9467bd", "ORR×DoR vs Median OS", "ORR_DoR_vs_MedianOS.png"
)

print(f"✅ All scatter plots saved in: {OUTPUT_DIR}")

# make_scatter_images_safe.py
import os, textwrap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

sns.set_theme(style="whitegrid", palette="muted")
def save_plt_safe(path):
    plt.tight_layout(rect=[0,0,1,1])
    plt.savefig(path, bbox_inches="tight", dpi=300, pad_inches=0.2)
    plt.close()

# CONFIG: update paths
PFS_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned.xlsx"
OS_PATH  = r"C:\LARVOL_WORK\Median_PFS\data\MedianOS_cleaned.xlsx"
OUT_DIR_PFS = r"C:\LARVOL_WORK\Median_PFS\outputs\scatter1_PFS"
OUT_DIR_OS  = r"C:\LARVOL_WORK\Median_PFS\outputs\scatter1_OS"
os.makedirs(OUT_DIR_PFS, exist_ok=True)
os.makedirs(OUT_DIR_OS, exist_ok=True)

# HELPER for product stripplot (wrapped labels)
def product_stripplot(df, y_col, out_path, top_n=15):
    top = df["Product"].value_counts().head(top_n).index
    sub = df[df["Product"].isin(top)]
    # prefer horizontal boxplot for better readability:
    order = sub.groupby("Product")[y_col].median().sort_values(ascending=False).index
    plt.figure(figsize=(12, max(6, len(order)*0.35)))
    ax = sns.boxplot(data=sub, y="Product", x=y_col, order=order, orient="h", width=0.6, showmeans=True,
                     meanprops={"marker":"o","markerfacecolor":"red","markeredgecolor":"black"})
    # wrap label text automatically on y-axis if too long
    labels = [textwrap.fill(l.get_text(), 35) for l in ax.get_yticklabels()]
    ax.set_yticklabels(labels, fontsize=9)
    plt.xlabel(y_col + " (months)")
    plt.title(f"{y_col} by Top {top_n} Products")
    save_plt_safe(out_path)

# ---------- PFS ----------
df_pfs = pd.read_excel(PFS_PATH)
df_pfs = df_pfs[df_pfs["Median PFS"].notna() & (df_pfs["Median PFS"]>0)]

# ORR vs PFS
if "Objective Response Rate Percentage" in df_pfs.columns:
    plt.figure(figsize=(7,5))
    sns.regplot(data=df_pfs, x="Objective Response Rate Percentage", y="Median PFS",
                scatter_kws={"alpha":0.7}, line_kws={"color":"red"})
    plt.xlabel("Objective Response Rate (%)")
    plt.ylabel("Median PFS (months)")
    plt.title("ORR vs Median PFS")
    save_plt_safe(os.path.join(OUT_DIR_PFS, "ORR_vs_MedianPFS.png"))

# DoR vs PFS
if "Duration of Response Median" in df_pfs.columns:
    plt.figure(figsize=(7,5))
    sns.regplot(data=df_pfs, x="Duration of Response Median", y="Median PFS",
                scatter_kws={"alpha":0.7}, line_kws={"color":"red"})
    plt.xlabel("Duration of Response (months)")
    plt.ylabel("Median PFS (months)")
    plt.title("DoR vs Median PFS")
    save_plt_safe(os.path.join(OUT_DIR_PFS, "DoR_vs_MedianPFS.png"))

# Product vs PFS (horizontal)
if "Product" in df_pfs.columns:
    product_stripplot(df_pfs, "Median PFS", os.path.join(OUT_DIR_PFS, "Product_vs_MedianPFS.png"), top_n=15)

# ---------- OS ----------
df_os = pd.read_excel(OS_PATH)
df_os = df_os[df_os["Median OS"].notna() & (df_os["Median OS"]>0)]

# ORR vs OS
if "Objective Response Rate Percentage" in df_os.columns:
    plt.figure(figsize=(7,5))
    sns.regplot(data=df_os, x="Objective Response Rate Percentage", y="Median OS",
                scatter_kws={"alpha":0.7}, line_kws={"color":"red"})
    plt.xlabel("Objective Response Rate (%)")
    plt.ylabel("Median OS (months)")
    plt.title("ORR vs Median OS")
    save_plt_safe(os.path.join(OUT_DIR_OS, "ORR_vs_MedianOS.png"))

# DoR vs OS
if "Duration of Response Median" in df_os.columns:
    plt.figure(figsize=(7,5))
    sns.regplot(data=df_os, x="Duration of Response Median", y="Median OS",
                scatter_kws={"alpha":0.7}, line_kws={"color":"red"})
    plt.xlabel("Duration of Response (months)")
    plt.ylabel("Median OS (months)")
    plt.title("DoR vs Median OS")
    save_plt_safe(os.path.join(OUT_DIR_OS, "DoR_vs_MedianOS.png"))

# Product vs OS (horizontal)
if "Product" in df_os.columns:
    product_stripplot(df_os, "Median OS", os.path.join(OUT_DIR_OS, "Product_vs_MedianOS.png"), top_n=15)

print("✅ Scatter images (PFS & OS) generated in:")
print(OUT_DIR_PFS)
print(OUT_DIR_OS)

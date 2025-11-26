# check_and_align_embeddings_excel.py
import pandas as pd

# -------------------------------
# 📂 INPUT FILE PATHS
# -------------------------------
BASE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
DISEASE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3_with_disease_emb.parquet"
DRUG = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3_with_drug_emb.parquet"

# -------------------------------
# 📂 OUTPUT PATHS (parquet + Excel)
# -------------------------------
OUT_PARQUET = r"C:\LARVOL_WORK\Median_PFS\data\final_emb_AplusB.parquet"
OUT_EXCEL   = r"C:\LARVOL_WORK\Median_PFS\data\final_emb_AplusB.xlsx"

KEYS = ["Trial ID", "Arm ID"]


print("Loading files...")
df_base = pd.read_excel(BASE)
df_dis  = pd.read_parquet(DISEASE)
df_drug = pd.read_parquet(DRUG)


# ===========================================================
# 1️⃣ SHAPE CHECK
# ===========================================================
print("\n=== SHAPE CHECK ===")
print("Base   :", df_base.shape)
print("Disease:", df_dis.shape)
print("Drug   :", df_drug.shape)


# ===========================================================
# 2️⃣ KEY ALIGNMENT CHECK
# ===========================================================
print("\n=== KEY ALIGNMENT CHECK ===")

same_keys_dis = (df_base[KEYS].values == df_dis[KEYS].values).all()
same_keys_drug = (df_base[KEYS].values == df_drug[KEYS].values).all()

print("Base vs Disease aligned:", same_keys_dis)
print("Base vs Drug aligned   :", same_keys_drug)


# ===========================================================
# 3️⃣ IF ALIGNED → MERGE DIRECTLY
# ===========================================================
if same_keys_dis and same_keys_drug:
    print("\n➡ Files are aligned → merging directly")

    df_dis  = df_dis.drop(columns=KEYS, errors="ignore")
    df_drug = df_drug.drop(columns=KEYS, errors="ignore")

    df_final = pd.concat([df_base, df_dis, df_drug], axis=1)

    df_final.to_parquet(OUT_PARQUET, index=False)
    df_final.to_excel(OUT_EXCEL, index=False)

    print("\nSaved:")
    print("Parquet:", OUT_PARQUET)
    print("Excel  :", OUT_EXCEL)
    raise SystemExit


# ===========================================================
# 4️⃣ IF NOT ALIGNED → FIX USING MERGE ON KEYS
# ===========================================================
print("\n⚠ Files NOT aligned → fixing alignment using join keys...")

# Check duplicates
for file, df in [("BASE", df_base), ("DISEASE", df_dis), ("DRUG", df_drug)]:
    if df.duplicated(KEYS).sum() > 0:
        raise ValueError(f"❌ Duplicate Trial ID + Arm ID in {file}")

# Merge (safe alignment)
df_final = (
    df_base
    .merge(df_dis.drop(columns=["Median PFS"], errors="ignore"), on=KEYS, how="left")
    .merge(df_drug.drop(columns=["Median PFS"], errors="ignore"), on=KEYS, how="left")
)

# Save outputs
df_final.to_parquet(OUT_PARQUET, index=False)
df_final.to_excel(OUT_EXCEL, index=False)

print("\nAlignment fixed and files saved:")
print("Parquet:", OUT_PARQUET)
print("Excel  :", OUT_EXCEL)

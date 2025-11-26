import pandas as pd
import os

# ================================
# CONFIG
# ================================
# INPUT_FILE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_cleaned1.xlsx"
INPUT_FILE=r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters.xlsx"
OUTPUT_FILE = r"C:\LARVOL_WORK\Median_PFS\data\all_unique_value_counts.xlsx"

# ================================
# LOAD FILE
# ================================
df = pd.read_excel(INPUT_FILE)
print(f"Loaded dataset: {df.shape}")

# ================================
# CREATE EXCEL WRITER
# ================================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    summary_rows = []

    # Loop through every column
    for col in df.columns:
        print(f"Processing column: {col}")

        # Compute value counts
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = [col, "count"]

        # Write each column's counts as a sheet
        sheet_name = col[:31]  # Excel limit
        counts.to_excel(writer, sheet_name=sheet_name, index=False)

        # Add summary info
        summary_rows.append({
            "Feature": col,
            "Unique Values": counts.shape[0]
        })

    # Summary sheet
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)

print(f"\n✅ All unique value counts saved to:\n{OUTPUT_FILE}")

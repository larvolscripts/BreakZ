import os
import pandas as pd

# -------------------------------
# ⚙️ Config
# -------------------------------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianOS_cleaned.xlsx"
OUTPUT_DIR = r"C:\LARVOL_WORK\Median_PFS\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 📂 Load Data
# -------------------------------
df = pd.read_excel(INPUT_PATH)
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------
# 🔍 Extract Unique Products
# -------------------------------
if "Product" not in df.columns:
    raise KeyError("❌ 'Product' column not found in the dataset!")

unique_products = df["Product"].dropna().unique()
unique_products_sorted = sorted(unique_products)

# Print summary in console
print(f"\n🔹 Total unique products: {len(unique_products_sorted)}\n")
for i, prod in enumerate(unique_products_sorted, 1):
    print(f"{i}. {prod}")

# Save to Excel
output_path = os.path.join(OUTPUT_DIR, "unique_products_list_OS.xlsx")
pd.DataFrame({"Unique Products": unique_products_sorted}).to_excel(output_path, index=False)
print(f"\n✅ Unique products saved to: {output_path}")

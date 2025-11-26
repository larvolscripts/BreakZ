import pandas as pd
import re
import os

# ================================
# 1. Input & Output Paths
# ================================
IN_FILE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA.xlsx"
OUT_FILE = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_categories.xlsx"

print("📂 Loading:", IN_FILE)
df = pd.read_excel(IN_FILE, dtype=str)

# Ensure column exists
if "Primary_MOA_all" not in df.columns:
    raise Exception("❌ ERROR: Column 'Primary_MOA_all' not found in the input file.")

# ================================
# 2. Clean MOA text
# ================================
def clean_moa(x):
    if pd.isna(x):
        return ""
    s = str(x).lower().strip()
    s = s.replace(";", ",")
    s = re.sub(r"[^a-z0-9, ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["MOA_clean"] = df["Primary_MOA_all"].apply(clean_moa)

# ================================
# 3. Rule-based Category Mapping
# ================================
MOA_RULES = {
    "PD1/PD-L1 (checkpoint)": ["pd1", "pd-1", "pdl1", "pd-l1", "pembrolizumab", "nivolumab",
                               "atezolizumab", "durvalumab", "avelumab"],
    "CTLA4": ["ctla4", "ctla", "ipilimumab"],
    "TKI / Multi-kinase": ["tinib", "kinase", "vegfr", "egfr", "braf", "alk", "ros1", "ret", "fgfr", "met", "mek"],
    "Antibody / ADC": ["adc", "antibody-drug", "antibody drug", "antibody", "conjugate", "mab"],
    "Chemotherapy": ["carboplatin", "cisplatin", "oxaliplatin", "paclitaxel", "docetaxel",
                     "fluorouracil", "5-fu", "irinotecan", "etoposide", "cyclophosphamide", "gemcitabine"],
    "Hormonal": ["letrozole", "tamoxifen", "anastrozole", "enzalutamide", "aromatase", "estrogen"],
    "CDK4/6": ["cdk4", "cdk6"],
    "PARP": ["parp"],
    "mTOR / PI3K / AKT": ["mtor", "pi3k", "akt"],
    "CAR-T / Cellular": ["car-t", "cart", "bcma", "cd19"],
    "IL / Cytokine": ["il-", "interleukin", "il6", "il1", "cytokine"],
    "DNA synthesis/Antimetabolite": ["thymidylate", "antimetabolite", "folate", "dna synthesis"],
    "Microtubule / Tubulin": ["tubulin", "microtubule", "taxane"],
    "VEGF / Angiogenesis": ["vegf", "vegfr", "bevacizumab"],
    "Other / Unknown": []
}

def assign_category(text):
    if not text:
        return "Other / Unknown"
    for cat, kws in MOA_RULES.items():
        for kw in kws:
            if kw in text:
                return cat
    return "Other / Unknown"

df["Primary_MOA_Category"] = df["MOA_clean"].apply(assign_category)

# ================================
# 4. Save updated file
# ================================
df.to_excel(OUT_FILE, index=False)
print("✅ Saved with categories →", OUT_FILE)

# Show summary
print("\n📊 Category distribution:")
print(df["Primary_MOA_Category"].value_counts())

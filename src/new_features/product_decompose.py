# product_decompose_features.py
import os
import re
from collections import defaultdict
import pandas as pd
import numpy as np

# ------------------------
# CONFIG - change paths here if needed
# ------------------------
INPUT_XLSX = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
OUTPUT_XLSX = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_product_features.xlsx"
OUTPUT_FEATURES_CSV = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_product_feature_list.csv"
OUTPUT_SUMMARY_CSV = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_product_feature_summary.csv"

# ------------------------
# Helper cleaning/tokenization
# ------------------------
def clean_product_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    # unify separators to comma
    s = re.sub(r"[+/;&|]", ",", s)
    s = s.replace(" and ", ",")
    s = s.replace(" + ", ",")
    # replace parentheses and other punctuation with commas
    s = re.sub(r"[()\[\]{}:]", ",", s)
    # remove common trailing words like "tablet", "iv", "po" as noise
    s = re.sub(r"\b(tablet|iv|po|oral|iv\/|iv,|im|sc|subcut|sc\.)\b", " ", s)
    # remove extra spaces and commas
    s = re.sub(r"[,]+", ",", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" ,")
    return s

def tokenize_product(s):
    s = clean_product_text(s)
    if s == "":
        return []
    parts = []
    # split by comma, underscore, whitespace
    for token in re.split(r"[,_\t\n\r]+|,|\s-\s", s):
        token = token.strip(" -,.")
        if token:
            parts.append(token)
    # further split tokens that are very long by whitespace
    flat = []
    for t in parts:
        sub = re.split(r"\s+", t)
        for st in sub:
            st = st.strip()
            if st:
                flat.append(st)
    # deduplicate while preserving order
    seen = set()
    out = []
    for t in flat:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ------------------------
# Canonical mapping: substring -> canonical token(s)
# (expand as needed — this list targets the frequent drugs/regimens in your list)
# ------------------------
TOKEN_MAP = {
    # PD-1 / PD-L1 / CTLA4
    "pembrolizumab": "pembrolizumab",
    "pembro": "pembrolizumab",
    "pembrolizumab_lambrolizumab": "pembrolizumab",
    "nivolumab": "nivolumab",
    "ipilimumab": "ipilimumab",
    "atezolizumab": "atezolizumab",
    "durvalumab": "durvalumab",
    "avelumab": "avelumab",
    "cemiplimab": "cemiplimab",
    "tislelizumab": "tislelizumab",
    "sintilimab": "sintilimab",
    "camrelizumab": "camrelizumab",
    "toripalimab": "toripalimab",
    "relatlimab": "relatlimab",
    # PARP
    "olaparib": "olaparib",
    "niraparib": "niraparib",
    "rucaparib": "rucaparib",
    # EGFR / ALK / ROS1 / BRAF / MEK
    "osimertinib": "osimertinib",
    "erlotinib": "erlotinib",
    "gefitinib": "gefitinib",
    "afatinib": "afatinib",
    "almonertinib": "almonertinib",
    "brigatinib": "brigatinib",
    "lorlatinib": "lorlatinib",
    "crizotinib": "crizotinib",
    "dabrafenib": "dabrafenib",
    "trametinib": "trametinib",
    # TKIs / VEGFR / multi-kinase
    "cabozantinib": "cabozantinib",
    "lenvatinib": "lenvatinib",
    "sunitinib": "sunitinib",
    "pazopanib": "pazopanib",
    "axitinib": "axitinib",
    "vandetanib": "vandetanib",
    # CHEMO (platinum, taxane, others)
    "cisplatin": "cisplatin",
    "carboplatin": "carboplatin",
    "oxaliplatin": "oxaliplatin",
    "paclitaxel": "paclitaxel",
    "docetaxel": "docetaxel",
    "gemcitabine": "gemcitabine",
    "doxorubicin": "doxorubicin",
    "capecitabine": "capecitabine",
    "pemetrexed": "pemetrexed",
    "irinotecan": "irinotecan",
    "etoposide": "etoposide",
    "5-fu": "5-fu",
    "fluorouracil": "5-fu",
    "folfox": "folfox",
    "capox": "capox",
    "folfiri": "fo lfiri",  # fallback
    # ADCs / antibodies
    "trastuzumab_deruxtecan": "trastuzumab_deruxtecan",
    "ado_trastuzumab_emtansine": "t_dmx",
    "trastuzumab": "trastuzumab",
    "brentuximab": "brentuximab_vedotin",
    "enfortumab": "enfortumab_vedotin",
    "polatuzumab": "polatuzumab",
    "mirvetuximab": "mirvetuximab",
    # Hormonal / endocrine
    "letrozole": "letrozole",
    "tamoxifen": "tamoxifen",
    "fulvestrant": "fulvestrant",
    "abiraterone": "abiraterone",
    "enzalutamide": "enzalutamide",
    # CDK4/6
    "palbociclib": "palbociclib",
    "ribociclib": "ribociclib",
    "abemaciclib": "abemaciclib",
    # BTK / BCL2 / venetoclax etc
    "ibrutinib": "ibrutinib",
    "venetoclax": "venetoclax",
    # Anti-angiogenic / VEGF antibodies
    "bevacizumab": "bevacizumab",
    # Immunomodulatory drugs
    "lenalidomide": "lenalidomide",
    "pomalidomide": "pomalidomide",
    # Proteasome
    "bortezomib": "bortezomib",
    "carfilzomib": "carfilzomib",
    # CAR-T and cell therapies
    "idecabtagene": "idecabtagene_vicleucel",
    "ciltacabtagene": "ciltacabtagene_autoleucel",
    "car-t": "car_t",
    "car_t": "car_t",
    # Others / fallbacks
    "placebo": "placebo",
    "best_supportive_care": "best_supportive_care",
    "surgery": "surgery",
    "rt": "radiation",
    "radiotherapy": "radiation",
    # add more mappings as you discover tokens...
}

# ------------------------
# Class mapping: canonical token -> set of classes
# (a canonical drug may map to multiple higher-level classes)
# ------------------------
CLASS_MAP = defaultdict(list)
# Immune checkpoint families
for drug in ["pembrolizumab", "nivolumab", "cemiplimab", "tislelizumab", "sintilimab", "camrelizumab", "toripalimab"]:
    CLASS_MAP[drug].append("PD1_inhibitor")
for drug in ["atezolizumab", "avelumab"]:
    CLASS_MAP[drug].append("PDL1_inhibitor")
CLASS_MAP["ipilimumab"].append("CTLA4_inhibitor")

# PARP
CLASS_MAP["olaparib"].append("PARP_inhibitor")
CLASS_MAP["niraparib"].append("PARP_inhibitor")
CLASS_MAP["rucaparib"].append("PARP_inhibitor")

# TKIs & targets
tkis = ["osimertinib","erlotinib","gefitinib","afatinib","brigatinib","lorlatinib","crizotinib",
        "cabozantinib","lenvatinib","sunitinib","pazopanib","axitinib","vandetanib"]
for d in tkis:
    CLASS_MAP[d].append("TKI")
# label specific target where useful
CLASS_MAP["osimertinib"].append("EGFR_TKI")
CLASS_MAP["erlotinib"].append("EGFR_TKI")
CLASS_MAP["gefitinib"].append("EGFR_TKI")
CLASS_MAP["brigatinib"].append("ALK_inhibitor")
CLASS_MAP["lorlatinib"].append("ALK_inhibitor")
CLASS_MAP["crizotinib"].append("ALK_inhibitor")
CLASS_MAP["cabozantinib"].append("VEGFR_MultiTKI")
CLASS_MAP["lenvatinib"].append("VEGFR_MultiTKI")

# Chemotherapy / platinums / taxanes
chemo_agents = ["cisplatin","carboplatin","oxaliplatin","paclitaxel","docetaxel","gemcitabine","doxorubicin",
                "capecitabine","pemetrexed","irinotecan","etoposide","5-fu"]
for a in chemo_agents:
    CLASS_MAP[a].append("Chemotherapy")
for p in ["cisplatin","carboplatin","oxaliplatin"]:
    CLASS_MAP[p].append("Platinum")
for t in ["paclitaxel","docetaxel"]:
    CLASS_MAP[t].append("Taxane")

# ADC / Antibodies
for a in ["trastuzumab","t_dmx","trastuzumab_deruxtecan","brentuximab_vedotin","enfortumab_vedotin","polatuzumab","mirvetuximab"]:
    CLASS_MAP[a].append("Antibody_or_ADC")
    CLASS_MAP[a].append("Antibody")

# Hormonal
for h in ["letrozole","tamoxifen","fulvestrant","abiraterone","enzalutamide"]:
    CLASS_MAP[h].append("Hormonal")

# CDK4/6
for c in ["palbociclib","ribociclib","abemaciclib"]:
    CLASS_MAP[c].append("CDK4_6")

# IMiDs & Proteasome
for d in ["lenalidomide","pomalidomide"]:
    CLASS_MAP[d].append("IMiD")
for d in ["bortezomib","carfilzomib"]:
    CLASS_MAP[d].append("Proteasome_inhibitor")

# BTK / BCL2
CLASS_MAP["ibrutinib"].append("BTK_inhibitor")
CLASS_MAP["venetoclax"].append("BCL2_inhibitor")

# CAR-T
CLASS_MAP["idecabtagene_vicleucel"].append("CAR_T")
CLASS_MAP["ciltacabtagene_autoleucel"].append("CAR_T")
CLASS_MAP["car_t"].append("CAR_T")

# Radiation / supportive
CLASS_MAP["radiation"].append("Radiation")
CLASS_MAP["placebo"].append("Placebo")
CLASS_MAP["best_supportive_care"].append("Supportive_Care")

# ------------------------
# Utility functions to map tokens -> canonical -> classes
# ------------------------
def canonicalize_token(tok):
    tok = tok.lower().strip()
    # direct exact match
    if tok in TOKEN_MAP:
        return TOKEN_MAP[tok]
    # substring matching fallback (longer tokens first)
    for key in sorted(TOKEN_MAP.keys(), key=lambda x: -len(x)):
        if key in tok:
            return TOKEN_MAP[key]
    # if token looks like "nab_paclitaxel" => paclitaxel
    if "paclitaxel" in tok:
        return "paclitaxel"
    if "carboplatin" in tok:
        return "carboplatin"
    if "cisplatin" in tok:
        return "cisplatin"
    # fallback: return original token (we'll label as Other)
    return tok

def classes_for_canonical(canon):
    return CLASS_MAP.get(canon, ["Other"])

# ------------------------
# Main process
# ------------------------
def process_dataframe(df):
    product_col = "Product" if "Product" in df.columns else None
    if product_col is None:
        raise ValueError("Product column not found in dataframe.")
    # ensure cluster columns exist (for interactions)
    moa_col = "MOA_cluster_labeL" if "MOA_cluster_labeL" in df.columns else ("Primary_MOA_all" if "Primary_MOA_all" in df.columns else None)
    precise_cluster_col = "Precise_cluster_label" if "Precise_cluster_label" in df.columns else ("Precise_Area_Name" if "Precise_Area_Name" in df.columns else None)

    # containers for new features
    new_cols = []
    rows = []

    # set of all discovered canonical tokens and classes
    discovered_tokens = set()
    discovered_classes = set()

    for idx, prod in df[product_col].fillna("").items():
        toks = tokenize_product(prod)
        canonical_tokens = []
        token_classes = []
        for t in toks:
            canon = canonicalize_token(t)
            canonical_tokens.append(canon)
            discovered_tokens.add(canon)
            cs = classes_for_canonical(canon)
            token_classes.extend(cs)
            for c in cs:
                discovered_classes.add(c)
        # counts
        drug_count = len(canonical_tokens)
        unique_drugs = len(set(canonical_tokens))
        # count by class groups
        class_counts = defaultdict(int)
        for c in token_classes:
            class_counts[c] += 1

        # build row features
        row = {
            "product_tokens": ";".join(canonical_tokens),
            "drug_count": drug_count,
            "unique_drug_count": unique_drugs,
            "is_combination": int(unique_drugs > 1),
        }
        # binary flags and counts for discovered class list (we will fill columns later)
        for cls in ["PD1_inhibitor","PDL1_inhibitor","CTLA4_inhibitor","TKI","EGFR_TKI","ALK_inhibitor","VEGFR_MultiTKI",
                    "PARP_inhibitor","Chemotherapy","Platinum","Taxane","Antibody_or_ADC","Antibody","Hormonal",
                    "CDK4_6","IMiD","Proteasome_inhibitor","BTK_inhibitor","BCL2_inhibitor","CAR_T","Radiation","Placebo","Supportive_Care"]:
            row[f"has_{cls}"] = int(class_counts.get(cls, 0) > 0)
            row[f"count_{cls}"] = int(class_counts.get(cls, 0))

        # immuno / chemo / tki numeric counts for combo intensity
        row["immuno_count"] = class_counts.get("PD1_inhibitor",0) + class_counts.get("PDL1_inhibitor",0) + class_counts.get("CTLA4_inhibitor",0)
        row["chemo_count"] = class_counts.get("Chemotherapy",0)
        row["tki_count"] = class_counts.get("TKI",0)

        # interaction flags with MOA / Precise clusters if present
        if moa_col:
            moa_val = str(df.at[idx, moa_col]) if not pd.isna(df.at[idx, moa_col]) else ""
            row["MOA_cluster_val"] = moa_val
            row["MOAcluster_has_PD1"] = int(bool(moa_val) and row["has_PD1_inhibitor"])
            row["MOAcluster_has_Chemo"] = int(bool(moa_val) and row["has_Chemotherapy"])
        else:
            row["MOA_cluster_val"] = ""
            row["MOAcluster_has_PD1"] = 0
            row["MOAcluster_has_Chemo"] = 0

        if precise_cluster_col:
            pc_val = str(df.at[idx, precise_cluster_col]) if not pd.isna(df.at[idx, precise_cluster_col]) else ""
            row["Precise_cluster_val"] = pc_val
            row["Precisecluster_has_PD1"] = int(bool(pc_val) and row["has_PD1_inhibitor"])
            row["Precisecluster_has_Chemo"] = int(bool(pc_val) and row["has_Chemotherapy"])
        else:
            row["Precise_cluster_val"] = ""
            row["Precisecluster_has_PD1"] = 0
            row["Precisecluster_has_Chemo"] = 0

        rows.append(row)

    # assemble new df
    new_df = pd.DataFrame(rows, index=df.index)

    # drop the verbose cluster_val columns from final training features, but keep for inspection
    # final feature list (exclude product_tokens, MOA/Precise cluster val strings)
    feature_cols = [c for c in new_df.columns if c not in ["product_tokens","MOA_cluster_val","Precise_cluster_val"]]

    return new_df, feature_cols, discovered_tokens, discovered_classes

# ------------------------
# Run
# ------------------------
def main():
    print("Loading:", INPUT_XLSX)
    df = pd.read_excel(INPUT_XLSX)
    print("Rows:", df.shape[0], "Cols:", df.shape[1])

    new_df, feature_cols, tokens, classes = process_dataframe(df)

    # merge back into original (keeping original Product etc)
    out = pd.concat([df.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    # Save enriched file
    os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
    out.to_excel(OUTPUT_XLSX, index=False)
    print("Saved enriched dataset to:", OUTPUT_XLSX)

    # Save feature list
    pd.Series(feature_cols).to_csv(OUTPUT_FEATURES_CSV, index=False)
    print("Saved feature list to:", OUTPUT_FEATURES_CSV)

    # Save short summary (feature counts)
    summary = []
    for c in feature_cols:
        if c.startswith("has_") or c.startswith("count_") or c.endswith("_count") or c in ["drug_count","unique_drug_count","is_combination","immuno_count","chemo_count","tki_count"]:
            series = out[c]
            summary.append({
                "feature": c,
                "nonzero_count": int((series != 0).sum()),
                "mean": float(series.mean()) if np.issubdtype(series.dtype, np.number) else np.nan
            })
    pd.DataFrame(summary).to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print("Saved summary to:", OUTPUT_SUMMARY_CSV)

    print("\nTop discovered canonical tokens (sample):")
    for t in list(tokens)[:40]:
        print(" -", t)
    print("\nTop discovered classes (sample):")
    for t in list(classes)[:40]:
        print(" -", t)

if __name__ == "__main__":
    main()

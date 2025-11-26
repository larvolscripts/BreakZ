
# import pandas as pd
# import json

# # ---------------------------
# # Step 1: Load product → MOA mapping
# # ---------------------------
# product_file = r"C:\LARVOL_WORK\Median_PFS\data\ProductsList 23May2025.xlsx"
# df_products = pd.read_excel(product_file)

# liid_to_moa = df_products.set_index("LI Id")[["Primary MOA","Primary MOA Category", "Secondary MOA","Secondary MOA Category"]].to_dict("index")

# # ---------------------------
# # Step 2: Load JSON (pairz)
# # ---------------------------
# json_file = r"C:\LARVOL_WORK\Median_PFS\data\pairz_all_data 2.json"
# with open(json_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # ---------------------------
# # Step 3: Build trial_id → aggregated MOAs
# # ---------------------------
# trial_to_moa = {}

# for trial in data:
#     trial_id = trial.get("trial_id")
#     if not trial_id or "arms" not in trial:
#         continue

#     primary_moas, secondary_moas = set(), set()

#     for _, arm in trial["arms"].items():
#         if not isinstance(arm, dict):
#             continue

#         for prod in arm.get("product", []):
#             li = prod.get("li_id")
#             if not li:
#                 continue

#             moa_info = liid_to_moa.get(li)
#             if not moa_info:
#                 continue

#             if moa_info.get("Primary MOA"):
#                 primary_moas.add(str(moa_info["Primary MOA"]))
#             if moa_info.get("Secondary MOA"):
#                 secondary_moas.add(str(moa_info["Secondary MOA"]))

#     trial_to_moa[trial_id] = (
#         "; ".join(sorted(primary_moas)),
#         "; ".join(sorted(secondary_moas))
#     )

# print(f"✅ Collected MOAs for {len(trial_to_moa)} trials")

# # ---------------------------
# # Step 4: Load your ITT/Subgroup data
# # ---------------------------
# itt_file = r"C:\LARVOL_WORK\Median_PFS\data\Median_PFS_precise.xlsx"
# df = pd.read_excel(itt_file)

# # Make sure Trial ID column is consistent with JSON keys
# df["Trial ID"] = df["Trial ID"].astype(str).str.strip()

# # ---------------------------
# # Step 5: Attach MOAs by Trial ID
# # ---------------------------
# df[["Primary_MOA_all", "Secondary_MOA_all"]] = df["Trial ID"].apply(
#     lambda tid: pd.Series(trial_to_moa.get(tid, ("", "")))
# )

# # ---------------------------
# # Step 6: Save updated Excel
# # ---------------------------
# output_file = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_precise_moa.xlsx"
# df.to_excel(output_file, index=False)

# print(f"✅ Updated file saved: {output_file}")

# -----------------------CATEGORY----CORRECT------------
# import pandas as pd
# import json
# import os

# # ---------------------------
# # Step 1: Load product → MOA mapping
# # ---------------------------
# product_file = r"C:\LARVOL_WORK\Median_PFS\data\ProductsList 23May2025.xlsx"
# df_products = pd.read_excel(product_file)

# moa_cols = [
#     "Primary MOA",
#     "Primary MOA Category",
#     "Secondary MOA",
#     "Secondary MOA Category"
# ]

# # li_id → {4 moa fields}
# liid_to_moa = df_products.set_index("LI Id")[moa_cols].to_dict("index")

# # ---------------------------
# # Step 2: Load JSON
# # ---------------------------
# json_file = r"C:\LARVOL_WORK\Median_PFS\data\pairz_all_data 2.json"
# with open(json_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # ---------------------------
# # Step 3: Build trial_id → aggregated MOAs (all 4)
# # ---------------------------
# trial_to_moa = {}

# for trial in data:
#     trial_id = str(trial.get("trial_id", "")).strip()
#     if not trial_id or "arms" not in trial:
#         continue

#     # store all MOA fields in sets
#     prim, prim_cat, sec, sec_cat = set(), set(), set(), set()

#     for _, arm in trial["arms"].items():
#         if not isinstance(arm, dict):
#             continue

#         for prod in arm.get("product", []):
#             li = prod.get("li_id")
#             if not li:
#                 continue

#             moa = liid_to_moa.get(li)
#             if not moa:
#                 continue

#             if moa.get("Primary MOA"):
#                 prim.add(str(moa["Primary MOA"]))
#             # if moa.get("Primary MOA Category"):
#             #     prim_cat.add(str(moa["Primary MOA Category"]))
#             if moa.get("Secondary MOA"):
#                 sec.add(str(moa["Secondary MOA"]))
#             # if moa.get("Secondary MOA Category"):
#             #     sec_cat.add(str(moa["Secondary MOA Category"]))

#     trial_to_moa[trial_id] = (
#         "; ".join(sorted(prim)),
#         # "; ".join(sorted(prim_cat)),
#         "; ".join(sorted(sec)),
#         # "; ".join(sorted(sec_cat))
#     )

# print(f"✅ Collected MOAs for {len(trial_to_moa)} trials")

# # ---------------------------
# # Step 4: Load your trial dataset
# # ---------------------------
# itt_file = r"C:\LARVOL_WORK\Median_PFS\data\Median_PFS_precise.xlsx"
# df = pd.read_excel(itt_file)

# df["Trial ID"] = df["Trial ID"].astype(str).str.strip()


# # ---------------------------
# # Step 5: Attach all 4 MOA fields
# # ---------------------------
# df[[
#     "Primary_MOA_all",
#     # "Primary_MOA_Category",
#     "Secondary_MOA_all",
#     # "Secondary_MOA_Category"
# ]] = df["Trial ID"].apply(
#     lambda tid: pd.Series(trial_to_moa.get(tid, ("", "", "", "")))
# )

# # ---------------------------
# # Step 6: Save
# # ---------------------------
# output_file = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_precise_moa.xlsx"
# df.to_excel(output_file, index=False)

# print(f"✅ Updated file saved: {output_file}")


import pandas as pd
import json
import os

# ---------------------------
# Step 1: Load product → MOA mapping
# ---------------------------
product_file = r"C:\LARVOL_WORK\Median_PFS\data\ProductsList 23May2025.xlsx"
df_products = pd.read_excel(product_file)

# Only take the 2 MOA fields you want:
moa_cols = [
    "Primary MOA",
    "Secondary MOA"
]

# li_id → {Primary MOA, Secondary MOA}
liid_to_moa = df_products.set_index("LI Id")[moa_cols].to_dict("index")

# ---------------------------
# Step 2: Load JSON
# ---------------------------
json_file = r"C:\LARVOL_WORK\Median_PFS\data\pairz_all_data 2.json"
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------------------------
# Step 3: Build trial_id → aggregated MOAs (only 2)
# ---------------------------
trial_to_moa = {}

for trial in data:
    trial_id = str(trial.get("trial_id", "")).strip()
    if not trial_id or "arms" not in trial:
        continue

    prim, sec = set(), set()

    # Loop through arms & products
    for _, arm in trial["arms"].items():
        if not isinstance(arm, dict):
            continue

        for prod in arm.get("product", []):
            li = prod.get("li_id")
            if not li:
                continue

            moa = liid_to_moa.get(li)
            if not moa:
                continue

            if moa.get("Primary MOA"):
                prim.add(str(moa["Primary MOA"]))

            if moa.get("Secondary MOA"):
                sec.add(str(moa["Secondary MOA"]))

    # save tuple with ONLY 2 fields
    trial_to_moa[trial_id] = (
        "; ".join(sorted(prim)),
        "; ".join(sorted(sec))
    )

print(f"✅ Collected MOAs for {len(trial_to_moa)} trials")

# ---------------------------
# Step 4: Load your Median PFS dataset
# ---------------------------
itt_file = r"C:\LARVOL_WORK\Median_PFS\data\Median_PFS_precise.xlsx"
df = pd.read_excel(itt_file)

df["Trial ID"] = df["Trial ID"].astype(str).str.strip()

# ---------------------------
# Step 5: Attach MOAs (only 2 fields)
# ---------------------------
df[["Primary_MOA_all", "Secondary_MOA_all"]] = df["Trial ID"].apply(
    lambda tid: pd.Series(trial_to_moa.get(tid, ("", "")))
)

# ---------------------------
# Step 6: Save to Excel
# ---------------------------
output_file = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_precise_moa.xlsx"
df.to_excel(output_file, index=False)

print(f"✅ Updated file saved: {output_file}")

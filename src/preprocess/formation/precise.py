import requests
import pandas as pd
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------
# Step 1: Load Excel
# ---------------------------
excel_file = r"C:\LARVOL_WORK\Median_PFS\outputs\MedianPFS_cleaned.xlsx"
df = pd.read_excel(excel_file)

trial_id_col = "Trial ID"
df[trial_id_col] = df[trial_id_col].astype(str)

# ---------------------------
# Step 2: Function with retry
# ---------------------------
def fetch_precise_area_names(trial_id, headers, retries=3, wait=5):
    url = f"https://lt.larvol.com/OncotrialsApi.php?source_id={trial_id}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=90)
            r.raise_for_status()
            data = r.json()
            if "precise_area" in data and isinstance(data["precise_area"], list):
                names = [item.get("name", "") for item in data["precise_area"] if "name" in item]
                return "; ".join(names) if names else None
            return None
        except Exception as e:
            print(f"⚠️ Error fetching {trial_id} (attempt {attempt+1}): {e}")
            time.sleep(wait)  # wait before retry
    return None  # failed after all retries

# ---------------------------
# Step 3: API headers
# ---------------------------
headers = {
    "Authorization": "099fffa1-2812-11ef-ab06-06f6dfe7d668"
}

# ---------------------------
# Step 4: Resume from cache if exists
# ---------------------------
cache_file = r"C:\LARVOL_WORK\Median_PFS\data\precise_area_cache.csv"

if os.path.exists(cache_file):
    cache_df = pd.read_csv(cache_file, dtype=str)
    trialid_to_precisearea = dict(zip(cache_df["Trial ID"], cache_df["Precise_Area_Name"]))
    print(f"♻️ Loaded cache for {len(trialid_to_precisearea)} trials.")
else:
    trialid_to_precisearea = {}

# ---------------------------
# Step 5: Fetch remaining trials in parallel
# ---------------------------
all_trial_ids = [tid for tid in df[trial_id_col].dropna().unique() if tid not in trialid_to_precisearea]

print(f"🚀 Fetching {len(all_trial_ids)} new trials in parallel...")

max_workers = 5  # number of parallel requests
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(fetch_precise_area_names, tid, headers): tid for tid in all_trial_ids}

    for idx, future in enumerate(as_completed(futures), start=1):
        trial_id = futures[future]
        try:
            trialid_to_precisearea[trial_id] = future.result()
        except Exception as e:
            print(f"⚠️ Failed for {trial_id}: {e}")

        # Save cache every 50 completed trials
        if idx % 50 == 0:
            pd.DataFrame({
                "Trial ID": list(trialid_to_precisearea.keys()),
                "Precise_Area_Name": list(trialid_to_precisearea.values())
            }).to_csv(cache_file, index=False)
            print(f"💾 Saved cache after {idx} trials")

# Final save of cache
pd.DataFrame({
    "Trial ID": list(trialid_to_precisearea.keys()),
    "Precise_Area_Name": list(trialid_to_precisearea.values())
}).to_csv(cache_file, index=False)

print(f"✅ Collected Precise Area names for {len(trialid_to_precisearea)} trials.")

# ---------------------------
# Step 6: Map results back into DataFrame
# ---------------------------
df["Precise_Area_Name"] = df[trial_id_col].map(trialid_to_precisearea)

# ---------------------------
# Step 7: Save updated Excel
# ---------------------------
output_file = r"C:\LARVOL_WORK\Median_PFS\data\Median_PFS_precise.xlsx"
df.to_excel(output_file, index=False)

print(f"✅ Updated file saved: {output_file}")
 
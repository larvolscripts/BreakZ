# disease_embed.py
"""
Compute BioClinicalBERT embeddings for Precise_Area_Name (Decision A, Decision F).
Saves:
 - <INPUT_PREFIX>_with_disease_emb.parquet (fast)
 - <INPUT_PREFIX>_with_disease_emb.xlsx (Excel, optional)
"""

import os
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# ---------- CONFIG ----------
INPUT_PATH = r"C:\LARVOL_WORK\Median_PFS\data\MedianPFS_PMOA_with_precise_clusters_v3.xlsx"
OUTPUT_PREFIX = os.path.splitext(INPUT_PATH)[0]  # same folder, same base name
PARQUET_OUT = OUTPUT_PREFIX + "_with_disease_emb.parquet"
EXCEL_OUT = OUTPUT_PREFIX + "_with_disease_emb.xlsx"  # optional (can be slower)
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[^a-z0-9\s\-\+\/\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mean_pool(model_output, mask):
    token_embeddings = model_output.last_hidden_state  # (B, L, H)
    mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()  # B,L,H
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def embed_texts(texts, tokenizer, model):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pool(out, attention_mask)  # (B, H)
            emb = pooled.cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)

def main():
    print("Loading data:", INPUT_PATH)
    df = pd.read_excel(INPUT_PATH, dtype=str)
    if "Precise_Area_Name" not in df.columns:
        raise SystemExit("Precise_Area_Name not found in file")

    # Prepare texts
    df["disease_text"] = df["Precise_Area_Name"].apply(clean_text)

    # Load model
    print("Loading tokenizer & model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    texts = df["disease_text"].fillna("").astype(str).tolist()
    print(f"Embedding {len(texts)} disease texts on {DEVICE} ...")

    emb = embed_texts(texts, tokenizer, model)  # shape (n_samples, 768)

    print("Embedding shape:", emb.shape)

    # attach into dataframe with columns disease_emb_0...disease_emb_767
    emb_cols = [f"disease_emb_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols, index=df.index)
    out_df = pd.concat([df, emb_df], axis=1)

    print("Saving parquet:", PARQUET_OUT)
    out_df.to_parquet(PARQUET_OUT, index=False)
    try:
        print("Also saving Excel (may be slow):", EXCEL_OUT)
        out_df.to_excel(EXCEL_OUT, index=False)
    except Exception as e:
        print("Excel save failed (OK):", e)

    print("Done. Outputs:", PARQUET_OUT)

if __name__ == "__main__":
    main()

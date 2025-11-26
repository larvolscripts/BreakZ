# drug_embed_clean_and_embed.py
"""
Cleans Product strings (Decision D), maps synonyms, splits combos, computes BioClinicalBERT mean embeddings per row.
Saves outputs similarly to disease script.
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
OUTPUT_PREFIX = os.path.splitext(INPUT_PATH)[0]
PARQUET_OUT = OUTPUT_PREFIX + "_with_drug_emb.parquet"
EXCEL_OUT = OUTPUT_PREFIX + "_with_drug_emb.xlsx"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

# small normalization map - expand as you like
SYNONYM_MAP = {
    "opdivo": "nivolumab",
    "keytruda": "pembrolizumab",
    "ipi": "ipilimumab",
    "pembro": "pembrolizumab",
    "atezo": "atezolizumab",
    "durva": "durvalumab",
    "tremel": "tremelimumab",
    # add more as you find in unique product list
}

# stop tokens to drop
STOP_TOKENS = set([
    "rt", "surgery", "placebo", "standard", "of", "care", "best", "supportive", "treatment",
    "chemotherapy", "chemo"  # optional: you may keep 'chemotherapy' if you want it as a token
])

def normalize_token(t):
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9\-]", "", t)
    if t in SYNONYM_MAP:
        return SYNONYM_MAP[t]
    return t

def product_to_drug_tokens(prod_str):
    if pd.isna(prod_str):
        return []
    s = str(prod_str).lower()
    # unify delimiters
    s = re.sub(r"[+/;,&\|]", " ", s)
    s = s.replace("_", " ").replace("-", " ")
    # remove text in parentheses
    s = re.sub(r"\(.*?\)", " ", s)
    # split and clean
    toks = [normalize_token(t) for t in re.split(r"\s+", s) if t.strip()]
    toks = [t for t in toks if len(t) > 1 and t not in STOP_TOKENS]
    # filter numeric-only tokens
    toks = [t for t in toks if not re.fullmatch(r"\d+", t)]
    return list(dict.fromkeys(toks))  # dedupe while preserving order

# same mean pooling helper
def mean_pool(model_output, mask):
    token_embeddings = model_output.last_hidden_state
    mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def embed_texts(texts, tokenizer, model):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64)
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pool(out, attention_mask)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)

def main():
    print("Loading data:", INPUT_PATH)
    df = pd.read_excel(INPUT_PATH, dtype=str)

    if "Product" not in df.columns:
        raise SystemExit("Product column missing")

    # compute token lists
    print("Converting Product -> cleaned tokens")
    df["drug_tokens"] = df["Product"].fillna("").apply(product_to_drug_tokens)

    # collect unique tokens across dataset
    all_tokens = sorted({t for toks in df["drug_tokens"] for t in toks if t})
    print("Unique tokens to embed:", len(all_tokens))

    if len(all_tokens) == 0:
        raise SystemExit("No drug tokens found after cleaning")

    # load model
    print("Loading tokenizer & model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # embed unique tokens in batches
    token_to_emb = {}
    chunks = [all_tokens[i:i+BATCH_SIZE] for i in range(0, len(all_tokens), BATCH_SIZE)]
    for chunk in tqdm(chunks, desc="Embedding token chunks"):
        emb_chunk = embed_texts(chunk, tokenizer, model)
        for t, e in zip(chunk, emb_chunk):
            token_to_emb[t] = e

    # build per-row embedding by averaging token embeddings
    emb_list = []
    for toks in tqdm(df["drug_tokens"].tolist(), desc="Aggregating row embeddings"):
        vecs = [token_to_emb[t] for t in toks if t in token_to_emb]
        if len(vecs) == 0:
            emb_list.append(np.zeros(next(iter(token_to_emb.values())).shape))
        else:
            emb_list.append(np.mean(vecs, axis=0))

    emb = np.vstack(emb_list)
    emb_cols = [f"drug_emb_{i}" for i in range(emb.shape[1])]
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

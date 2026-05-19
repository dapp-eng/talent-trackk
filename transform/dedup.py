import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_PROCESSED_DIR


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def dedup_within_batch(df: pd.DataFrame, embeddings: np.ndarray,
                        sim_threshold: float = 0.92) -> pd.Series:
    n = len(df)
    if n == 0:
        return pd.Series(dtype=bool)

    sim_matrix = cosine_similarity_matrix(embeddings, embeddings)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            same_company = (
                str(df.iloc[i].get("company_clean", "")).lower() ==
                str(df.iloc[j].get("company_clean", "")).lower()
            )
            title_i = str(df.iloc[i].get("title_clean", ""))
            title_j = str(df.iloc[j].get("title_clean", ""))
            title_sim = _title_sim(title_i, title_j)

            if sim_matrix[i, j] > sim_threshold and (same_company or title_sim > 0.8):
                keep[j] = False

    return pd.Series(keep, index=df.index)


def _title_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def dedup_against_existing_hashes(df: pd.DataFrame, existing_hashes: set) -> pd.DataFrame:
    before = len(df)
    df = df[~df["source_hash"].isin(existing_hashes)]
    after = len(df)
    print(f"Dedup against existing: removed {before - after}, kept {after}")
    return df


def run_dedup(preprocessed_path: str, embeddings_path: str = None,
               existing_hashes: set = None) -> Path:
    df = pd.read_parquet(preprocessed_path)
    print(f"Dedup input: {len(df)} rows")

    if existing_hashes:
        df = dedup_against_existing_hashes(df, existing_hashes)

    if embeddings_path and len(df) > 1:
        npz = np.load(embeddings_path, allow_pickle=True)
        hash_arr = npz["source_hashes"].astype(str)
        sbert_arr = npz["sbert"]
        hash_to_idx = {h: i for i, h in enumerate(hash_arr)}

        indices = [hash_to_idx.get(h) for h in df["source_hash"].values]
        valid = [i for i in indices if i is not None]
        valid_mask = [i is not None for i in indices]

        if len(valid) > 1:
            sub_embs = sbert_arr[valid]
            sub_df = df.iloc[[i for i, v in enumerate(valid_mask) if v]].reset_index(drop=True)
            keep_mask = dedup_within_batch(sub_df, sub_embs)
            kept_hashes = set(sub_df.loc[keep_mask.values, "source_hash"])
            invalid_hashes = set(df.iloc[[i for i, v in enumerate(valid_mask) if not v]]["source_hash"])
            df = df[df["source_hash"].isin(kept_hashes | invalid_hashes)]

    out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_deduped.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"Dedup output: {len(df)} rows → {out_path}")
    return out_path

import gc
import time
import requests
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from config import (
    JOBBERT_MODEL, SBERT_MODEL,
    EMBEDDING_DIM, SBERT_DIM,
    DATA_PROCESSED_DIR, HF_API_TOKEN,
)

logger = logging.getLogger(__name__)

HF_API_URL_JOBBERT = f"https://api-inference.huggingface.co/models/{JOBBERT_MODEL}"
HF_API_URL_SBERT = f"https://api-inference.huggingface.co/models/{SBERT_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def _chunk_text(text: str, max_chars: int = 1800, overlap_chars: int = 200) -> list:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += max_chars - overlap_chars
    return chunks


def _call_hf_embedding(url: str, texts: list, retries: int = 3) -> np.ndarray | None:
    dim = EMBEDDING_DIM if "jobbert" in url else SBERT_DIM
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    arr = np.array(result, dtype=np.float32)
                    if arr.ndim == 2:
                        return arr
                    elif arr.ndim == 3:
                        return arr.mean(axis=1).astype(np.float32)
                return None
            elif resp.status_code == 503:
                wait = int(resp.json().get("estimated_time", 20))
                logger.warning(f"HF model loading, waiting {wait}s...")
                time.sleep(min(wait, 30))
            elif resp.status_code == 429:
                logger.warning(f"HF rate limit, waiting 10s (attempt {attempt+1})")
                time.sleep(10)
            else:
                logger.warning(f"HF API error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            logger.warning(f"HF embedding call failed (attempt {attempt+1}): {e}")
            time.sleep(5)
    return None


def embed_dataframe_jobbert(df: pd.DataFrame,
                             text_col: str = "description_clean",
                             batch_size: int = 8) -> np.ndarray:
    texts = df[text_col].fillna("").astype(str).tolist()
    n = len(texts)
    result = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

    for i, text in enumerate(texts):
        chunks = _chunk_text(text)
        chunk_embeddings = []

        for j in range(0, len(chunks), batch_size):
            batch = chunks[j: j + batch_size]
            arr = _call_hf_embedding(HF_API_URL_JOBBERT, batch)
            if arr is not None and arr.shape[0] == len(batch):
                chunk_embeddings.append(arr)
            else:
                logger.warning(f"JobBERT row {i} chunk {j} failed, using zero.")
                chunk_embeddings.append(np.zeros((len(batch), EMBEDDING_DIM), dtype=np.float32))
            time.sleep(0.3)

        if chunk_embeddings:
            all_chunks = np.concatenate(chunk_embeddings, axis=0)
            weights = np.array([len(c) for c in chunks[:len(all_chunks)]], dtype=np.float32)
            weights = weights / weights.sum()
            result[i] = np.average(all_chunks, axis=0, weights=weights)

        if (i + 1) % 10 == 0:
            logger.info(f"JobBERT: {i+1}/{n}")

    logger.info(f"JobBERT embeddings done: shape={result.shape}")
    return result


def embed_dataframe_sbert(df: pd.DataFrame,
                           text_col: str = "description_clean",
                           batch_size: int = 16) -> np.ndarray:
    texts = df[text_col].fillna("").astype(str).tolist()
    n = len(texts)
    result = np.zeros((n, SBERT_DIM), dtype=np.float32)

    for i in range(0, n, batch_size):
        batch = [t[:512] for t in texts[i: i + batch_size]]
        arr = _call_hf_embedding(HF_API_URL_SBERT, batch)
        if arr is not None and arr.shape[0] == len(batch):
            result[i: i + len(batch)] = arr
        else:
            logger.warning(f"SBERT batch {i} failed or shape mismatch, using zero vectors.")
        if (i // batch_size) % 5 == 0:
            logger.info(f"SBERT: {min(i + batch_size, n)}/{n}")
        time.sleep(0.5)

    logger.info(f"SBERT embeddings done: shape={result.shape}")
    return result


def compute_and_save_embeddings(preprocessed_path: str) -> Path:
    if not HF_API_TOKEN:
        logger.warning("HF_API_TOKEN tidak di-set. Embedding dilewati, menulis zero vectors.")
        out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_embeddings.npz")
        np.savez_compressed(
            out_path,
            source_hashes=np.array([], dtype=str),
            jobbert=np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
            sbert=np.zeros((0, SBERT_DIM), dtype=np.float32),
        )
        return out_path

    try:
        df = pd.read_parquet(preprocessed_path)
        df = df.reset_index(drop=True)

        out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_embeddings.npz")

        if df.empty:
            logger.warning(f"Empty dataframe at {preprocessed_path}, skipping embeddings.")
            np.savez_compressed(
                out_path,
                source_hashes=np.array([], dtype=str),
                jobbert=np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
                sbert=np.zeros((0, SBERT_DIM), dtype=np.float32),
            )
            return out_path

        max_rows = 500
        if len(df) > max_rows:
            logger.info(f"Large dataset: {len(df)} rows. Limiting to {max_rows}.")
            df = df.iloc[:max_rows].copy()

        logger.info(f"Computing embeddings for {len(df)} rows via HF API")

        if "description_clean" not in df.columns or df["description_clean"].isna().all():
            logger.warning("description_clean missing or all null, using empty strings.")
            df["description_clean"] = ""

        jobbert_embs = embed_dataframe_jobbert(df)
        gc.collect()

        sbert_embs = embed_dataframe_sbert(df)
        gc.collect()

        np.savez_compressed(
            out_path,
            source_hashes=df["source_hash"].astype(str).values,
            jobbert=jobbert_embs,
            sbert=sbert_embs,
        )
        logger.info(f"Embeddings saved: {out_path} "
                    f"(jobbert={jobbert_embs.shape}, sbert={sbert_embs.shape})")
        return out_path

    except Exception as e:
        logger.error(f"ERROR in compute_and_save_embeddings: {e}")
        raise
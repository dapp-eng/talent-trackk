import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from config import (
    JOBBERT_MODEL, SBERT_MODEL,
    JOBBERT_MAX_TOKENS, JOBBERT_CHUNK_OVERLAP,
    EMBEDDING_DIM, SBERT_DIM,
    DATA_PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

_jobbert_tokenizer = None
_jobbert_model = None
_sbert_model = None

CONTENT_TOKENS = JOBBERT_MAX_TOKENS - 2
STRIDE = CONTENT_TOKENS - JOBBERT_CHUNK_OVERLAP

assert CONTENT_TOKENS == 510, f"Expected 510 content tokens, got {CONTENT_TOKENS}"


def _load_jobbert():
    global _jobbert_tokenizer, _jobbert_model
    if _jobbert_tokenizer is None:
        from transformers import AutoTokenizer, AutoModel
        logger.info(f"Loading JobBERT: {JOBBERT_MODEL}")
        _jobbert_tokenizer = AutoTokenizer.from_pretrained(JOBBERT_MODEL)
        _jobbert_model = AutoModel.from_pretrained(JOBBERT_MODEL)
        _jobbert_model.eval()
        if torch.cuda.is_available():
            _jobbert_model = _jobbert_model.cuda()
            logger.info("JobBERT loaded on GPU")
        else:
            logger.info("JobBERT loaded on CPU")
    return _jobbert_tokenizer, _jobbert_model


def _load_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SBERT: {SBERT_MODEL}")
        _sbert_model = SentenceTransformer(SBERT_MODEL)
    return _sbert_model


def _mean_pool(last_hidden_state: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def _embed_text_jobbert_chunked(text: str, tokenizer, model) -> np.ndarray:
    if not text or not isinstance(text, str) or not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    all_input_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(all_input_ids) == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    chunk_embeddings = []
    start = 0

    while start < len(all_input_ids):
        end = start + STRIDE
        chunk_ids = all_input_ids[start:end]

        input_ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
        pad_len = JOBBERT_MAX_TOKENS - len(input_ids)

        if pad_len < 0:
            input_ids = input_ids[:JOBBERT_MAX_TOKENS]
            pad_len = 0

        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len

        input_ids_t = torch.tensor([input_ids])
        attention_mask_t = torch.tensor([attention_mask])

        if torch.cuda.is_available():
            input_ids_t = input_ids_t.cuda()
            attention_mask_t = attention_mask_t.cuda()

        with torch.no_grad():
            output = model(input_ids=input_ids_t, attention_mask=attention_mask_t)

        chunk_emb = _mean_pool(output.last_hidden_state, attention_mask_t)
        chunk_embeddings.append(chunk_emb.cpu().float().squeeze(0).numpy())

        if end >= len(all_input_ids):
            break

        start += STRIDE

    if not chunk_embeddings:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    token_counts = []
    start = 0
    for _ in chunk_embeddings:
        end = start + STRIDE
        token_counts.append(min(STRIDE, len(all_input_ids) - start))
        start += STRIDE

    weights = np.array(token_counts, dtype=np.float32)
    weights = weights / weights.sum()

    stacked = np.stack(chunk_embeddings, axis=0)
    final = np.average(stacked, axis=0, weights=weights)
    return final.astype(np.float32)


def embed_dataframe_jobbert(df: pd.DataFrame,
                             text_col: str = "description_clean",
                             batch_size: int = 16) -> np.ndarray:
    tokenizer, model = _load_jobbert()
    texts = df[text_col].fillna("").astype(str).tolist()
    embeddings = []

    for i, text in enumerate(texts):
        emb = _embed_text_jobbert_chunked(text, tokenizer, model)
        embeddings.append(emb)
        if (i + 1) % 50 == 0:
            logger.info(f"  JobBERT: {i + 1}/{len(texts)}")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = np.stack(embeddings, axis=0)
    logger.info(f"JobBERT embeddings done: shape={result.shape}")
    return result


def embed_dataframe_sbert(df: pd.DataFrame,
                           text_col: str = "description_clean",
                           batch_size: int = 64) -> np.ndarray:
    sbert = _load_sbert()
    texts = df[text_col].fillna("").astype(str).tolist()
    embeddings = sbert.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    result = embeddings.astype(np.float32)
    logger.info(f"SBERT embeddings done: shape={result.shape}")
    return result


def compute_and_save_embeddings(preprocessed_path: str) -> Path:
    try:
        df = pd.read_parquet(preprocessed_path)
        df = df.reset_index(drop=True)

        if df.empty:
            logger.warning(f"Empty dataframe at {preprocessed_path}, skipping embeddings.")
            out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_embeddings.npz")
            np.savez_compressed(
                out_path,
                source_hashes=np.array([], dtype=str),
                jobbert=np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
                sbert=np.zeros((0, SBERT_DIM), dtype=np.float32),
            )
            return out_path

        max_rows = 3000
        if len(df) > max_rows:
            logger.info(f"Large dataset detected: {len(df)} rows. Limiting to {max_rows} for stability.")
            df = df.iloc[:max_rows].copy()

        logger.info(f"Computing embeddings for {len(df)} rows from {preprocessed_path}")

        if "description_clean" not in df.columns or df["description_clean"].isna().all():
            logger.warning("description_clean column missing or all null, using empty strings.")
            df["description_clean"] = ""

        jobbert_embs = embed_dataframe_jobbert(df)
        sbert_embs = embed_dataframe_sbert(df)

        out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_embeddings.npz")
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

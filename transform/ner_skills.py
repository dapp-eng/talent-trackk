import re
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_PROCESSED_DIR, HF_API_TOKEN

logger = logging.getLogger(__name__)

NER_KNOWLEDGE_MODEL = "jjzha/jobbert_knowledge_extraction"
NER_SKILL_MODEL = "jjzha/jobbert_skill_extraction"

HF_API_URL_KNOWLEDGE = f"https://api-inference.huggingface.co/models/{NER_KNOWLEDGE_MODEL}"
HF_API_URL_SKILL = f"https://api-inference.huggingface.co/models/{NER_SKILL_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def _normalize_entity(word: str) -> str:
    word = re.sub(r"\s+", " ", word).strip()
    word = re.sub(r"^[^\w]+|[^\w]+$", "", word)
    return word


def _call_hf_api(url: str, text: str, retries: int = 3) -> list:
    payload = {"inputs": text[:512], "options": {"wait_for_model": True}}
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=HF_HEADERS, json=payload, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list):
                    return result
                return []
            elif resp.status_code == 503:
                wait = int(resp.json().get("estimated_time", 20))
                logger.warning(f"HF model loading, waiting {wait}s...")
                time.sleep(min(wait, 30))
            elif resp.status_code == 429:
                logger.warning(f"HF rate limit, waiting 10s (attempt {attempt+1})")
                time.sleep(10)
            else:
                logger.warning(f"HF API error {resp.status_code}: {resp.text[:200]}")
                return []
        except Exception as e:
            logger.warning(f"HF API call failed (attempt {attempt+1}): {e}")
            time.sleep(5)
    return []


def _parse_hf_response(entities: list, source_model: str) -> list:
    is_knowledge = "knowledge" in source_model.lower()
    entity_type_label = "Knowledge" if is_knowledge else "Skill"

    merged = []
    current_tokens = []
    current_scores = []

    for ent in entities:
        raw_label = (ent.get("entity_group") or ent.get("entity") or "").upper()
        raw_label = re.sub(r"^(B-|I-|B_|I_)", "", raw_label).strip()
        word = ent.get("word", "").strip()
        score = float(ent.get("score", 0.0))

        if raw_label == "B":
            if current_tokens:
                merged.append((current_tokens, current_scores))
            current_tokens = [word]
            current_scores = [score]
        elif raw_label == "I" and current_tokens:
            current_tokens.append(word)
            current_scores.append(score)

    if current_tokens:
        merged.append((current_tokens, current_scores))

    results = []
    seen = set()
    for tokens, scores in merged:
        word = re.sub(r"\s*##", "", " ".join(tokens)).strip()
        word = re.sub(r"\s+", " ", word)
        word = _normalize_entity(word)
        if not word or len(word) < 3:
            continue
        if re.fullmatch(r"[\W\d]+", word):
            continue
        avg_score = round(sum(scores) / len(scores), 4)
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "entity_text": word,
            "entity_type": entity_type_label,
            "source_model": source_model,
            "extraction_confidence": avg_score,
        })
    return results


def extract_entities_from_text(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []

    results = []
    for url, model in [
        (HF_API_URL_KNOWLEDGE, NER_KNOWLEDGE_MODEL),
        (HF_API_URL_SKILL, NER_SKILL_MODEL),
    ]:
        raw = _call_hf_api(url, text)
        results.extend(_parse_hf_response(raw, model))

    seen_final = set()
    deduped = []
    for r in results:
        key = (r["entity_text"].lower(), r["entity_type"])
        if key not in seen_final:
            seen_final.add(key)
            deduped.append(r)
    return deduped


def extract_entities_dataframe(
    df: pd.DataFrame,
    text_col: str = "description_clean",
    batch_size: int = 32,
) -> pd.DataFrame:
    if not HF_API_TOKEN:
        logger.warning("HF_API_TOKEN tidak di-set. NER dilewati.")
        return pd.DataFrame(columns=[
            "source_hash", "entity_text", "entity_type",
            "source_model", "extraction_confidence",
        ])

    texts = df[text_col].fillna("").astype(str).tolist()
    src_hashes = (
        df["source_hash"].astype(str).tolist()
        if "source_hash" in df.columns
        else [str(i) for i in df.index]
    )

    records = []
    for i, (text, src_hash) in enumerate(zip(texts, src_hashes)):
        entities = extract_entities_from_text(text)
        for e in entities:
            records.append({
                "source_hash": src_hash,
                "entity_text": e["entity_text"],
                "entity_type": e["entity_type"],
                "source_model": e["source_model"],
                "extraction_confidence": e["extraction_confidence"],
            })
        if (i + 1) % 10 == 0:
            logger.info(f"NER progress: {i+1}/{len(texts)}")

    return pd.DataFrame(records)


def run_ner(preprocessed_path: str) -> Path:
    df = pd.read_parquet(preprocessed_path)
    logger.info(f"Running NER on {len(df)} rows via HF API")
    entities_df = extract_entities_dataframe(df)
    out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_entities.parquet")
    entities_df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"NER done: {len(entities_df)} entity records → {out_path}")
    return out_path
import re
import logging
import pandas as pd
from pathlib import Path
from config import DATA_PROCESSED_DIR, NER_AGGREGATION

logger = logging.getLogger(__name__)

NER_KNOWLEDGE_MODEL = "jjzha/jobbert_knowledge_extraction"
NER_SKILL_MODEL = "jjzha/jobbert_skill_extraction"

_pipeline_knowledge = None
_pipeline_skill = None


def _load_pipeline(model_id: str):
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            pipeline,
        )
        logger.info(f"Loading NER model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=NER_AGGREGATION,
            device=-1,
        )
        logger.info(f"NER pipeline loaded: {model_id}")
        return pipe
    except Exception as e:
        logger.error(f"Failed to load NER model '{model_id}': {e}")
        return None


def _get_pipeline_knowledge():
    global _pipeline_knowledge
    if _pipeline_knowledge is None:
        _pipeline_knowledge = _load_pipeline(NER_KNOWLEDGE_MODEL)
    return _pipeline_knowledge


def _get_pipeline_skill():
    global _pipeline_skill
    if _pipeline_skill is None:
        _pipeline_skill = _load_pipeline(NER_SKILL_MODEL)
    return _pipeline_skill


def _normalize_entity(word: str) -> str:
    word = re.sub(r"\s+", " ", word).strip()
    word = re.sub(r"^[^\w]+|[^\w]+$", "", word)
    return word


def _run_pipeline(pipe, text: str, source_model: str) -> list:
    if pipe is None or not isinstance(text, str) or not text.strip():
        return []
    try:
        entities = pipe(text[:2000])
    except Exception as e:
        logger.warning(f"NER inference failed ({source_model}): {e}")
        return []

    is_knowledge = "knowledge" in source_model.lower()
    entity_type_label = "Knowledge" if is_knowledge else "Skill"

    merged = []
    current_tokens = []
    current_score = []

    for ent in entities:
        raw_label = (ent.get("entity_group") or ent.get("entity") or "").upper()
        raw_label = re.sub(r"^(B-|I-|B_|I_)", "", raw_label).strip()
        word = ent.get("word", "").strip()
        score = float(ent.get("score", 0.0))

        if raw_label == "B" or (raw_label not in ("B", "I") and word):
            if current_tokens:
                merged.append((current_tokens, current_score))
            current_tokens = [word]
            current_score = [score]
        elif raw_label == "I" and current_tokens:
            current_tokens.append(word)
            current_score.append(score)

    if current_tokens:
        merged.append((current_tokens, current_score))

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
    results = []
    results.extend(_run_pipeline(_get_pipeline_knowledge(), text, NER_KNOWLEDGE_MODEL))
    results.extend(_run_pipeline(_get_pipeline_skill(),     text, NER_SKILL_MODEL))
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
    pipe_k = _get_pipeline_knowledge()
    pipe_s = _get_pipeline_skill()

    if pipe_k is None and pipe_s is None:
        logger.warning("Both NER pipelines unavailable. Returning empty DataFrame.")
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
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start: batch_start + batch_size]
        batch_hashes = src_hashes[batch_start: batch_start + batch_size]
        for text, src_hash in zip(batch_texts, batch_hashes):
            entities = extract_entities_from_text(text)
            for e in entities:
                records.append({
                    "source_hash": src_hash,
                    "entity_text": e["entity_text"],
                    "entity_type": e["entity_type"],
                    "source_model": e["source_model"],
                    "extraction_confidence": e["extraction_confidence"],
                })
        if (batch_start // batch_size) % 10 == 0:
            logger.info(
                f"NER progress: {min(batch_start + batch_size, len(texts))}/{len(texts)}"
            )

    return pd.DataFrame(records)


def run_ner(preprocessed_path: str) -> Path:
    df = pd.read_parquet(preprocessed_path)
    logger.info(f"Running NER on {len(df)} rows (knowledge + skill models)")
    entities_df = extract_entities_dataframe(df)
    out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_entities.parquet")
    entities_df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"NER done: {len(entities_df)} entity records → {out_path}")
    return out_path
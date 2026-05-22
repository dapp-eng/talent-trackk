import gc
import re
import logging
import pandas as pd
from pathlib import Path
from config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

GLINER_MODEL = "urchade/gliner_small-v2.1"
GLINER_LABELS = [
    "technical skill",
    "soft skill",
    "programming language",
    "tool",
    "framework",
    "certification",
    "knowledge",
]
LABEL_TO_TYPE = {
    "technical skill": "Skill",
    "soft skill": "Skill",
    "programming language": "Skill",
    "tool": "Knowledge",
    "framework": "Knowledge",
    "certification": "Knowledge",
    "knowledge": "Knowledge",
}


def _normalize_entity(word: str) -> str:
    word = re.sub(r"\s+", " ", word).strip()
    word = re.sub(r"^[^\w]+|[^\w]+$", "", word)
    return word


def _load_gliner():
    try:
        from gliner import GLiNER
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        logger.warning(f"Loading GliNER: {GLINER_MODEL}")
        model = GLiNER.from_pretrained(GLINER_MODEL)
        logger.warning("GliNER loaded ok")
        return model
    except Exception as e:
        logger.warning(f"Failed to load GliNER: {e}")
        return None


def _run_gliner(model, texts: list, src_hashes: list) -> list:
    records = []
    try:
        for i, (text, src_hash) in enumerate(zip(texts, src_hashes)):
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                entities = model.predict_entities(text[:1000], GLINER_LABELS, threshold=0.5)
                seen = set()
                for ent in entities:
                    word = _normalize_entity(ent.get("text", ""))
                    if not word or len(word) < 2:
                        continue
                    if re.fullmatch(r"[\W\d]+", word):
                        continue
                    label = ent.get("label", "")
                    entity_type = LABEL_TO_TYPE.get(label, "Skill")
                    score = round(float(ent.get("score", 0.0)), 4)
                    key = word.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append({
                        "source_hash": src_hash,
                        "entity_text": word,
                        "entity_type": entity_type,
                        "source_model": GLINER_MODEL,
                        "extraction_confidence": score,
                    })
            except Exception as e:
                logger.warning(f"GliNER inference failed row {i}: {e}")
            if (i + 1) % 10 == 0:
                logger.warning(f"NER progress: {i+1}/{len(texts)}")
    finally:
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return records


def extract_entities_dataframe(
    df: pd.DataFrame,
    text_col: str = "description_clean",
) -> pd.DataFrame:
    texts = df[text_col].fillna("").astype(str).tolist()
    src_hashes = (
        df["source_hash"].astype(str).tolist()
        if "source_hash" in df.columns
        else [str(i) for i in df.index]
    )

    model = _load_gliner()
    if model is None:
        logger.warning("GliNER unavailable, returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "source_hash", "entity_text", "entity_type",
            "source_model", "extraction_confidence",
        ])

    records = _run_gliner(model, texts, src_hashes)

    seen = set()
    deduped = []
    for r in records:
        key = (r["source_hash"], r["entity_text"].lower(), r["entity_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return pd.DataFrame(deduped) if deduped else pd.DataFrame(columns=[
        "source_hash", "entity_text", "entity_type",
        "source_model", "extraction_confidence",
    ])


def run_ner(preprocessed_path: str) -> Path:
    df = pd.read_parquet(preprocessed_path)
    logger.warning(f"Running NER on {len(df)} rows via GliNER")
    entities_df = extract_entities_dataframe(df)
    out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_entities.parquet")
    entities_df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.warning(f"NER done: {len(entities_df)} entity records → {out_path}")
    return out_path
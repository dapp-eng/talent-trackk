import gc
import re
import logging
import pandas as pd
from pathlib import Path
from config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

GLINER_MODEL = "urchade/gliner_small-v2.1"
GLINER_LABELS = [
    "programming language",
    "framework",
    "library",
    "database",
    "cloud platform",
    "cloud service",
    "devops tool",
    "data tool",
    "machine learning tool",
    "operating system",
    "version control",
    "api",
    "protocol",
    "messaging system",
    "containerization tool",
    "orchestration tool",
    "monitoring tool",
    "testing tool",
    "build tool",
    "ide",
    "technical skill",
    "soft skill",
    "methodology",
    "certification",
    "security tool",
    "data format",
    "visualization tool",
    "web technology",
    "networking tool",
    "hardware",
    "embedded system",
    "blockchain platform",
    "erp system",
    "crm system",
    "tool",
    "knowledge",
]
LABEL_TO_TYPE = {
    "programming language": "Skill",
    "framework": "Skill",
    "library": "Skill",
    "database": "Knowledge",
    "cloud platform": "Knowledge",
    "cloud service": "Knowledge",
    "devops tool": "Knowledge",
    "data tool": "Knowledge",
    "machine learning tool": "Knowledge",
    "operating system": "Knowledge",
    "version control": "Knowledge",
    "api": "Knowledge",
    "protocol": "Knowledge",
    "messaging system": "Knowledge",
    "containerization tool": "Knowledge",
    "orchestration tool": "Knowledge",
    "monitoring tool": "Knowledge",
    "testing tool": "Knowledge",
    "build tool": "Knowledge",
    "ide": "Knowledge",
    "technical skill": "Skill",
    "soft skill": "Skill",
    "methodology": "Knowledge",
    "certification": "Knowledge",
    "security tool": "Knowledge",
    "data format": "Knowledge",
    "visualization tool": "Knowledge",
    "web technology": "Knowledge",
    "networking tool": "Knowledge",
    "hardware": "Knowledge",
    "embedded system": "Knowledge",
    "blockchain platform": "Knowledge",
    "erp system": "Knowledge",
    "crm system": "Knowledge",
    "tool": "Knowledge",
    "knowledge": "Knowledge",
}

GLINER_THRESHOLD = 0.35
GLINER_THRESHOLD_NON_EN = 0.25
ENGLISH_LANGS = {"en"}
FALLBACK_CHUNK_SIZE = 1300

SECTION_PATTERNS = re.compile(
    r"(?:^|\n)"
    r"(?P<header>"
    r"requirements?|required skills?|minimum qualifications?|basic qualifications?"
    r"|preferred qualifications?|technical (skills?|requirements?|qualifications?)"
    r"|tech(nical)? stack|technologies|tools?\s*(&|and)\s*technologies"
    r"|what (you('ll|will) need|we('re| are) looking for|you bring|you have)"
    r"|skills?( required| needed| &amp; experience)?"
    r"|qualifications?|experience( required| needed)?"
    r"|must.?have|nice.?to.?have|good.?to.?have"
    r"|kualifikasi|persyaratan|keahlian|kemampuan|teknologi|stack teknologi"
    r"|kompeten[cs]i|syarat"
    r")"
    r"\s*:?\s*\n",
    re.IGNORECASE | re.MULTILINE,
)

ANY_HEADER_PATTERN = re.compile(
    r"\n(?=[A-Z][A-Za-z\s&]{2,40}:?\s*\n)",
    re.MULTILINE,
)


def _extract_skill_sections(text: str) -> str:
    matches = list(SECTION_PATTERNS.finditer(text))
    if not matches:
        return ""

    all_headers = list(ANY_HEADER_PATTERN.finditer(text))
    header_positions = sorted({m.start() for m in all_headers} | {m.start() for m in matches})

    sections = []
    for match in matches:
        start = match.end()
        next_boundary = next(
            (pos for pos in header_positions if pos > match.start()),
            len(text),
        )
        section_text = text[start:next_boundary].strip()
        if section_text:
            sections.append(section_text)

    return "\n".join(sections)


def _get_ner_input(text: str) -> str:
    extracted = _extract_skill_sections(text)
    if extracted and len(extracted) >= 100:
        return extracted
    return text[:FALLBACK_CHUNK_SIZE]


def _normalize_entity(word: str) -> str:
    word = re.sub(r"\s+", " ", word).strip()
    word = re.sub(r"^[^\w]+|[^\w]+$", "", word)
    return word.lower()


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


def _run_gliner(model, texts: list, src_hashes: list, langs: list) -> list:
    records = []
    section_hits = 0
    fallback_hits = 0
    try:
        for i, (text, src_hash, lang) in enumerate(zip(texts, src_hashes, langs)):
            if not isinstance(text, str) or not text.strip():
                continue
            threshold = GLINER_THRESHOLD if lang in ENGLISH_LANGS else GLINER_THRESHOLD_NON_EN

            extracted = _extract_skill_sections(text)
            if extracted and len(extracted) >= 100:
                ner_input = extracted
                section_hits += 1
            else:
                ner_input = text[:FALLBACK_CHUNK_SIZE]
                fallback_hits += 1

            seen = set()
            try:
                entities = model.predict_entities(ner_input, GLINER_LABELS, threshold=threshold)
                for ent in entities:
                    word = _normalize_entity(ent.get("text", ""))
                    if not word or len(word) < 1:
                        continue
                    if re.fullmatch(r"[\W\d]+", word):
                        continue
                    label = ent.get("label", "")
                    entity_type = LABEL_TO_TYPE.get(label, "Skill")
                    score = round(float(ent.get("score", 0.0)), 4)
                    if word in seen:
                        continue
                    seen.add(word)
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
                logger.warning(f"NER progress: {i+1}/{len(texts)} (section={section_hits}, fallback={fallback_hits})")
    finally:
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    logger.warning(f"NER extraction complete: section_hit={section_hits}, fallback={fallback_hits}")
    return records


def extract_entities_dataframe(
    df: pd.DataFrame,
    text_col: str = "description_ner",
) -> pd.DataFrame:
    col = text_col if text_col in df.columns else "description_clean"
    texts = df[col].fillna("").astype(str).tolist()
    src_hashes = (
        df["source_hash"].astype(str).tolist()
        if "source_hash" in df.columns
        else [str(i) for i in df.index]
    )
    langs = (
        df["description_lang"].astype(str).tolist()
        if "description_lang" in df.columns
        else ["en"] * len(df)
    )

    model = _load_gliner()
    if model is None:
        logger.warning("GliNER unavailable, returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "source_hash", "entity_text", "entity_type",
            "source_model", "extraction_confidence",
        ])

    records = _run_gliner(model, texts, src_hashes, langs)

    seen = set()
    deduped = []
    for r in records:
        key = (r["source_hash"], r["entity_text"], r["entity_type"])
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

    if "description_lang" in df.columns:
        lang_counts = df["description_lang"].value_counts().to_dict()
        logger.warning(f"  Input language distribution: {lang_counts}")
        non_en = {k: v for k, v in lang_counts.items() if k not in ENGLISH_LANGS and k != "unknown"}
        if non_en:
            logger.warning(f"  Non-English rows (threshold={GLINER_THRESHOLD_NON_EN}): {non_en}")

    entities_df = extract_entities_dataframe(df)
    out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_entities.parquet")
    entities_df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.warning(f"NER done: {len(entities_df)} entity records → {out_path}")
    return out_path
import re
import hashlib
import pandas as pd
from pathlib import Path
from config import DATA_PROCESSED_DIR


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _content_hash(row: pd.Series) -> str:
    title = _clean_text(str(row.get("title_clean") or row.get("title") or ""))
    company = _clean_text(str(row.get("company_clean") or row.get("company") or ""))
    description = _clean_text(str(row.get("description_clean") or row.get("description") or ""))
    content = f"{title}|{company}|{description[:500]}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _strict_dedup_key(row: pd.Series) -> str:
    title = _clean_text(str(row.get("title_clean") or row.get("title") or ""))
    company = _clean_text(str(row.get("company_clean") or row.get("company") or ""))
    location = _clean_text(str(row.get("loc_city") or row.get("location") or ""))
    date = str(row.get("date_parsed") or row.get("date_posted") or "").strip()

    if not any([title, company, location, date]):
        return str(row.get("source_hash", hashlib.md5(str(row.values).encode()).hexdigest()))

    return hashlib.md5(f"{title}|{company}|{location}|{date}".encode("utf-8")).hexdigest()


def dedup_against_existing_hashes(df: pd.DataFrame, existing_hashes: set) -> pd.DataFrame:
    before = len(df)
    df = df[~df["source_hash"].isin(existing_hashes)]
    after = len(df)
    print(f"Dedup against existing: removed {before - after}, kept {after}")
    return df


def dedup_within_batch(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df["_strict_key"] = df.apply(_strict_dedup_key, axis=1)
    df = df.drop_duplicates(subset=["_strict_key"])
    df = df.drop(columns=["_strict_key"])
    after = len(df)
    print(f"Dedup within batch: removed {before - after}, kept {after}")
    return df


def run_dedup(preprocessed_path: str, embeddings_path: str = None,
               existing_hashes: set = None) -> Path:
    df = pd.read_parquet(preprocessed_path)
    print(f"Dedup input: {len(df)} rows")

    if existing_hashes:
        df = dedup_against_existing_hashes(df, existing_hashes)

    if len(df) > 1:
        df = dedup_within_batch(df)

    out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_deduped.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"Dedup output: {len(df)} rows → {out_path}")
    return out_path
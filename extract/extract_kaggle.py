import json
import hashlib
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import KAGGLE_DATASET_PATH, DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

KAGGLE_BATCH_LIMIT = 150

EXPECTED_COLUMNS = {
    "title": ["job_title", "title", "position", "job title"],
    "company": ["company", "company_name", "employer"],
    "location": ["location", "job_location", "city"],
    "description": ["description", "job_description", "details"],
    "date_posted": ["date_posted", "posted_date", "post_date", "date"],
    "salary_min": ["salary_min", "min_salary", "salary_from"],
    "salary_max": ["salary_max", "max_salary", "salary_to"],
    "is_remote": ["is_remote", "remote", "work_from_home"],
    "platform": ["platform", "source", "site"],
    "employment_type": ["employment_type", "job_type", "type"],
}

EMPTY_COLS = [
    "title", "company", "location", "description", "date_posted",
    "salary_min", "salary_max", "is_remote", "platform",
    "employment_type", "source_hash", "extraction_ts", "data_source",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in EXPECTED_COLUMNS.items():
        for alias in aliases:
            if alias.lower() in df_cols_lower:
                col_map[df_cols_lower[alias.lower()]] = canonical
                break
    df = df.rename(columns=col_map)
    for canonical in EXPECTED_COLUMNS:
        if canonical not in df.columns:
            df[canonical] = None
    return df


def _make_hash(row: pd.Series) -> str:
    company = str(row.get("company") or "").strip().lower()
    title   = str(row.get("title")   or "").strip().lower()
    date    = str(row.get("date_posted") or "").strip()
    return hashlib.md5(f"{company}|{title}|{date}".encode("utf-8")).hexdigest()


def extract_kaggle(path: str = None) -> Path:
    try:
        src = Path(path) if path else KAGGLE_DATASET_PATH
        if not src.exists():
            raise FileNotFoundError(
                f"Kaggle dataset not found at {src}. "
                "Upload it via VSCode to data/raw/kaggle_jobs_2024.csv"
            )

        out_path = DATA_PROCESSED_DIR / "kaggle_staged.parquet"

        seen_hashes: set = set()
        output_chunks: list = []

        reader = pd.read_csv(src, chunksize=50, low_memory=False, on_bad_lines="skip")

        for i, chunk in enumerate(reader):
            chunk = _normalize_columns(chunk)
            chunk["source_hash"] = chunk.apply(_make_hash, axis=1)
            chunk["extraction_ts"] = datetime.utcnow().isoformat()
            chunk["data_source"] = "kaggle_2024"

            new_rows = chunk[~chunk["source_hash"].isin(seen_hashes)].copy()
            if new_rows.empty:
                continue

            collected = sum(len(c) for c in output_chunks)
            remaining = KAGGLE_BATCH_LIMIT - collected
            if remaining <= 0:
                break

            new_rows = new_rows.head(remaining)
            seen_hashes.update(new_rows["source_hash"].values)
            output_chunks.append(new_rows)

            collected = sum(len(c) for c in output_chunks)
            logger.info(f"Kaggle chunk {i + 1}: +{len(new_rows)} rows (total: {collected})")

            if collected >= KAGGLE_BATCH_LIMIT:
                break

        if not output_chunks:
            logger.warning("Kaggle extraction: no rows collected.")
            df = pd.DataFrame(columns=EMPTY_COLS)
        else:
            df = pd.concat(output_chunks, ignore_index=True)

        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"Kaggle extraction done: {len(df)} rows → {out_path}")

        meta = {
            "source": "kaggle_2024",
            "rows": len(df),
            "target_limit": KAGGLE_BATCH_LIMIT,
            "extracted_at": datetime.utcnow().isoformat(),
            "output": str(out_path),
        }
        (DATA_PROCESSED_DIR / "kaggle_staged_meta.json").write_text(
            json.dumps(meta, indent=2)
        )

        return out_path

    except Exception as e:
        logger.error(f"ERROR in extract_kaggle: {e}")
        raise
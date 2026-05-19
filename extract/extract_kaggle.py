import pandas as pd
import hashlib
import json
from pathlib import Path
from datetime import datetime
from config import KAGGLE_DATASET_PATH, DATA_PROCESSED_DIR


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
    raw = f"{row.get('company','')}-{row.get('title','')}-{row.get('date_posted','')}"
    return hashlib.md5(raw.encode()).hexdigest()


def extract_kaggle(path: str = None, chunksize: int = 100) -> Path:
    try:
        src = Path(path) if path else KAGGLE_DATASET_PATH
        if not src.exists():
            raise FileNotFoundError(f"Kaggle dataset not found at {src}. "
                                    "Upload it via VSCode to data/raw/kaggle_jobs_2024.csv")

        out_path = DATA_PROCESSED_DIR / "kaggle_staged.parquet"
        seen_hashes = set()
        output_chunks = []
        chunk_buffer_size = 500
        reader = pd.read_csv(src, chunksize=chunksize, low_memory=False, on_bad_lines="skip")

        for i, chunk in enumerate(reader):
            chunk = _normalize_columns(chunk)
            chunk["source_hash"] = chunk.apply(_make_hash, axis=1)
            chunk["extraction_ts"] = datetime.utcnow().isoformat()
            chunk["data_source"] = "kaggle_2024"
            
            chunk = chunk[~chunk["source_hash"].isin(seen_hashes)]
            if len(chunk) > 0:
                seen_hashes.update(chunk["source_hash"].values)
                output_chunks.append(chunk)
            
            if (i + 1) % 10 == 0:
                print(f"  Kaggle processed chunk {i+1}: {len(chunk)} new rows, total unique: {len(seen_hashes)}")
            
            if len(output_chunks) >= chunk_buffer_size:
                break

        if not output_chunks:
            df = pd.DataFrame(columns=["title", "company", "location", "description", "date_posted",
                                       "salary_min", "salary_max", "is_remote", "platform", 
                                       "employment_type", "source_hash", "extraction_ts", "data_source"])
        else:
            df = pd.concat(output_chunks, ignore_index=True)

        df.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Kaggle extraction done: {len(df)} rows → {out_path}")

        meta = {
            "source": "kaggle_2024",
            "rows": len(df),
            "extracted_at": datetime.utcnow().isoformat(),
            "output": str(out_path),
        }
        meta_path = DATA_PROCESSED_DIR / "kaggle_staged_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        return out_path
    except Exception as e:
        print(f"ERROR in extract_kaggle: {e}")
        raise

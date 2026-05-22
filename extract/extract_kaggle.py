import json
import hashlib
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import KAGGLE_DATASET_PATH, DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

KAGGLE_BATCH_LIMIT = 150

LINKEDIN_COLUMN_MAP = {
    "title": ["title", "job_title", "position", "job title"],
    "company": ["company_name", "company", "employer", "organization"],
    "location": ["location", "job_location", "formatted_location", "city"],
    "description": ["description", "job_description", "details", "body"],
    "date_posted": ["listed_time", "original_listed_time", "date_posted",
                    "posted_date", "post_date", "date"],
    "salary_min": ["min_salary", "salary_min", "normalized_salary",
                        "compensation_min", "salary_from"],
    "salary_max": ["max_salary", "salary_max", "compensation_max", "salary_to"],
    "is_remote": ["remote_allowed", "is_remote", "remote", "work_from_home"],
    "platform": ["source", "platform", "site", "job_board"],
    "employment_type": ["formatted_work_type", "employment_type", "job_type",
                        "work_type", "type"],
}

EMPTY_COLS = [
    "title", "company", "location", "description", "date_posted",
    "salary_min", "salary_max", "is_remote", "platform",
    "employment_type", "source_hash", "extraction_ts", "data_source",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in LINKEDIN_COLUMN_MAP.items():
        for alias in aliases:
            if alias.lower() in df_cols_lower:
                col_map[df_cols_lower[alias.lower()]] = canonical
                break
    df = df.rename(columns=col_map)
    for canonical in LINKEDIN_COLUMN_MAP:
        if canonical not in df.columns:
            df[canonical] = None
    return df


def _parse_listed_time(val) -> str:
    if val is None:
        return None
    try:
        v = float(val)
        if v > 1e10:
            return pd.Timestamp(v, unit="ms").strftime("%Y-%m-%d")
        return pd.Timestamp(v, unit="s").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(val).strip() if val else None


def _make_hash(row: pd.Series) -> str:
    company = str(row.get("company") or "").strip().lower()
    title = str(row.get("title")   or "").strip().lower()
    date = str(row.get("date_posted") or "").strip()
    return hashlib.md5(f"{company}|{title}|{date}".encode("utf-8")).hexdigest()


def _resolve_company_names(df: pd.DataFrame, src_dir: Path) -> pd.DataFrame:
    companies_path = src_dir / "companies" / "companies.csv"
    if not companies_path.exists():
        companies_path = src_dir / "companies.csv"

    if companies_path.exists():
        try:
            comp_df = pd.read_csv(companies_path, usecols=["company_id", "name"],
                                  low_memory=False, on_bad_lines="skip")
            comp_df = comp_df.rename(columns={"name": "_company_name_resolved"})
            comp_df["company_id"] = comp_df["company_id"].astype(str)
            df["_cid_str"] = df.get("company_id", pd.Series(dtype=str)).astype(str)
            df = df.merge(comp_df, left_on="_cid_str", right_on="company_id",
                          how="left", suffixes=("", "_comp"))
            mask = df["company"].isna() | (df["company"].astype(str).str.strip() == "")
            df.loc[mask, "company"] = df.loc[mask, "_company_name_resolved"]
            df.drop(columns=["_cid_str", "_company_name_resolved",
                              "company_id_comp"], errors="ignore", inplace=True)
        except Exception as e:
            logger.warning(f"Could not merge companies.csv: {e}")
    return df


def _load_existing_hashes() -> set:
    try:
        from db import get_connection
        import psycopg2.extras
        conn = get_connection()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT DISTINCT source_hash FROM fact_job_posting;")
            hashes = {r["source_hash"] for r in cur.fetchall()}
            logger.warning(f"Loaded {len(hashes)} existing hashes from DB.")
            return hashes
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"Could not load existing hashes from DB: {e}. "
                       "Proceeding without cross-run dedup.")
        return set()


def extract_kaggle(path: str = None) -> Path:
    try:
        src = Path(path) if path else KAGGLE_DATASET_PATH

        if not src.exists():
            candidates = [
                src.parent / "job_postings.csv",
                src.parent / "postings.csv",
                src.parent / "linkedin_job_postings.csv",
                src.parent / "jobs.csv",
            ]
            found = next((p for p in candidates if p.exists()), None)
            if found:
                src = found
                logger.warning(f"Using dataset file: {src}")
            else:
                raise FileNotFoundError(
                    f"Kaggle LinkedIn dataset not found. Looked for: {src} and "
                    f"{[str(c) for c in candidates]}. "
                    "Download from https://www.kaggle.com/datasets/arshkon/linkedin-job-postings "
                    "and place job_postings.csv in data/raw/"
                )

        out_path = DATA_PROCESSED_DIR / "kaggle_staged.parquet"
        src_dir  = src.parent

        existing_db_hashes: set = _load_existing_hashes()

        seen_hashes: set = set(existing_db_hashes)
        output_chunks: list = []

        reader = pd.read_csv(src, chunksize=500, low_memory=False, on_bad_lines="skip")

        for i, chunk in enumerate(reader):
            chunk = _normalize_columns(chunk)
            chunk = _resolve_company_names(chunk, src_dir)

            if "date_posted" in chunk.columns:
                chunk["date_posted"] = chunk["date_posted"].apply(_parse_listed_time)

            if "platform" not in chunk.columns or chunk["platform"].isna().all():
                chunk["platform"] = "LinkedIn"
            else:
                chunk["platform"] = chunk["platform"].fillna("LinkedIn")

            chunk["source_hash"] = chunk.apply(_make_hash, axis=1)
            chunk["extraction_ts"] = datetime.utcnow().isoformat()
            chunk["data_source"] = "kaggle_linkedin"
            
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
            logger.warning(f"Kaggle chunk {i + 1}: +{len(new_rows)} rows (total: {collected})")

            if collected >= KAGGLE_BATCH_LIMIT:
                break

        if not output_chunks:
            logger.warning("Kaggle extraction: no new rows (all already in DB or empty source).")
            return None

        df = pd.concat(output_chunks, ignore_index=True)

        keep_cols = [c for c in EMPTY_COLS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in EMPTY_COLS]
        df = df[keep_cols + extra_cols]

        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.warning(f"Kaggle extraction done: {len(df)} new rows → {out_path}")

        meta = {
            "source": "kaggle_linkedin",
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
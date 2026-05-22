import json
import time
import random
import hashlib
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from config import (
    JOBSPY_GLOBAL_LOCATIONS,
    JOBSPY_RESULTS_PER_SEARCH,
    JOBSPY_SITES,
    PERIODIC_RAW_PATH,
    DATA_PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

PERIODIC_ROW_LIMIT = 100
_MAX_LOCATIONS = 10

COLUMN_RENAME = {
    "job_title": "title", "position": "title",
    "company_name": "company", "employer": "company",
    "job_location": "location",
    "job_description": "description",
    "posted_date": "date_posted", "post_date": "date_posted", "date": "date_posted",
    "min_salary": "salary_min", "salary_from": "salary_min",
    "max_salary": "salary_max", "salary_to": "salary_max",
    "remote": "is_remote", "work_from_home": "is_remote",
    "source": "platform", "site": "platform",
    "job_type": "employment_type",
}

REQUIRED_COLUMNS = [
    "title", "company", "location", "description", "date_posted",
    "salary_min", "salary_max", "is_remote", "platform", "employment_type",
    "search_location", "extraction_ts", "data_source",
]


def _make_hash(row: pd.Series) -> str:
    company = str(row.get("company") or "").strip().lower()
    title = str(row.get("title") or "").strip().lower()
    date = str(row.get("date_posted") or "").strip()
    return hashlib.md5(f"{company}|{title}|{date}".encode("utf-8")).hexdigest()


def _align_columns(df: pd.DataFrame, location: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_RENAME.items() if k in df.columns})
    df["search_location"] = location
    df["extraction_ts"] = datetime.utcnow().isoformat()
    df["data_source"] = "jobspy_periodic"
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]]


def _scrape_one(scrape_jobs, location: str, log: dict) -> pd.DataFrame:
    logger.warning(f"Scraping: @ {location!r} ...")
    try:
        df = scrape_jobs(
            site_name=JOBSPY_SITES,
            search_term="",
            location=location,
            results_wanted=JOBSPY_RESULTS_PER_SEARCH,
        )
        if df is None or len(df) == 0:
            logger.warning(f"  Scraped@ {location!r}: 0 result")
            return pd.DataFrame()
        df = _align_columns(df, location)
        logger.warning(f"  Scraped @ {location!r}: {len(df)} raw result")
        log["runs"].append({"location": location, "rows": len(df)})
        return df
    except Exception as e:
        err_msg = str(e)
        logger.warning(f"JobSpy failed: location={location!r} → {err_msg}")
        log["errors"].append({"location": location, "error": err_msg})
        return pd.DataFrame()


def _write_empty(out_path: Path, ts: str, log: dict) -> Path:
    PERIODIC_RAW_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_parquet(out_path, index=False, engine="pyarrow")
    log["total"] = 0
    (PERIODIC_RAW_PATH / f"periodic_{ts}_log.json").write_text(json.dumps(log, indent=2))
    return out_path


def _load_existing_hashes() -> set:
    try:
        from db import get_connection
        import psycopg2.extras
        conn = get_connection()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT DISTINCT source_hash FROM fact_job_posting;")
            hashes = {r["source_hash"] for r in cur.fetchall()}
            logger.warning(f"Periodic: loaded {len(hashes)} existing hashes from DB.")
            return hashes
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"Could not load existing hashes from DB (periodic): {e}.")
        return set()


def scrape_periodic(execution_date: str = None) -> Path:
    PERIODIC_RAW_PATH.mkdir(parents=True, exist_ok=True)

    ts = execution_date or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = PERIODIC_RAW_PATH / f"periodic_{ts}.parquet"

    log = {
        "ts": ts, "runs": [], "total": 0,
        "errors": [], "skipped": 0,
        "row_limit": PERIODIC_ROW_LIMIT,
        "max_locations": _MAX_LOCATIONS,
    }

    try:
        from jobspy import scrape_jobs
    except ImportError:
        logger.warning(
            "jobspy is not installed in this environment. "
            "Periodic scraping skipped — writing empty parquet so pipeline continues."
        )
        return _write_empty(out_path, ts, log)

    existing_db_hashes: set = _load_existing_hashes()
    locations = JOBSPY_GLOBAL_LOCATIONS[:_MAX_LOCATIONS]

    logger.warning(
        f"Periodic scrape: {len(locations)} locations, cap={PERIODIC_ROW_LIMIT} rows"
    )

    seen_hashes: set = set(existing_db_hashes)
    all_frames: list = []
    total_unique = 0
    reached_cap = False

    for location in locations:
        if reached_cap:
            break

        chunk = _scrape_one(scrape_jobs, location, log)

        if not chunk.empty:
            chunk["title"] = chunk["title"].astype(str).str.strip()
            chunk["company"] = chunk["company"].fillna("Unknown").astype(str).str.strip()
            chunk["description"] = chunk["description"].fillna("").astype(str)
            chunk["location"] = chunk["location"].fillna("Unknown").astype(str).str.strip()
            chunk = chunk[chunk["title"].str.len() > 0]
            chunk = chunk[chunk["description"].str.len() >= 20]

            if not chunk.empty:
                chunk["source_hash"] = chunk.apply(_make_hash, axis=1)
                new_rows = chunk[~chunk["source_hash"].isin(seen_hashes)].copy()

                remaining = PERIODIC_ROW_LIMIT - total_unique
                if remaining <= 0:
                    reached_cap = True
                    break

                new_rows = new_rows.head(remaining)
                if not new_rows.empty:
                    seen_hashes.update(new_rows["source_hash"].values)
                    all_frames.append(new_rows)
                    total_unique += len(new_rows)
                    logger.warning(
                        f"  @ {location!r}: +{len(new_rows)} baris "
                        f"(total: {total_unique}/{PERIODIC_ROW_LIMIT})"
                    )

                if total_unique >= PERIODIC_ROW_LIMIT:
                    reached_cap = True

        time.sleep(random.uniform(8.0, 15.0))

    if not all_frames:
        logger.warning("Periodic scrape: no new data collected.")
        return _write_empty(out_path, ts, log)

    df_all = pd.concat(all_frames, ignore_index=True)

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["source_hash"])
    log["skipped"] = before - len(df_all)

    df_all.to_parquet(out_path, index=False, engine="pyarrow")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(
        DATA_PROCESSED_DIR / f"periodic_{ts}_staged.parquet",
        index=False, engine="pyarrow",
    )

    log["total"] = len(df_all)
    (PERIODIC_RAW_PATH / f"periodic_{ts}_log.json").write_text(json.dumps(log, indent=2))

    logger.warning(
        f"Periodic extraction done: {len(df_all)} unique rows "
        f"({log['skipped']} duplicates removed) → {out_path}"
    )
    return out_path
import pandas as pd
import hashlib
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path

from config import (
    JOBSPY_SEARCH_TERMS,
    JOBSPY_GLOBAL_LOCATIONS,
    JOBSPY_RESULTS_PER_SEARCH,
    JOBSPY_SITES,
    JOBSPY_HOURS_OLD,
    PERIODIC_RAW_PATH,
    DATA_PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

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
    "search_term", "search_location", "extraction_ts", "data_source",
]


def _make_hash(row: pd.Series) -> str:
    company = str(row.get("company", "") or "").strip().lower()
    title = str(row.get("title", "") or "").strip().lower()
    date = str(row.get("date_posted", "") or "").strip()
    raw = f"{company}|{title}|{date}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _align_columns(df: pd.DataFrame, search_term: str, location: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_RENAME.items() if k in df.columns})
    df["search_term"] = search_term
    df["search_location"] = location
    df["extraction_ts"] = datetime.utcnow().isoformat()
    df["data_source"] = "jobspy_periodic"
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]]


def _scrape_one(scrape_jobs, term: str, location: str, log: dict) -> pd.DataFrame:
    try:
        df = scrape_jobs(
            site_name=JOBSPY_SITES,
            search_term=term,
            location=location,
            results_wanted=JOBSPY_RESULTS_PER_SEARCH,
            hours_old=JOBSPY_HOURS_OLD,
            verbose=0,
        )
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df = _align_columns(df, term, location)
        log["runs"].append({"term": term, "location": location, "rows": len(df)})
        return df
    except Exception as e:
        err_msg = str(e)
        logger.warning(f"JobSpy failed: term={term!r} location={location!r} → {err_msg}")
        log["errors"].append({"term": term, "location": location, "error": err_msg})
        return pd.DataFrame()


def scrape_periodic(execution_date: str = None) -> Path:
    try:
        from jobspy import scrape_jobs
    except ImportError:
        raise ImportError(
            "jobspy is not installed on this environment. "
            "Run: pip install jobspy --break-system-packages"
        )

    PERIODIC_RAW_PATH.mkdir(parents=True, exist_ok=True)

    ts = execution_date or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = PERIODIC_RAW_PATH / f"periodic_{ts}.parquet"
    log = {"ts": ts, "runs": [], "total": 0, "errors": [], "skipped": 0, "max_terms": 10, "max_locations": 10}

    all_frames = []
    terms = JOBSPY_SEARCH_TERMS[:10]
    locations = JOBSPY_GLOBAL_LOCATIONS[:10]
    total_combinations = len(terms) * len(locations)
    processed = 0

    logger.info(
        f"Starting scrape: {len(terms)} terms × {len(locations)} locations = {total_combinations} combinations"
    )

    for term in terms:
        for location in locations:
            chunk = _scrape_one(scrape_jobs, term, location, log)
            if not chunk.empty:
                all_frames.append(chunk)
            processed += 1

            if processed % 10 == 0:
                logger.info(f"  Progress: {processed}/{total_combinations}, "
                            f"collected {sum(len(f) for f in all_frames)} rows so far")

            delay = random.uniform(2.0, 4.0)
            time.sleep(delay)

    if not all_frames:
        logger.warning("No data collected from any search combination.")
        empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        empty_df.to_parquet(out_path, index=False, engine="pyarrow")
        log["total"] = 0
        log_path = PERIODIC_RAW_PATH / f"periodic_{ts}_log.json"
        log_path.write_text(json.dumps(log, indent=2))
        return out_path

    df_all = pd.concat(all_frames, ignore_index=True)

    df_all["title"] = df_all["title"].astype(str).str.strip()
    df_all["company"] = df_all["company"].fillna("Unknown").astype(str).str.strip()
    df_all["description"] = df_all["description"].fillna("").astype(str)
    df_all["location"] = df_all["location"].fillna("Unknown").astype(str).str.strip()

    df_all = df_all[df_all["title"].str.len() > 0]
    df_all = df_all[df_all["description"].str.len() >= 20]

    df_all["source_hash"] = df_all.apply(_make_hash, axis=1)
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["source_hash"])
    log["skipped"] = before - len(df_all)

    df_all.to_parquet(out_path, index=False, engine="pyarrow")
    log["total"] = len(df_all)

    staged_path = DATA_PROCESSED_DIR / f"periodic_{ts}_staged.parquet"
    df_all.to_parquet(staged_path, index=False, engine="pyarrow")

    log_path = PERIODIC_RAW_PATH / f"periodic_{ts}_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    logger.info(
        f"Periodic extraction done: {len(df_all)} unique rows "
        f"({log['skipped']} duplicates removed) → {out_path}"
    )
    return out_path

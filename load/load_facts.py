import logging
import pandas as pd
import numpy as np
import psycopg2.extras
from db import get_connection

logger = logging.getLogger(__name__)


def _safe_float(val) -> float:
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int:
    if val is None:
        return None
    try:
        f = float(val)
        return int(f) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _resolve_location_id(row: pd.Series, location_map: dict):
    city = str(row.get("loc_city", "") or "")
    country = str(row.get("loc_country", "") or "Unknown")

    key = (city, country)
    if key in location_map:
        return location_map[key]

    fallback_key = ("", country)
    if fallback_key in location_map:
        return location_map[fallback_key]

    return None


def load_fact_job_posting(
    df: pd.DataFrame,
    time_map: dict,
    location_map: dict,
    company_map: dict,
    position_map: dict,
    platform_map: dict,
) -> dict:
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        date_key = str(row["date_parsed"].date()) if pd.notna(row.get("date_parsed")) else None
        time_id = time_map.get(date_key)
        if time_id is None:
            skipped += 1
            continue

        location_id = _resolve_location_id(row, location_map)
        company_name = str(row.get("company_clean", "Unknown")).strip() or "Unknown"
        company_id = company_map.get(company_name, company_map.get("Unknown"))
        title = str(row.get("title_clean", "")).strip()[:255] or "unknown"
        level = str(row.get("job_level", "Unknown"))
        cat = str(row.get("job_category", "Other"))
        position_id = position_map.get((title, level, cat))
        platform = str(row.get("platform_norm", "LinkedIn")).strip() or "LinkedIn"
        platform_id = platform_map.get(platform, platform_map.get("LinkedIn"))

        rows.append((
            int(time_id),
            _safe_int(location_id),
            _safe_int(company_id),
            _safe_int(position_id),
            _safe_int(platform_id),
            1,
            bool(row.get("has_salary", False)),
            bool(row.get("is_remote", False)),
            _safe_float(row.get("salary_min")),
            _safe_float(row.get("salary_max")),
            str(row.get("source_hash", ""))[:64],
            str(row.get("title", ""))[:500],
        ))

    if skipped > 0:
        logger.warning(f"load_fact: skipped {skipped} rows with no time_id match")
    if not rows:
        logger.warning("load_fact: no rows to insert")
        return {}

    conn = get_connection()
    try:
        cur = conn.cursor()
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO fact_job_posting
                (time_id, location_id, company_id, position_id, platform_id,
                 posting_count, has_salary, is_remote,
                 salary_min, salary_max, source_hash, raw_title)
            VALUES %s
            ON CONFLICT (source_hash, time_id) DO NOTHING;
            """,
            rows,
            page_size=500,
        )
        conn.commit()
        logger.warning(f"load_fact: inserted up to {len(rows)} rows")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    conn2 = get_connection()
    try:
        cur2 = conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        hashes = [r[11] for r in rows]
        cur2.execute(
            "SELECT job_id, source_hash FROM fact_job_posting WHERE source_hash = ANY(%s);",
            (hashes,),
        )
        job_id_map = {r["source_hash"]: r["job_id"] for r in cur2.fetchall()}
    finally:
        conn2.close()

    logger.warning(f"load_fact: {len(job_id_map)} job_id mappings retrieved")
    return job_id_map


def load_bridge_job_skill(
    entities_df: pd.DataFrame,
    source_hash_to_job_id: dict,
    skill_id_map: dict,
):
    rows = []
    for _, row in entities_df.iterrows():
        job_id = source_hash_to_job_id.get(str(row["source_hash"]))
        skill_id = skill_id_map.get(str(row["entity_text"]))
        if job_id is None or skill_id is None:
            continue
        confidence = _safe_float(row.get("extraction_confidence", 0.9)) or 0.9
        rows.append((int(job_id), int(skill_id), round(confidence, 4)))

    if not rows:
        logger.warning("load_bridge_job_skill: no rows to insert")
        return

    conn = get_connection()
    try:
        cur = conn.cursor()
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO bridge_job_skill (job_id, skill_id, extraction_confidence)
            VALUES %s
            ON CONFLICT (job_id, skill_id) DO NOTHING;
            """,
            rows,
            page_size=1000,
        )
        conn.commit()
        logger.warning(f"load_bridge_job_skill: {len(rows)} rows inserted")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
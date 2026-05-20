import logging
import pandas as pd
import psycopg2.extras
from db import get_connection

logger = logging.getLogger(__name__)


def _exec_returning(conn, sql: str, params: tuple):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params)
    return cur.fetchone()


def upsert_dim_time(dates: pd.Series) -> dict:
    unique_dates = pd.to_datetime(dates.dropna()).dt.normalize().unique()
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for d in unique_dates:
            if pd.isna(d):
                continue
            iso = d.isocalendar()
            cur.execute("""
                INSERT INTO dim_time (date, week, month, quarter, year)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (date) DO NOTHING;
            """, (d.date(), int(iso[1]), int(d.month), int((d.month - 1) // 3 + 1), int(d.year)))
        conn.commit()
        cur.execute("SELECT time_id, date FROM dim_time;")
        mapping = {str(r["date"]): r["time_id"] for r in cur.fetchall()}
    finally:
        conn.close()
    logger.info(f"dim_time: {len(mapping)} dates total")
    return mapping


def upsert_dim_location(df: pd.DataFrame) -> dict:
    loc_cols = ["loc_city", "loc_province", "loc_country", "global_region"]
    rows = df[loc_cols].drop_duplicates().fillna("").replace("nan", "")
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for _, row in rows.iterrows():
            city = row["loc_city"] or None
            province = row["loc_province"] or None
            country = row["loc_country"] or "Unknown"
            region = row["global_region"] or "Other"
            cur.execute("""
                INSERT INTO dim_location (city, province_state, country, global_region)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (city, province_state, country, global_region) DO NOTHING;
            """, (city, province, country, region))
        conn.commit()
        cur.execute("""
            SELECT location_id, city, province_state, country, global_region
            FROM dim_location;
        """)
        mapping = {}
        for r in cur.fetchall():
            key = (
                r["city"] or "",
                r["province_state"] or "",
                r["country"] or "Unknown",
                r["global_region"] or "Other",
            )
            mapping[key] = r["location_id"]
    finally:
        conn.close()
    logger.info(f"dim_location: {len(mapping)} locations total")
    return mapping


def upsert_dim_company(df: pd.DataFrame) -> dict:
    names = df["company_clean"].dropna().unique()
    conn  = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for name in names:
            name = str(name).strip() or "Unknown"
            cur.execute("""
                INSERT INTO dim_company (company_name)
                VALUES (%s)
                ON CONFLICT (company_name) DO NOTHING;
            """, (name,))
        conn.commit()
        cur.execute("SELECT company_id, company_name FROM dim_company;")
        mapping = {r["company_name"]: r["company_id"] for r in cur.fetchall()}
    finally:
        conn.close()
    logger.info(f"dim_company: {len(mapping)} companies total")
    return mapping


def upsert_dim_position(df: pd.DataFrame) -> dict:
    cols = ["title_clean", "job_level", "job_category"]
    rows = df[cols].drop_duplicates().fillna("")
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for _, row in rows.iterrows():
            title = str(row["title_clean"]).strip()[:255] or "Unknown"
            level = str(row["job_level"]).strip() or "Unknown"
            cat = str(row["job_category"]).strip() or "Other"
            cur.execute("""
                INSERT INTO dim_position (normalized_job_title, job_level, job_category)
                VALUES (%s, %s, %s)
                ON CONFLICT (normalized_job_title, job_level, job_category) DO NOTHING;
            """, (title, level, cat))
        conn.commit()
        cur.execute("""
            SELECT position_id, normalized_job_title, job_level, job_category
            FROM dim_position;
        """)
        mapping = {
            (r["normalized_job_title"], r["job_level"], r["job_category"]): r["position_id"]
            for r in cur.fetchall()
        }
    finally:
        conn.close()
    logger.info(f"dim_position: {len(mapping)} positions total")
    return mapping


def upsert_dim_platform(platforms: list) -> dict:
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for p in set(str(x).strip() or "Unknown" for x in platforms):
            cur.execute("""
                INSERT INTO dim_platform (platform_name)
                VALUES (%s)
                ON CONFLICT (platform_name) DO NOTHING;
            """, (p,))
        conn.commit()
        cur.execute("SELECT platform_id, platform_name FROM dim_platform;")
        mapping = {r["platform_name"]: r["platform_id"] for r in cur.fetchall()}
    finally:
        conn.close()
    logger.info(f"dim_platform: {len(mapping)} platforms total")
    return mapping


def upsert_dim_skill(entities_df: pd.DataFrame) -> dict:
    if entities_df.empty:
        return {}

    mask   = entities_df["entity_type"].str.contains("Knowledge|Skill", case=False, na=False)
    skills = (
        entities_df[mask][["entity_text", "entity_type", "source_model"]]
        .drop_duplicates("entity_text")
        .rename(columns={"entity_text": "skill_name", "entity_type": "skill_type", "source_model": "skill_domain"})
    )

    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for _, row in skills.iterrows():
            cur.execute("""
                INSERT INTO dim_skill (skill_name, skill_type, skill_domain)
                VALUES (%s, %s, %s)
                ON CONFLICT (skill_name) DO UPDATE
                    SET skill_type = EXCLUDED.skill_type,
                        skill_domain = EXCLUDED.skill_domain;
            """, (row["skill_name"], row["skill_type"], row["skill_domain"]))
        conn.commit()
        cur.execute("SELECT skill_id, skill_name FROM dim_skill;")
        mapping = {r["skill_name"]: r["skill_id"] for r in cur.fetchall()}
    finally:
        conn.close()
    logger.info(f"dim_skill: {len(mapping)} skills total")
    return mapping


def upsert_dim_entity(entities_df: pd.DataFrame) -> dict:
    if entities_df.empty:
        return {}

    uniq = entities_df[["entity_text", "entity_type", "source_model"]].drop_duplicates(
        subset=["entity_text", "entity_type"]
    )

    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for _, row in uniq.iterrows():
            cur.execute("""
                INSERT INTO dim_entity (entity_text, entity_type, source_model)
                VALUES (%s, %s, %s)
                ON CONFLICT (entity_text, entity_type) DO UPDATE
                    SET source_model = EXCLUDED.source_model;
            """, (row["entity_text"], row["entity_type"], row["source_model"]))
        conn.commit()
        cur.execute("SELECT entity_id, entity_text, entity_type FROM dim_entity;")
        mapping = {
            (r["entity_text"], r["entity_type"]): r["entity_id"]
            for r in cur.fetchall()
        }
    finally:
        conn.close()
    logger.info(f"dim_entity: {len(mapping)} entities total")
    return mapping
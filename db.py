import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from config import DB_CONFIG, DB_URL


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


@contextmanager
def get_cursor(commit=True):
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


def run_ddl(sql_path: str):
    with open(sql_path, "r") as f:
        sql = f.read()
    with get_cursor() as cur:
        cur.execute(sql)
    print(f"DDL executed: {sql_path}")


def refresh_materialized_views():
    views = [
        "mv_weekly_skill_demand",
        "mv_platform_monthly",
        "mv_company_hiring",
    ]
    with get_cursor() as cur:
        for v in views:
            cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {v};")
    print("All materialized views refreshed.")

import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from sqlalchemy import create_engine
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


def run_partition_setup():
    sql = """
    DO $$
    DECLARE
        yr INT;
        id_start INT;
        id_end INT;
        tbl TEXT;
    BEGIN
        FOR yr IN 2023..2026 LOOP
            tbl := 'fact_job_posting_' || yr;
            IF NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = tbl AND n.nspname = 'public'
            ) THEN
                SELECT time_id INTO id_start FROM dim_time WHERE date = make_date(yr, 1, 1);
                SELECT time_id INTO id_end FROM dim_time WHERE date = make_date(yr + 1, 1, 1);
                IF id_start IS NOT NULL AND id_end IS NOT NULL THEN
                    EXECUTE format(
                        'CREATE TABLE %I PARTITION OF fact_job_posting FOR VALUES FROM (%s) TO (%s)',
                        tbl, id_start, id_end
                    );
                    RAISE NOTICE 'Created partition %', tbl;
                END IF;
            ELSE
                RAISE NOTICE 'Partition % sudah ada, skip.', tbl;
            END IF;
        END LOOP;
    END $$;
    """
    with get_cursor() as cur:
        cur.execute(sql)
    print("Partition setup done.")


def refresh_materialized_views():
    views = [
        "mv_weekly_skill_demand",
        "mv_platform_monthly",
        "mv_company_hiring",
    ]
    conn = get_connection()
    conn.autocommit = True
    try:
        cur = conn.cursor()
        for v in views:
            try:
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {v};")
                print(f"Refreshed: {v}")
            except psycopg2.errors.ObjectNotInPrerequisiteState:
                cur.execute(f"REFRESH MATERIALIZED VIEW {v};")
                print(f"Refreshed (non-concurrent): {v}")
            except Exception as e:
                print(f"Warning: could not refresh {v}: {e}")
    finally:
        conn.close()
    print("All materialized views refreshed.")
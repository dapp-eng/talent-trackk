import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import db
from db import get_cursor, get_connection
import psycopg2.extras
from datetime import date, timedelta


def verify_connection():
    with get_cursor() as cur:
        cur.execute("SELECT version();")
        row = cur.fetchone()
        print(f"Connected to PostgreSQL: {row['version'][:60]}...")


def run_ddl():
    ddl_path = ROOT / "sql" / "ddl.sql"
    db.run_ddl(str(ddl_path))


def seed_dim_time():
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        start = date(2021, 1, 1)
        end = date(2028, 12, 31)
        d = start
        count = 0
        while d <= end:
            iso = d.isocalendar()
            cur.execute("""
                INSERT INTO dim_time (date, week, month, quarter, year)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (date) DO NOTHING;
            """, (d, int(iso[1]), d.month, (d.month - 1) // 3 + 1, d.year))
            count += 1
            d += timedelta(days=1)
        conn.commit()
        print(f"dim_time seeded: {start} to {end} ({count} rows)")
    finally:
        conn.close()


def seed_platforms():
    platforms = [
        ("LinkedIn", "Global"),
        ("Indeed", "Global"),
        ("Glassdoor", "Global"),
        ("ZipRecruiter", "USA"),
        ("Google Jobs", "Global"),
        ("Kaggle Historical", "Global"),
        ("Synthetic", "Testing"),
        ("Unknown", "Unknown"),
    ]
    with get_cursor() as cur:
        for name, region in platforms:
            cur.execute("""
                INSERT INTO dim_platform (platform_name, regional_focus)
                VALUES (%s, %s)
                ON CONFLICT (platform_name) DO NOTHING;
            """, (name, region))
    print(f"Seeded {len(platforms)} platforms.")


def verify_extensions():
    with get_cursor() as cur:
        cur.execute("SELECT extname FROM pg_extension;")
        exts = [r["extname"] for r in cur.fetchall()]
        print(f"Installed extensions: {exts}")
        required = ["vector", "pg_trgm", "btree_gin"]
        for ext in required:
            status = "OK" if ext in exts else "MISSING"
            print(f"  {ext}: {status}")


def verify_partitions():
    with get_cursor() as cur:
        cur.execute("""
            SELECT c.relname, pg_get_expr(c.relpartbound, c.oid) AS bounds
            FROM pg_class c
            JOIN pg_inherits i ON i.inhrelid = c.oid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = 'fact_job_posting'
            ORDER BY c.relname;
        """)
        rows = cur.fetchall()
        print(f"fact_job_posting partitions ({len(rows)}):")
        for r in rows:
            print(f"  {r['relname']}: {r['bounds']}")


if __name__ == "__main__":
    print("TalentTrack: One-Time DB Initialization")
    verify_connection()
    verify_extensions()
    run_ddl()
    seed_dim_time()
    seed_platforms()
    verify_partitions()
    print("Initialization complete")
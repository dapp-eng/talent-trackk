import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import db
from db import get_cursor


def verify_connection():
    with get_cursor() as cur:
        cur.execute("SELECT version();")
        row = cur.fetchone()
        print(f"Connected to PostgreSQL: {row['version'][:60]}...")


def run_ddl():
    ddl_path = ROOT / "sql" / "ddl.sql"
    db.run_ddl(str(ddl_path))


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


if __name__ == "__main__":
    print("TalentTrack: One-Time DB Initialization")
    verify_connection()
    run_ddl()
    seed_platforms()
    verify_extensions()
    print("Initialization complete")

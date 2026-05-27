from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import ExternalPythonOperator

VENV_PYTHON = "/opt/airflow/.venv/bin/python"

_DEFAULT_ARGS = {
    "owner": "inter24",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}


def task_init_db():
    import os, sys
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    import db
    db.run_ddl(os.path.join(root, "sql", "ddl.sql"))


def task_seed_dim_time():
    import os, sys
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    import psycopg2.extras
    from datetime import date, timedelta
    from db import get_connection
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        start = date(2023, 1, 1)
        end = date(2027, 12, 31)
        d = start
        while d <= end:
            iso = d.isocalendar()
            cur.execute("""
                INSERT INTO dim_time (date, week, month, quarter, year)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (date) DO NOTHING;
            """, (d, int(iso[1]), d.month, (d.month - 1) // 3 + 1, d.year))
            d += timedelta(days=1)
        conn.commit()
        print(f"dim_time seeded from {start} to {end}")
    finally:
        conn.close()


def task_setup_partitions():
    import os, sys
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    import db
    db.run_partition_setup()


def task_extract_kaggle():
    import os, sys
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from extract.extract_kaggle import extract_kaggle
    return str(extract_kaggle())


def task_extract_periodic(execution_date):
    import os, sys
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from extract.extract_periodic import scrape_periodic
    return str(scrape_periodic(execution_date=execution_date))


def task_preprocess_kaggle(kaggle_raw_path):
    import os, sys, logging
    from pathlib import Path
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    _logger = logging.getLogger(__name__)
    from transform.preprocess import preprocess_file
    if not kaggle_raw_path:
        _logger.warning("There's no new data from Kaggle, skip preprocess.")
        return None
    if not Path(kaggle_raw_path).exists():
        fallback = os.path.join(root, "data", "raw", "kaggle_staged.parquet")
        if not os.path.exists(fallback):
            raise FileNotFoundError(
                f"Kaggle raw parquet is not found in {kaggle_raw_path} neither in {fallback}"
            )
        kaggle_raw_path = fallback
    return str(preprocess_file(kaggle_raw_path, source_label="kaggle_2024"))


def task_preprocess_periodic(periodic_raw_path):
    import os, sys
    from pathlib import Path
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.preprocess import preprocess_file
    if not periodic_raw_path or not Path(periodic_raw_path).exists():
        return ""
    return str(preprocess_file(periodic_raw_path, source_label="periodic"))


def task_ner_kaggle(kaggle_preprocessed_path):
    import os, sys
    from pathlib import Path
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.ner_skills import run_ner
    if not kaggle_preprocessed_path or not Path(kaggle_preprocessed_path).exists():
        return ""
    return str(run_ner(kaggle_preprocessed_path))


def task_ner_periodic(periodic_preprocessed_path):
    import os, sys
    from pathlib import Path
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.ner_skills import run_ner
    if not periodic_preprocessed_path or not Path(periodic_preprocessed_path).exists():
        return ""
    return str(run_ner(periodic_preprocessed_path))


def task_dedup_kaggle(kaggle_preprocessed_path):
    import os, sys
    import psycopg2.extras
    from pathlib import Path
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.dedup import run_dedup
    from db import get_connection
    if not kaggle_preprocessed_path or not Path(kaggle_preprocessed_path).exists():
        return ""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT DISTINCT source_hash FROM fact_job_posting;")
        existing_hashes = {r["source_hash"] for r in cur.fetchall()}
    finally:
        conn.close()
    return str(run_dedup(kaggle_preprocessed_path, existing_hashes=existing_hashes))


def task_dedup_periodic(periodic_preprocessed_path):
    import os, sys
    import psycopg2.extras
    from pathlib import Path
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.dedup import run_dedup
    from db import get_connection
    if not periodic_preprocessed_path or not Path(periodic_preprocessed_path).exists():
        return ""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT DISTINCT source_hash FROM fact_job_posting;")
        existing_hashes = {r["source_hash"] for r in cur.fetchall()}
    finally:
        conn.close()
    return str(run_dedup(periodic_preprocessed_path, existing_hashes=existing_hashes))


def task_load_kaggle(kaggle_deduped_path, kaggle_entities_path):
    import os, sys
    import pandas as pd
    from pathlib import Path
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from load.load_dimensions import (
        upsert_dim_time, upsert_dim_location, upsert_dim_company,
        upsert_dim_position, upsert_dim_platform, upsert_dim_skill,
    )
    from load.load_facts import load_fact_job_posting, load_bridge_job_skill
    if not kaggle_deduped_path or not Path(kaggle_deduped_path).exists():
        return
    df = pd.read_parquet(kaggle_deduped_path)
    if df.empty:
        return
    df = df[pd.to_datetime(df["date_parsed"]).dt.year >= 2023]
    if df.empty:
        return
    entities_df = (
        pd.read_parquet(kaggle_entities_path)
        if (kaggle_entities_path and Path(kaggle_entities_path).exists())
        else pd.DataFrame()
    )
    time_map = upsert_dim_time(df["date_parsed"])
    location_map = upsert_dim_location(df)
    company_map = upsert_dim_company(df)
    position_map = upsert_dim_position(df)
    platform_map = upsert_dim_platform(df["platform_norm"].dropna().unique().tolist())
    skill_map = upsert_dim_skill(entities_df) if not entities_df.empty else {}
    job_id_map = load_fact_job_posting(df, time_map, location_map, company_map, position_map, platform_map)
    if not entities_df.empty and skill_map:
        load_bridge_job_skill(entities_df, job_id_map, skill_map)


def task_load_periodic(periodic_deduped_path, periodic_entities_path):
    import os, sys
    import pandas as pd
    from pathlib import Path
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from load.load_dimensions import (
        upsert_dim_time, upsert_dim_location, upsert_dim_company,
        upsert_dim_position, upsert_dim_platform, upsert_dim_skill,
    )
    from load.load_facts import load_fact_job_posting, load_bridge_job_skill
    if not periodic_deduped_path or not Path(periodic_deduped_path).exists():
        return
    df = pd.read_parquet(periodic_deduped_path)
    if df.empty:
        return
    df = df[pd.to_datetime(df["date_parsed"]).dt.year >= 2023]
    if df.empty:
        return
    entities_df = (
        pd.read_parquet(periodic_entities_path)
        if (periodic_entities_path and Path(periodic_entities_path).exists())
        else pd.DataFrame()
    )
    time_map = upsert_dim_time(df["date_parsed"])
    location_map = upsert_dim_location(df)
    company_map = upsert_dim_company(df)
    position_map = upsert_dim_position(df)
    platform_map = upsert_dim_platform(df["platform_norm"].dropna().unique().tolist())
    skill_map = upsert_dim_skill(entities_df) if not entities_df.empty else {}
    job_id_map = load_fact_job_posting(df, time_map, location_map, company_map, position_map, platform_map)
    if not entities_df.empty and skill_map:
        load_bridge_job_skill(entities_df, job_id_map, skill_map)


def task_refresh_views():
    import os, sys
    import psycopg2, psycopg2.extras
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from db import get_connection
    conn = get_connection()
    conn.autocommit = True
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT schemaname || '.' || matviewname AS full_name "
            "FROM pg_matviews WHERE schemaname = 'public';"
        )
        views = [row["full_name"] for row in cur.fetchall()]
        for v in views:
            try:
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {v};")
                print(f"Refreshed: {v}")
            except psycopg2.errors.ObjectNotInPrerequisiteState:
                cur.execute(f"REFRESH MATERIALIZED VIEW {v};")
                print(f"Refreshed (non-concurrent): {v}")
            except Exception as e:
                print(f"Warning: skipping {v}: {e}")
    finally:
        conn.close()


def task_run_forecasting():
    import os, sys
    _CANDIDATES_LOCAL = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    _SUBS_LOCAL = ["", "extract", "transform", "load", "analysis"]
    root = next((p for p in _CANDIDATES_LOCAL if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {_CANDIDATES_LOCAL}")
    for sub in _SUBS_LOCAL:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from analysis.forecasting import run_forecasting
    from db import get_engine
    run_forecasting(engine=get_engine())


def _epo(task_id, python_callable, op_kwargs=None,
         retries=1, execution_timeout=timedelta(hours=4)):
    return ExternalPythonOperator(
        task_id=task_id,
        python=VENV_PYTHON,
        python_callable=python_callable,
        op_kwargs=op_kwargs or {},
        expect_airflow=False,
        retries=retries,
        execution_timeout=execution_timeout,
    )


with DAG(
    dag_id="group3_talenttrack_kaggle",
    default_args=_DEFAULT_ARGS,
    description="TalentTrack: Kaggle ETL pipeline (every 3 hours)",
    schedule="0 */3 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["talent-trackk", "dwh", "etl", "kaggle"],
) as dag_kaggle:

    k_init_db = _epo("init_db", task_init_db)
    k_seed_time = _epo("seed_dim_time", task_seed_dim_time)
    k_setup_partitions = _epo("setup_partitions", task_setup_partitions)
    k_extract = _epo("extract_kaggle", task_extract_kaggle)
    k_preprocess = _epo(
        "preprocess_kaggle", task_preprocess_kaggle,
        {"kaggle_raw_path": "{{ ti.xcom_pull(task_ids='extract_kaggle') }}"},
    )
    k_ner = _epo(
        "ner_kaggle", task_ner_kaggle,
        {"kaggle_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_kaggle') }}"},
        retries=0, execution_timeout=timedelta(hours=6),
    )
    k_dedup = _epo(
        "dedup_kaggle", task_dedup_kaggle,
        {"kaggle_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_kaggle') }}"},
    )
    k_load = _epo(
        "load_kaggle", task_load_kaggle,
        {
            "kaggle_deduped_path": "{{ ti.xcom_pull(task_ids='dedup_kaggle') }}",
            "kaggle_entities_path": "{{ ti.xcom_pull(task_ids='ner_kaggle') }}",
        },
    )
    k_refresh = _epo("refresh_views", task_refresh_views)
    k_forecast = _epo("run_forecasting", task_run_forecasting)

    (
        k_init_db >> k_seed_time >> k_setup_partitions >> k_extract
        >> k_preprocess >> [k_ner, k_dedup]
        >> k_load >> k_refresh >> k_forecast
    )


with DAG(
    dag_id="group3_talenttrack_jobspy",
    default_args=_DEFAULT_ARGS,
    description="TalentTrack: Periodic scraping pipeline (once per day)",
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["talent-trackk", "dwh", "etl", "scraping"],
) as dag_periodic:

    p_init_db = _epo("init_db", task_init_db)
    p_seed_time = _epo("seed_dim_time", task_seed_dim_time)
    p_setup_partitions = _epo("setup_partitions", task_setup_partitions)
    p_extract = _epo(
        "extract_periodic", task_extract_periodic,
        {"execution_date": "{{ ds }}"},
    )
    p_preprocess = _epo(
        "preprocess_periodic", task_preprocess_periodic,
        {"periodic_raw_path": "{{ ti.xcom_pull(task_ids='extract_periodic') }}"},
    )
    p_ner = _epo(
        "ner_periodic", task_ner_periodic,
        {"periodic_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_periodic') }}"},
        retries=0, execution_timeout=timedelta(hours=2),
    )
    p_dedup = _epo(
        "dedup_periodic", task_dedup_periodic,
        {"periodic_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_periodic') }}"},
    )
    p_load = _epo(
        "load_periodic", task_load_periodic,
        {
            "periodic_deduped_path": "{{ ti.xcom_pull(task_ids='dedup_periodic') }}",
            "periodic_entities_path": "{{ ti.xcom_pull(task_ids='ner_periodic') }}",
        },
    )
    p_refresh = _epo("refresh_views", task_refresh_views)
    p_forecast = _epo("run_forecasting", task_run_forecasting)

    (
        p_init_db >> p_seed_time >> p_setup_partitions >> p_extract
        >> p_preprocess >> [p_ner, p_dedup]
        >> p_load >> p_refresh >> p_forecast
    )
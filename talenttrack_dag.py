from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import ExternalPythonOperator

VENV_PYTHON = "/opt/airflow/.venv/bin/python"

default_args = {
    "owner": "inter24",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}


def task_init_db():
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    import db
    db.run_ddl(os.path.join(root, "sql", "ddl.sql"))


def task_seed_dim_time():
    import sys, os, psycopg2.extras
    from datetime import date, timedelta
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from db import get_connection
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        start = date(2021, 1, 1)
        end = date(2028, 12, 31)
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
        print(f"dim_time seeded dari {start} sampai {end}")
    finally:
        conn.close()


def task_extract_kaggle():
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from extract.extract_kaggle import extract_kaggle
    return str(extract_kaggle())


def task_extract_periodic(execution_date):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from extract.extract_periodic import scrape_periodic
    return str(scrape_periodic(execution_date=execution_date))


def task_preprocess_kaggle(kaggle_raw_path):
    import sys, os
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.preprocess import preprocess_file
    if not kaggle_raw_path or not Path(kaggle_raw_path).exists():
        fallback = os.path.join(root, "data", "raw", "kaggle_staged.parquet")
        if not os.path.exists(fallback):
            raise FileNotFoundError(
                f"Kaggle raw parquet tidak ditemukan di {kaggle_raw_path} maupun {fallback}"
            )
        kaggle_raw_path = fallback
    return str(preprocess_file(kaggle_raw_path, source_label="kaggle_2024"))


def task_preprocess_periodic(periodic_raw_path):
    import sys, os
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.preprocess import preprocess_file
    if not periodic_raw_path or not Path(periodic_raw_path).exists():
        return ""
    return str(preprocess_file(periodic_raw_path, source_label="periodic"))


def task_ner_kaggle(kaggle_preprocessed_path):
    import sys, os
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.ner_skills import run_ner
    if not kaggle_preprocessed_path or not Path(kaggle_preprocessed_path).exists():
        return ""
    return str(run_ner(kaggle_preprocessed_path))


def task_ner_periodic(periodic_preprocessed_path):
    import sys, os
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.ner_skills import run_ner
    if not periodic_preprocessed_path or not Path(periodic_preprocessed_path).exists():
        return ""
    return str(run_ner(periodic_preprocessed_path))


def task_embed_kaggle(kaggle_preprocessed_path):
    import sys, os, gc, logging, numpy as np
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    logger = logging.getLogger(__name__)
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
    except Exception:
        pass
    if not kaggle_preprocessed_path or not Path(kaggle_preprocessed_path).exists():
        return ""
    try:
        from transform.embed import compute_and_save_embeddings
        result = compute_and_save_embeddings(kaggle_preprocessed_path)
        gc.collect()
        return str(result)
    except MemoryError as e:
        logger.error(f"MemoryError during embedding (kaggle): {e}")
        from config import DATA_PROCESSED_DIR, EMBEDDING_DIM, SBERT_DIM
        stem = Path(kaggle_preprocessed_path).stem
        out_path = DATA_PROCESSED_DIR / f"{stem}_embeddings.npz"
        np.savez_compressed(
            out_path,
            source_hashes=np.array([], dtype=str),
            jobbert=np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
            sbert=np.zeros((0, SBERT_DIM), dtype=np.float32),
        )
        return str(out_path)
    except Exception as e:
        logger.error(f"Embedding task failed (kaggle): {e}", exc_info=True)
        raise


def task_embed_periodic(periodic_preprocessed_path):
    import sys, os, gc, logging, numpy as np
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    logger = logging.getLogger(__name__)
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
    except Exception:
        pass
    if not periodic_preprocessed_path or not Path(periodic_preprocessed_path).exists():
        return ""
    try:
        from transform.embed import compute_and_save_embeddings
        result = compute_and_save_embeddings(periodic_preprocessed_path)
        gc.collect()
        return str(result)
    except MemoryError as e:
        logger.error(f"MemoryError during embedding (periodic): {e}")
        from config import DATA_PROCESSED_DIR, EMBEDDING_DIM, SBERT_DIM
        stem = Path(periodic_preprocessed_path).stem
        out_path = DATA_PROCESSED_DIR / f"{stem}_embeddings.npz"
        np.savez_compressed(
            out_path,
            source_hashes=np.array([], dtype=str),
            jobbert=np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
            sbert=np.zeros((0, SBERT_DIM), dtype=np.float32),
        )
        return str(out_path)
    except Exception as e:
        logger.error(f"Embedding task failed (periodic): {e}", exc_info=True)
        raise


def task_dedup_kaggle(kaggle_preprocessed_path, kaggle_embeddings_path):
    import sys, os
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.dedup import run_dedup
    if not kaggle_preprocessed_path or not Path(kaggle_preprocessed_path).exists():
        return ""
    emb = kaggle_embeddings_path if (kaggle_embeddings_path and Path(kaggle_embeddings_path).exists()) else None
    return str(run_dedup(kaggle_preprocessed_path, embeddings_path=emb))


def task_dedup_periodic(periodic_preprocessed_path, periodic_embeddings_path):
    import sys, os, psycopg2.extras
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
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
    emb = periodic_embeddings_path if (periodic_embeddings_path and Path(periodic_embeddings_path).exists()) else None
    return str(run_dedup(periodic_preprocessed_path, embeddings_path=emb, existing_hashes=existing_hashes))


def task_load_kaggle(kaggle_deduped_path, kaggle_skills_path, kaggle_embeddings_path):
    import sys, os, numpy as np, pandas as pd
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from load.load_dimensions import (
        upsert_dim_time, upsert_dim_location, upsert_dim_company,
        upsert_dim_position, upsert_dim_platform, upsert_dim_skill,
    )
    from load.load_facts import load_fact_job_posting, load_bridge_job_skill, load_embeddings
    if not kaggle_deduped_path or not Path(kaggle_deduped_path).exists():
        return
    df = pd.read_parquet(kaggle_deduped_path)
    if df.empty:
        return
    df = df[pd.to_datetime(df["date_parsed"]).dt.year >= 2024]
    if df.empty:
        return
    skills_df = (
        pd.read_parquet(kaggle_skills_path)
        if (kaggle_skills_path and Path(kaggle_skills_path).exists())
        else pd.DataFrame()
    )
    time_map = upsert_dim_time(df["date_parsed"])
    location_map = upsert_dim_location(df)
    company_map = upsert_dim_company(df)
    position_map = upsert_dim_position(df)
    platform_map = upsert_dim_platform(df["platform_norm"].dropna().unique().tolist())
    skill_map = upsert_dim_skill(skills_df) if not skills_df.empty else {}
    job_id_map = load_fact_job_posting(df, time_map, location_map, company_map, position_map, platform_map)
    if not skills_df.empty and skill_map:
        load_bridge_job_skill(skills_df, job_id_map, skill_map)
    if kaggle_embeddings_path and Path(kaggle_embeddings_path).exists():
        npz = np.load(kaggle_embeddings_path, allow_pickle=True)
        if npz["jobbert"].shape[0] > 0:
            load_embeddings(df, job_id_map, npz["jobbert"], npz["sbert"])


def task_load_periodic(periodic_deduped_path, periodic_skills_path, periodic_embeddings_path):
    import sys, os, numpy as np, pandas as pd
    from pathlib import Path
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from load.load_dimensions import (
        upsert_dim_time, upsert_dim_location, upsert_dim_company,
        upsert_dim_position, upsert_dim_platform, upsert_dim_skill,
    )
    from load.load_facts import load_fact_job_posting, load_bridge_job_skill, load_embeddings
    if not periodic_deduped_path or not Path(periodic_deduped_path).exists():
        return
    df = pd.read_parquet(periodic_deduped_path)
    if df.empty:
        return
    df = df[pd.to_datetime(df["date_parsed"]).dt.year >= 2024]
    if df.empty:
        return
    skills_df = (
        pd.read_parquet(periodic_skills_path)
        if (periodic_skills_path and Path(periodic_skills_path).exists())
        else pd.DataFrame()
    )
    time_map = upsert_dim_time(df["date_parsed"])
    location_map = upsert_dim_location(df)
    company_map = upsert_dim_company(df)
    position_map = upsert_dim_position(df)
    platform_map = upsert_dim_platform(df["platform_norm"].dropna().unique().tolist())
    skill_map = upsert_dim_skill(skills_df) if not skills_df.empty else {}
    job_id_map = load_fact_job_posting(df, time_map, location_map, company_map, position_map, platform_map)
    if not skills_df.empty and skill_map:
        load_bridge_job_skill(skills_df, job_id_map, skill_map)
    if periodic_embeddings_path and Path(periodic_embeddings_path).exists():
        npz = np.load(periodic_embeddings_path, allow_pickle=True)
        if npz["jobbert"].shape[0] > 0:
            load_embeddings(df, job_id_map, npz["jobbert"], npz["sbert"])


def task_refresh_views():
    import sys, os, psycopg2, psycopg2.extras
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from db import get_connection
    conn = get_connection()
    conn.autocommit = True
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT schemaname || '.' || matviewname AS full_name FROM pg_matviews WHERE schemaname = 'public';"
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
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. Dicari di: {candidates}")
    for sub in ["", "extract", "transform", "load", "analysis"]:
        p = os.path.join(root, sub) if sub else root
        if p not in sys.path:
            sys.path.insert(0, p)
    from analysis.forecasting import run_forecasting
    from db import get_engine
    run_forecasting(engine=get_engine())


def _epo(task_id, python_callable, op_kwargs=None, retries=1, execution_timeout=timedelta(hours=4)):
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
    dag_id="talent-trackk_group3",
    default_args=default_args,
    description="TalentTrack: Full ETL + Forecasting pipeline",
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["talent-trackk", "dwh", "etl"],
) as dag:

    t_init_db = _epo("init_db", task_init_db)

    t_seed_dim_time = _epo("seed_dim_time", task_seed_dim_time)

    t_extract_kaggle = _epo("extract_kaggle", task_extract_kaggle)

    t_extract_periodic = _epo(
        "extract_periodic",
        task_extract_periodic,
        {"execution_date": "{{ ds }}"},
    )

    t_preprocess_kaggle = _epo(
        "preprocess_kaggle",
        task_preprocess_kaggle,
        {"kaggle_raw_path": "{{ ti.xcom_pull(task_ids='extract_kaggle') }}"},
    )

    t_preprocess_periodic = _epo(
        "preprocess_periodic",
        task_preprocess_periodic,
        {"periodic_raw_path": "{{ ti.xcom_pull(task_ids='extract_periodic') }}"},
    )

    t_ner_kaggle = _epo(
        "ner_kaggle",
        task_ner_kaggle,
        {"kaggle_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_kaggle') }}"},
    )

    t_ner_periodic = _epo(
        "ner_periodic",
        task_ner_periodic,
        {"periodic_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_periodic') }}"},
    )

    t_embed_kaggle = _epo(
        "embed_kaggle",
        task_embed_kaggle,
        {"kaggle_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_kaggle') }}"},
        retries=0,
        execution_timeout=timedelta(hours=6),
    )

    t_embed_periodic = _epo(
        "embed_periodic",
        task_embed_periodic,
        {"periodic_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_periodic') }}"},
        retries=0,
        execution_timeout=timedelta(hours=2),
    )

    t_dedup_kaggle = _epo(
        "dedup_kaggle",
        task_dedup_kaggle,
        {
            "kaggle_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_kaggle') }}",
            "kaggle_embeddings_path": "{{ ti.xcom_pull(task_ids='embed_kaggle') }}",
        },
    )

    t_dedup_periodic = _epo(
        "dedup_periodic",
        task_dedup_periodic,
        {
            "periodic_preprocessed_path": "{{ ti.xcom_pull(task_ids='preprocess_periodic') }}",
            "periodic_embeddings_path": "{{ ti.xcom_pull(task_ids='embed_periodic') }}",
        },
    )

    t_load_kaggle = _epo(
        "load_kaggle",
        task_load_kaggle,
        {
            "kaggle_deduped_path": "{{ ti.xcom_pull(task_ids='dedup_kaggle') }}",
            "kaggle_skills_path": "{{ ti.xcom_pull(task_ids='ner_kaggle') }}",
            "kaggle_embeddings_path": "{{ ti.xcom_pull(task_ids='embed_kaggle') }}",
        },
    )

    t_load_periodic = _epo(
        "load_periodic",
        task_load_periodic,
        {
            "periodic_deduped_path": "{{ ti.xcom_pull(task_ids='dedup_periodic') }}",
            "periodic_skills_path": "{{ ti.xcom_pull(task_ids='ner_periodic') }}",
            "periodic_embeddings_path": "{{ ti.xcom_pull(task_ids='embed_periodic') }}",
        },
    )

    t_refresh_views = _epo("refresh_views", task_refresh_views)
    t_run_forecasting = _epo("run_forecasting", task_run_forecasting)

    t_init_db >> t_seed_dim_time >> [t_extract_kaggle, t_extract_periodic]

    t_extract_kaggle >> t_preprocess_kaggle
    t_preprocess_kaggle >> [t_ner_kaggle, t_embed_kaggle]
    [t_ner_kaggle, t_embed_kaggle] >> t_dedup_kaggle
    t_dedup_kaggle >> t_load_kaggle

    t_extract_periodic >> t_preprocess_periodic
    t_preprocess_periodic >> [t_ner_periodic, t_embed_periodic]
    [t_ner_periodic, t_embed_periodic] >> t_dedup_periodic
    t_dedup_periodic >> t_load_periodic

    [t_load_kaggle, t_load_periodic] >> t_refresh_views >> t_run_forecasting
import sys
import os
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

_CANDIDATE_ROOTS = [
    "/opt/airflow/dags/inter24-dag/talent-trackk",
    "/home/inter24/dags/talent-trackk",
    "/home/inter24/inter24-dag/talent-trackk",
]

def _inject_paths():
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(
            f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}"
        )
    for p in [root,
              os.path.join(root, "extract"),
              os.path.join(root, "transform"),
              os.path.join(root, "load"),
              os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def task_init_db(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    import db
    db.run_ddl(os.path.join(root, "sql", "ddl.sql"))


def task_extract_kaggle(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from extract.extract_kaggle import extract_kaggle
    out = extract_kaggle()
    context["ti"].xcom_push(key="kaggle_raw_path", value=str(out))


def task_extract_periodic(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from extract.extract_periodic import scrape_periodic
    out = scrape_periodic(execution_date=context["ds"])
    context["ti"].xcom_push(key="periodic_raw_path", value=str(out))


def task_preprocess_kaggle(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from pathlib import Path
    from transform.preprocess import preprocess_file
    path = context["ti"].xcom_pull(key="kaggle_raw_path", task_ids="extract_kaggle")
    if not path or not Path(path).exists():
        fallback = os.path.join(root, "data", "raw", "kaggle_staged.parquet")
        if not os.path.exists(fallback):
            raise FileNotFoundError(
                f"Kaggle raw parquet tidak ditemukan di {path} maupun {fallback}"
            )
        path = fallback
    out = preprocess_file(path, source_label="kaggle_2024")
    context["ti"].xcom_push(key="kaggle_preprocessed_path", value=str(out))


def task_preprocess_periodic(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from pathlib import Path
    from transform.preprocess import preprocess_file
    path = context["ti"].xcom_pull(key="periodic_raw_path", task_ids="extract_periodic")
    if not path or not Path(path).exists():
        return
    out = preprocess_file(path, source_label="periodic")
    context["ti"].xcom_push(key="periodic_preprocessed_path", value=str(out))


def task_ner_kaggle(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.ner_skills import run_ner
    path = context["ti"].xcom_pull(key="kaggle_preprocessed_path", task_ids="preprocess_kaggle")
    if not path:
        return
    out = run_ner(path)
    context["ti"].xcom_push(key="kaggle_skills_path", value=str(out))


def task_ner_periodic(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.ner_skills import run_ner
    path = context["ti"].xcom_pull(key="periodic_preprocessed_path", task_ids="preprocess_periodic")
    if not path:
        return
    out = run_ner(path)
    context["ti"].xcom_push(key="periodic_skills_path", value=str(out))


def task_embed_kaggle(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.embed import compute_and_save_embeddings
    path = context["ti"].xcom_pull(key="kaggle_preprocessed_path", task_ids="preprocess_kaggle")
    if not path:
        return
    out = compute_and_save_embeddings(path)
    context["ti"].xcom_push(key="kaggle_embeddings_path", value=str(out))


def task_embed_periodic(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.embed import compute_and_save_embeddings
    path = context["ti"].xcom_pull(key="periodic_preprocessed_path", task_ids="preprocess_periodic")
    if not path:
        return
    out = compute_and_save_embeddings(path)
    context["ti"].xcom_push(key="periodic_embeddings_path", value=str(out))


def task_dedup_kaggle(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.dedup import run_dedup
    prep_path = context["ti"].xcom_pull(key="kaggle_preprocessed_path", task_ids="preprocess_kaggle")
    emb_path  = context["ti"].xcom_pull(key="kaggle_embeddings_path",   task_ids="embed_kaggle")
    if not prep_path:
        return
    out = run_dedup(prep_path, embeddings_path=emb_path)
    context["ti"].xcom_push(key="kaggle_deduped_path", value=str(out))


def task_dedup_periodic(**context):
    import sys, os, psycopg2.extras
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from transform.dedup import run_dedup
    from db import get_connection
    prep_path = context["ti"].xcom_pull(key="periodic_preprocessed_path", task_ids="preprocess_periodic")
    emb_path  = context["ti"].xcom_pull(key="periodic_embeddings_path",   task_ids="embed_periodic")
    if not prep_path:
        return
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT DISTINCT source_hash FROM fact_job_posting;")
        existing_hashes = {r["source_hash"] for r in cur.fetchall()}
    finally:
        conn.close()
    out = run_dedup(prep_path, embeddings_path=emb_path, existing_hashes=existing_hashes)
    context["ti"].xcom_push(key="periodic_deduped_path", value=str(out))


def task_load_kaggle(**context):
    import sys, os, numpy as np, pandas as pd
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from pathlib import Path
    from load.load_dimensions import (
        upsert_dim_time, upsert_dim_location, upsert_dim_company,
        upsert_dim_position, upsert_dim_platform, upsert_dim_skill,
    )
    from load.load_facts import load_fact_job_posting, load_bridge_job_skill, load_embeddings

    deduped_path = context["ti"].xcom_pull(key="kaggle_deduped_path",   task_ids="dedup_kaggle")
    skills_path  = context["ti"].xcom_pull(key="kaggle_skills_path",    task_ids="ner_kaggle")
    emb_path     = context["ti"].xcom_pull(key="kaggle_embeddings_path", task_ids="embed_kaggle")

    if not deduped_path or not Path(deduped_path).exists():
        return
    df = pd.read_parquet(deduped_path)
    if df.empty:
        return
    skills_df = pd.read_parquet(skills_path) if (skills_path and Path(skills_path).exists()) else pd.DataFrame()

    time_map     = upsert_dim_time(df["date_parsed"])
    location_map = upsert_dim_location(df)
    company_map  = upsert_dim_company(df)
    position_map = upsert_dim_position(df)
    platform_map = upsert_dim_platform(df["platform_norm"].dropna().unique().tolist())
    skill_map    = upsert_dim_skill(skills_df) if not skills_df.empty else {}

    job_id_map = load_fact_job_posting(df, time_map, location_map, company_map, position_map, platform_map)
    if not skills_df.empty and skill_map:
        load_bridge_job_skill(skills_df, job_id_map, skill_map)
    if emb_path and Path(emb_path).exists():
        npz = np.load(emb_path, allow_pickle=True)
        if npz["jobbert"].shape[0] > 0:
            load_embeddings(df, job_id_map, npz["jobbert"], npz["sbert"])


def task_load_periodic(**context):
    import sys, os, numpy as np, pandas as pd
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from pathlib import Path
    from load.load_dimensions import (
        upsert_dim_time, upsert_dim_location, upsert_dim_company,
        upsert_dim_position, upsert_dim_platform, upsert_dim_skill,
    )
    from load.load_facts import load_fact_job_posting, load_bridge_job_skill, load_embeddings

    deduped_path = context["ti"].xcom_pull(key="periodic_deduped_path",   task_ids="dedup_periodic")
    skills_path  = context["ti"].xcom_pull(key="periodic_skills_path",    task_ids="ner_periodic")
    emb_path     = context["ti"].xcom_pull(key="periodic_embeddings_path", task_ids="embed_periodic")

    if not deduped_path or not Path(deduped_path).exists():
        return
    df = pd.read_parquet(deduped_path)
    if df.empty:
        return
    skills_df = pd.read_parquet(skills_path) if (skills_path and Path(skills_path).exists()) else pd.DataFrame()

    time_map     = upsert_dim_time(df["date_parsed"])
    location_map = upsert_dim_location(df)
    company_map  = upsert_dim_company(df)
    position_map = upsert_dim_position(df)
    platform_map = upsert_dim_platform(df["platform_norm"].dropna().unique().tolist())
    skill_map    = upsert_dim_skill(skills_df) if not skills_df.empty else {}

    job_id_map = load_fact_job_posting(df, time_map, location_map, company_map, position_map, platform_map)
    if not skills_df.empty and skill_map:
        load_bridge_job_skill(skills_df, job_id_map, skill_map)
    if emb_path and Path(emb_path).exists():
        npz = np.load(emb_path, allow_pickle=True)
        if npz["jobbert"].shape[0] > 0:
            load_embeddings(df, job_id_map, npz["jobbert"], npz["sbert"])


def task_refresh_views(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from db import refresh_materialized_views
    refresh_materialized_views()


def task_run_forecasting(**context):
    import sys, os
    candidates = [
        "/opt/airflow/dags/inter24-dag/talent-trackk",
        "/home/inter24/dags/talent-trackk",
        "/home/inter24/inter24-dag/talent-trackk",
    ]
    root = next((p for p in candidates if os.path.isfile(os.path.join(p, "db.py"))), None)
    if root is None:
        raise RuntimeError(f"PROJECT_ROOT tidak ditemukan. db.py dicari di: {candidates}")
    for p in [root, os.path.join(root, "extract"), os.path.join(root, "transform"),
              os.path.join(root, "load"), os.path.join(root, "analysis")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from analysis.forecasting import run_forecasting
    from db import get_engine
    run_forecasting(engine=get_engine())


def _epo(task_id, python_callable):
    return ExternalPythonOperator(
        task_id=task_id,
        python=VENV_PYTHON,
        python_callable=python_callable,
        expect_airflow=False,
    )


with DAG(
    dag_id="talent-trackk_pipeline",
    default_args=default_args,
    description="TalentTrack: Full ETL + Forecasting pipeline",
    schedule="0 6 */3 * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["talent-trackk", "dwh", "etl"],
) as dag:

    t_init_db             = _epo("init_db",             task_init_db)
    t_extract_kaggle      = _epo("extract_kaggle",      task_extract_kaggle)
    t_extract_periodic    = _epo("extract_periodic",    task_extract_periodic)
    t_preprocess_kaggle   = _epo("preprocess_kaggle",   task_preprocess_kaggle)
    t_preprocess_periodic = _epo("preprocess_periodic", task_preprocess_periodic)
    t_ner_kaggle          = _epo("ner_kaggle",          task_ner_kaggle)
    t_ner_periodic        = _epo("ner_periodic",        task_ner_periodic)
    t_embed_kaggle        = _epo("embed_kaggle",        task_embed_kaggle)
    t_embed_periodic      = _epo("embed_periodic",      task_embed_periodic)
    t_dedup_kaggle        = _epo("dedup_kaggle",        task_dedup_kaggle)
    t_dedup_periodic      = _epo("dedup_periodic",      task_dedup_periodic)
    t_load_kaggle         = _epo("load_kaggle",         task_load_kaggle)
    t_load_periodic       = _epo("load_periodic",       task_load_periodic)
    t_refresh_views       = _epo("refresh_views",       task_refresh_views)
    t_run_forecasting     = _epo("run_forecasting",     task_run_forecasting)

    t_init_db >> [t_extract_kaggle, t_extract_periodic]

    t_extract_kaggle >> t_preprocess_kaggle
    t_preprocess_kaggle >> [t_ner_kaggle, t_embed_kaggle]
    [t_ner_kaggle, t_embed_kaggle] >> t_dedup_kaggle
    t_dedup_kaggle >> t_load_kaggle

    t_extract_periodic >> t_preprocess_periodic
    t_preprocess_periodic >> [t_ner_periodic, t_embed_periodic]
    [t_ner_periodic, t_embed_periodic] >> t_dedup_periodic
    t_dedup_periodic >> t_load_periodic

    [t_load_kaggle, t_load_periodic] >> t_refresh_views >> t_run_forecasting
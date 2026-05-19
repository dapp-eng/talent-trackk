import pandas as pd
from db import get_engine


def get_engine_cached():
    return get_engine()


def q_skill_demand_cube(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine_cached()
    query = """
        SELECT
            dt.year,
            dt.quarter,
            dt.month,
            dt.week_label,
            ds.skill_name,
            ds.skill_type,
            ds.skill_domain,
            dp.job_category,
            dp.job_level,
            dl.global_region,
            dl.country,
            dpl.platform_name,
            COUNT(DISTINCT f.job_id)                AS posting_count,
            SUM(CASE WHEN f.is_remote THEN 1 END)   AS remote_count,
            AVG(f.salary_min)                        AS avg_salary_min,
            AVG(f.salary_max)                        AS avg_salary_max
        FROM fact_job_posting f
        JOIN bridge_job_skill b ON f.job_id = b.job_id
        JOIN dim_skill ds        ON b.skill_id = ds.skill_id
        JOIN dim_time dt         ON f.time_id = dt.time_id
        JOIN dim_position dp     ON f.position_id = dp.position_id
        JOIN dim_location dl     ON f.location_id = dl.location_id
        JOIN dim_platform dpl    ON f.platform_id = dpl.platform_id
        GROUP BY CUBE(
            dt.year, dt.quarter, dt.month, dt.week_label,
            ds.skill_name, ds.skill_type, ds.skill_domain,
            dp.job_category, dp.job_level,
            dl.global_region, dl.country,
            dpl.platform_name
        )
        HAVING COUNT(DISTINCT f.job_id) > 0
        ORDER BY dt.year NULLS LAST, dt.month NULLS LAST, posting_count DESC;
    """
    return pd.read_sql(query, engine)


def q_top_skills_per_category(engine=None, top_n: int = 20) -> pd.DataFrame:
    if engine is None:
        engine = get_engine_cached()
    query = f"""
        SELECT
            job_category,
            skill_name,
            skill_domain,
            SUM(posting_count) AS total_postings
        FROM mv_weekly_skill_demand
        GROUP BY job_category, skill_name, skill_domain
        ORDER BY job_category, total_postings DESC
        LIMIT {top_n * 10};
    """
    df = pd.read_sql(query, engine)
    return df.groupby("job_category").head(top_n).reset_index(drop=True)


def q_skill_trend_weekly(skill_name: str, engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine_cached()
    query = """
        SELECT
            week_label, year, week,
            job_category, global_region,
            SUM(posting_count) AS posting_count
        FROM mv_weekly_skill_demand
        WHERE skill_name = %(skill)s
        GROUP BY week_label, year, week, job_category, global_region
        ORDER BY year, week;
    """
    return pd.read_sql(query, engine, params={"skill": skill_name})


def q_platform_comparison(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine_cached()
    query = """
        SELECT
            platform_name, job_category, global_region,
            SUM(posting_count)      AS total_postings,
            AVG(with_salary_count * 1.0 / NULLIF(posting_count, 0)) AS pct_with_salary,
            AVG(remote_count * 1.0 / NULLIF(posting_count, 0))      AS pct_remote
        FROM mv_platform_monthly
        GROUP BY platform_name, job_category, global_region
        ORDER BY total_postings DESC;
    """
    return pd.read_sql(query, engine)


def q_forecast_next_n_weeks(skill_name: str = None, n_weeks: int = 8,
                              engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine_cached()
    base = """
        SELECT
            fsd.forecast_week_label,
            fsd.forecast_year,
            fsd.forecast_week,
            ds.skill_name,
            fsd.job_category,
            fsd.global_region,
            fsd.predicted_count,
            fsd.lower_bound,
            fsd.upper_bound,
            fsd.model_name,
            fsd.generated_at
        FROM forecast_skill_demand fsd
        JOIN dim_skill ds ON fsd.skill_id = ds.skill_id
    """
    if skill_name:
        query = base + " WHERE ds.skill_name = %(skill)s ORDER BY fsd.forecast_week_label LIMIT %(n)s;"
        return pd.read_sql(query, engine, params={"skill": skill_name, "n": n_weeks * 10})
    else:
        query = base + " ORDER BY fsd.forecast_week_label, fsd.predicted_count DESC;"
        return pd.read_sql(query, engine)


def q_salary_by_skill_level(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine_cached()
    query = """
        SELECT
            ds.skill_name,
            ds.skill_domain,
            dp.job_level,
            dp.job_category,
            dl.country,
            COUNT(DISTINCT f.job_id)    AS posting_count,
            AVG(f.salary_min)           AS avg_salary_min,
            AVG(f.salary_max)           AS avg_salary_max,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.salary_max) AS median_salary_max
        FROM fact_job_posting f
        JOIN bridge_job_skill b ON f.job_id = b.job_id
        JOIN dim_skill ds        ON b.skill_id = ds.skill_id
        JOIN dim_position dp     ON f.position_id = dp.position_id
        JOIN dim_location dl     ON f.location_id = dl.location_id
        WHERE f.has_salary = TRUE
        GROUP BY ds.skill_name, ds.skill_domain, dp.job_level, dp.job_category, dl.country
        HAVING COUNT(DISTINCT f.job_id) >= 5
        ORDER BY avg_salary_max DESC NULLS LAST;
    """
    return pd.read_sql(query, engine)

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
            dl.country,
            dl.region,
            dpl.platform_name,
            COUNT(DISTINCT f.job_id) AS posting_count,
            SUM(CASE WHEN f.is_remote THEN 1 END) AS remote_count,
            AVG(f.salary_min) AS avg_salary_min,
            AVG(f.salary_max) AS avg_salary_max
        FROM fact_job_posting f
        JOIN bridge_job_skill b ON f.job_id = b.job_id
        JOIN dim_skill ds ON b.skill_id = ds.skill_id
        JOIN dim_time dt ON f.time_id = dt.time_id
        JOIN dim_position dp ON f.position_id = dp.position_id
        JOIN dim_location dl ON f.location_id = dl.location_id
        JOIN dim_platform dpl ON f.platform_id = dpl.platform_id
        GROUP BY CUBE(
            dt.year, dt.quarter, dt.month, dt.week_label,
            ds.skill_name, ds.skill_type, ds.skill_domain,
            dp.job_category, dp.job_level,
            dl.country, dl.region,
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
            job_category,
            SUM(posting_count) AS posting_count
        FROM mv_weekly_skill_demand
        WHERE skill_name = %(skill)s
        GROUP BY week_label, year, week, job_category
        ORDER BY year, week;
    """
    return pd.read_sql(query, engine, params={"skill": skill_name})


def q_platform_comparison(engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine_cached()
    query = """
        SELECT
            platform_name, job_category, country,
            SUM(posting_count) AS total_postings,
            AVG(with_salary_count * 1.0 / NULLIF(posting_count, 0)) AS pct_with_salary,
            AVG(remote_count * 1.0 / NULLIF(posting_count, 0)) AS pct_remote
        FROM mv_platform_monthly
        GROUP BY platform_name, job_category, country
        ORDER BY total_postings DESC;
    """
    return pd.read_sql(query, engine)


def q_current_top_skills(engine=None, last_n_weeks: int = 4, top_n: int = 30,
                          job_category: str = None) -> pd.DataFrame:
    """
    Skill yang sedang demand saat ini (berdasarkan data historis N minggu terakhir).
    Aggregasi global (seluruh dunia) tanpa filter negara.
    """
    if engine is None:
        engine = get_engine_cached()

    cat_filter = "AND job_category = %(cat)s" if job_category else ""
    query = f"""
        WITH recent_weeks AS (
            SELECT DISTINCT week_label
            FROM mv_weekly_skill_demand
            ORDER BY week_label DESC
            LIMIT %(last_n)s
        )
        SELECT
            skill_name,
            skill_type,
            skill_domain,
            job_category,
            SUM(posting_count) AS total_postings,
            COUNT(DISTINCT week_label) AS weeks_active
        FROM mv_weekly_skill_demand
        WHERE week_label IN (SELECT week_label FROM recent_weeks)
        {cat_filter}
        GROUP BY skill_name, skill_type, skill_domain, job_category
        ORDER BY total_postings DESC
        LIMIT %(top_n)s;
    """
    params = {"last_n": last_n_weeks, "top_n": top_n}
    if job_category:
        params["cat"] = job_category
    return pd.read_sql(query, engine, params=params)


def q_forecast_next_n_weeks(skill_name: str = None, n_weeks: int = 8,
                              engine=None) -> pd.DataFrame:
    """
    Forecast skill demand untuk N minggu ke depan (global, tanpa filter negara).
    """
    if engine is None:
        engine = get_engine_cached()
    base = """
        SELECT
            fsd.forecast_week_label,
            fsd.forecast_year,
            fsd.forecast_week,
            ds.skill_name,
            fsd.job_category,
            fsd.predicted_count,
            fsd.lower_bound,
            fsd.upper_bound,
            fsd.trend_score,
            fsd.model_name,
            fsd.generated_at
        FROM forecast_skill_demand fsd
        JOIN dim_skill ds ON fsd.skill_id = ds.skill_id
    """
    if skill_name:
        query = base + """
            WHERE ds.skill_name = %(skill)s
            ORDER BY fsd.forecast_week_label
            LIMIT %(n)s;
        """
        return pd.read_sql(query, engine, params={"skill": skill_name, "n": n_weeks})
    else:
        query = base + " ORDER BY fsd.forecast_week_label, fsd.predicted_count DESC;"
        return pd.read_sql(query, engine)


def q_trending_skills(top_n: int = 20, min_postings: int = 3,
                       job_category: str = None, engine=None) -> pd.DataFrame:
    """
    Skill yang diprediksi naik demand-nya di masa depan (global).
    trend_score positif = naik, negatif = turun.
    Hanya menampilkan forecast dari current week ke depan.
    """
    if engine is None:
        engine = get_engine_cached()

    cat_filter = "AND fsd.job_category = %(cat)s" if job_category else ""
    query = f"""
        WITH current_week AS (
            SELECT TO_CHAR(CURRENT_DATE, 'IYYY') || '-W' ||
                   LPAD(TO_CHAR(CURRENT_DATE, 'IW'), 2, '0') AS week_label
        )
        SELECT
            ds.skill_name,
            fsd.job_category,
            fsd.forecast_week_label,
            fsd.predicted_count,
            fsd.lower_bound,
            fsd.upper_bound,
            fsd.trend_score,
            fsd.model_name
        FROM forecast_skill_demand fsd
        JOIN dim_skill ds ON fsd.skill_id = ds.skill_id
        CROSS JOIN current_week cw
        WHERE fsd.predicted_count >= %(min_p)s
          AND fsd.forecast_week_label >= cw.week_label
          {cat_filter}
        ORDER BY fsd.trend_score DESC, fsd.predicted_count DESC
        LIMIT %(n)s;
    """
    params = {"min_p": min_postings, "n": top_n}
    if job_category:
        params["cat"] = job_category
    return pd.read_sql(query, engine, params=params)


def q_global_skill_demand_summary(engine=None, top_n: int = 50) -> pd.DataFrame:
    """
    Ringkasan demand skill secara global: gabungkan data historis + forecast.
    Berguna untuk lihat skill mana yang sedang dan akan terus populer.
    """
    if engine is None:
        engine = get_engine_cached()
    query = """
        WITH historical AS (
            SELECT
                skill_name,
                job_category,
                SUM(posting_count) AS historical_count,
                MAX(week_label) AS last_seen_week
            FROM mv_weekly_skill_demand
            GROUP BY skill_name, job_category
        ),
        forecast_agg AS (
            SELECT
                ds.skill_name,
                fsd.job_category,
                SUM(fsd.predicted_count) AS forecasted_count,
                MAX(fsd.trend_score) AS max_trend_score,
                MAX(fsd.model_name) AS model_name
            FROM forecast_skill_demand fsd
            JOIN dim_skill ds ON fsd.skill_id = ds.skill_id
            GROUP BY ds.skill_name, fsd.job_category
        )
        SELECT
            h.skill_name,
            h.job_category,
            h.historical_count,
            h.last_seen_week,
            COALESCE(f.forecasted_count, 0) AS forecasted_count,
            COALESCE(f.max_trend_score, 0) AS trend_score,
            f.model_name
        FROM historical h
        LEFT JOIN forecast_agg f
            ON h.skill_name = f.skill_name AND h.job_category = f.job_category
        ORDER BY h.historical_count DESC
        LIMIT %(top_n)s;
    """
    return pd.read_sql(query, engine, params={"top_n": top_n})


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
            COUNT(DISTINCT f.job_id) AS posting_count,
            AVG(f.salary_min) AS avg_salary_min,
            AVG(f.salary_max) AS avg_salary_max,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.salary_max) AS median_salary_max
        FROM fact_job_posting f
        JOIN bridge_job_skill b ON f.job_id = b.job_id
        JOIN dim_skill ds ON b.skill_id = ds.skill_id
        JOIN dim_position dp ON f.position_id = dp.position_id
        JOIN dim_location dl ON f.location_id = dl.location_id
        WHERE f.has_salary = TRUE
        GROUP BY ds.skill_name, ds.skill_domain, dp.job_level, dp.job_category, dl.country
        HAVING COUNT(DISTINCT f.job_id) >= 5
        ORDER BY avg_salary_max DESC NULLS LAST;
    """
    return pd.read_sql(query, engine)
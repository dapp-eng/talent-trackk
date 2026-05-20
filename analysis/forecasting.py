import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sqlalchemy import text

warnings.filterwarnings("ignore")

from db import get_cursor, get_engine, get_connection
from config import FORECAST_HORIZON_WEEKS, FORECAST_MIN_HISTORY_WEEKS


def _get_weekly_skill_data(engine) -> pd.DataFrame:
    query = text("""
        SELECT
            week_label,
            year,
            week,
            skill_name,
            skill_domain,
            job_category,
            global_region,
            posting_count
        FROM mv_weekly_skill_demand
        ORDER BY year, week;
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
        cols = result.keys()
    df = pd.DataFrame(rows, columns=list(cols))
    return df


def _forecast_series(series: pd.Series, horizon: int, model_name: str) -> dict:
    n = len(series)
    values = series.values.astype(float)

    if model_name == "linear_trend":
        x = np.arange(n)
        coeffs = np.polyfit(x, values, 1)
        x_future = np.arange(n, n + horizon)
        preds = np.polyval(coeffs, x_future)
        preds = np.maximum(preds, 0)
        residuals = values - np.polyval(coeffs, x)
        std = np.std(residuals)
        return {
            "predictions": preds,
            "lower": np.maximum(preds - 1.96 * std, 0),
            "upper": preds + 1.96 * std,
            "model": model_name,
        }

    if model_name == "holt_winters":
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            if n >= 24:
                model = ExponentialSmoothing(
                    values, trend="add", seasonal="add", seasonal_periods=4
                )
            else:
                model = ExponentialSmoothing(values, trend="add")
            fit = model.fit(optimized=True, use_brute=False)
            preds = fit.forecast(horizon)
            preds = np.maximum(preds, 0)
            ci = fit.simulate(horizon, repetitions=100, error="add")
            lower = np.maximum(ci.quantile(0.025, axis=1).values, 0)
            upper = ci.quantile(0.975, axis=1).values
            return {
                "predictions": preds,
                "lower": lower,
                "upper": upper,
                "model": model_name,
            }
        except Exception:
            return _forecast_series(series, horizon, "linear_trend")

    if model_name == "prophet":
        try:
            from prophet import Prophet
            df_p = pd.DataFrame({
                "ds": pd.date_range(start="2024-01-01", periods=n, freq="W"),
                "y": values,
            })
            m = Prophet(weekly_seasonality=False, yearly_seasonality=True,
                        changepoint_prior_scale=0.3, interval_width=0.95)
            m.fit(df_p, iter=300)
            future = m.make_future_dataframe(periods=horizon, freq="W")
            forecast = m.predict(future)
            tail = forecast.tail(horizon)
            return {
                "predictions": np.maximum(tail["yhat"].values, 0),
                "lower": np.maximum(tail["yhat_lower"].values, 0),
                "upper": tail["yhat_upper"].values,
                "model": model_name,
            }
        except Exception:
            return _forecast_series(series, horizon, "holt_winters")

    return _forecast_series(series, horizon, "linear_trend")


def _get_future_week_labels(last_year: int, last_week: int, horizon: int) -> list:
    labels = []
    yr, wk = last_year, last_week
    for _ in range(horizon):
        wk += 1
        if wk > 52:
            wk = 1
            yr += 1
        labels.append((yr, wk, f"{yr}-W{wk:02d}"))
    return labels


def run_forecasting(engine=None, horizon: int = None):
    if engine is None:
        engine = get_engine()
    if horizon is None:
        horizon = FORECAST_HORIZON_WEEKS

    print("Loading weekly skill demand data...")
    df = _get_weekly_skill_data(engine)
    if df.empty:
        print("No data in mv_weekly_skill_demand. Skipping forecasting.")
        return

    group_cols = ["skill_name", "job_category", "global_region"]
    results = []

    groups = df.groupby(group_cols)
    total = len(groups)
    print(f"Forecasting {total} skill-category-region combinations, horizon={horizon} weeks")

    for i, (key, grp) in enumerate(groups):
        grp_sorted = grp.sort_values(["year", "week"]).reset_index(drop=True)
        if len(grp_sorted) < FORECAST_MIN_HISTORY_WEEKS:
            continue

        series = grp_sorted["posting_count"].astype(float)
        last_year = int(grp_sorted.iloc[-1]["year"])
        last_week = int(grp_sorted.iloc[-1]["week"])

        model_name = "holt_winters" if len(series) >= 16 else "linear_trend"
        result = _forecast_series(series, horizon, model_name)

        future_labels = _get_future_week_labels(last_year, last_week, horizon)
        for j, (yr, wk, wk_label) in enumerate(future_labels):
            results.append({
                "skill_name": key[0],
                "job_category": key[1],
                "global_region": key[2],
                "forecast_week_label": wk_label,
                "forecast_year": yr,
                "forecast_week": wk,
                "predicted_count": float(result["predictions"][j]),
                "lower_bound": float(result["lower"][j]),
                "upper_bound": float(result["upper"][j]),
                "model_name": result["model"],
            })

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{total}")

    if not results:
        print("No forecasts generated (insufficient history).")
        return

    forecast_df = pd.DataFrame(results)
    print(f"Generated {len(forecast_df)} forecast rows. Loading to DB...")

    with get_cursor() as cur:
        cur.execute("SELECT skill_id, skill_name FROM dim_skill;")
        skill_map = {r["skill_name"]: r["skill_id"] for r in cur.fetchall()}

    rows = []
    for _, row in forecast_df.iterrows():
        skill_id = skill_map.get(row["skill_name"])
        if skill_id is None:
            continue
        rows.append((
            int(skill_id),
            row["job_category"],
            row["global_region"],
            row["forecast_week_label"],
            int(row["forecast_year"]),
            int(row["forecast_week"]),
            float(row["predicted_count"]),
            float(row["lower_bound"]),
            float(row["upper_bound"]),
            row["model_name"],
        ))

    if not rows:
        print("No forecast rows to insert.")
        return

    import psycopg2.extras
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM forecast_skill_demand WHERE generated_at < NOW() - INTERVAL '7 days';"
        )
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO forecast_skill_demand
                (skill_id, job_category, global_region, forecast_week_label,
                 forecast_year, forecast_week, predicted_count, lower_bound,
                 upper_bound, model_name)
            VALUES %s
            ON CONFLICT DO NOTHING;
            """,
            rows,
            page_size=500,
        )
        conn.commit()
        print(f"Forecast: {len(rows)} rows inserted.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
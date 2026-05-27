import warnings
import logging
import numpy as np
import pandas as pd
import datetime
from sqlalchemy import text

warnings.filterwarnings("ignore")

from db import get_cursor, get_engine, get_connection
from config import FORECAST_HORIZON_WEEKS, FORECAST_MIN_HISTORY_WEEKS

logger = logging.getLogger(__name__)


def _is_sparse(values: np.ndarray, threshold: float = 0.60) -> bool:
    if len(values) == 0:
        return False
    return (np.sum(values == 0) / len(values)) > threshold


def _compute_trend_score(values: np.ndarray, window: int = 4) -> float:
    if len(values) < window * 2:
        if len(values) < 2:
            return 0.0
        half = len(values) // 2
        recent = float(np.mean(values[half:]))
        prior = float(np.mean(values[:half]))
        baseline = max(prior, 0.1)
        return round((recent - prior) / baseline, 4)
    recent = float(np.mean(values[-window:]))
    prior = float(np.mean(values[-window * 2:-window]))
    baseline = max(prior, 0.1)
    return round((recent - prior) / baseline, 4)


def _exp_smoothing(values: np.ndarray, horizon: int) -> dict:
    try:
        if len(values) < 2:
            avg = float(np.mean(values))
            preds = np.full(horizon, max(avg, 0))
            std = 0.0
            return {"predictions": preds, "lower": np.maximum(preds - 1.96 * std, 0), "upper": preds + 1.96 * std, "model": "mean"}
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        fit = SimpleExpSmoothing(values).fit(optimized=True)
        preds = np.maximum(fit.forecast(horizon), 0)
        std = float(np.std(values[-min(len(values), 8):]))
        return {
            "predictions": preds,
            "lower": np.maximum(preds - 1.96 * std, 0),
            "upper": preds + 1.96 * std,
            "model": "exp_smoothing",
        }
    except Exception as e:
        logger.warning(f"Exp smoothing failed: {e}. Falling back to mean.")
        avg = float(np.mean(values))
        std = float(np.std(values)) if len(values) > 1 else 0.0
        preds = np.full(horizon, max(avg, 0))
        return {"predictions": preds, "lower": np.maximum(preds - 1.96 * std, 0), "upper": preds + 1.96 * std, "model": "mean"}


def _croston(values: np.ndarray, horizon: int) -> dict:
    alpha = 0.1
    non_zero_idx = np.where(values > 0)[0]
    if len(non_zero_idx) == 0:
        preds = np.zeros(horizon)
        return {"predictions": preds, "lower": preds.copy(), "upper": preds.copy(), "model": "croston"}
    z = float(values[non_zero_idx[0]])
    p = float(non_zero_idx[0] + 1)
    for i in range(1, len(non_zero_idx)):
        idx = non_zero_idx[i]
        prev_idx = non_zero_idx[i - 1]
        interval = float(idx - prev_idx)
        demand = float(values[idx])
        z = alpha * demand + (1 - alpha) * z
        p = alpha * interval + (1 - alpha) * p
    forecast_val = max(z / p if p > 0 else 0.0, 0.0)
    nz_vals = values[non_zero_idx]
    std = float(np.std(nz_vals)) if len(nz_vals) > 1 else float(forecast_val * 0.3)
    preds = np.full(horizon, forecast_val)
    return {"predictions": preds, "lower": np.maximum(preds - 1.96 * std, 0), "upper": preds + 1.96 * std, "model": "croston"}


def _holt_winters(values: np.ndarray, horizon: int) -> dict:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        if len(values) >= 24:
            model = ExponentialSmoothing(values, trend="add", seasonal="add", seasonal_periods=4)
        else:
            model = ExponentialSmoothing(values, trend="add")
        fit = model.fit(optimized=True, use_brute=False)
        preds = np.maximum(fit.forecast(horizon), 0)
        ci = fit.simulate(horizon, repetitions=100, error="add")
        lower = np.maximum(ci.quantile(0.025, axis=1).values, 0)
        upper = ci.quantile(0.975, axis=1).values
        return {"predictions": preds, "lower": lower, "upper": upper, "model": "holt_winters"}
    except Exception as e:
        logger.warning(f"Holt-Winters failed: {e}. Falling back to exp_smoothing.")
        return _exp_smoothing(values, horizon)


def _prophet(values: np.ndarray, horizon: int) -> dict:
    try:
        from prophet import Prophet
        n = len(values)
        df_p = pd.DataFrame({
            "ds": pd.date_range(start="2024-01-01", periods=n, freq="W"),
            "y": values,
        })
        m = Prophet(
            weekly_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.3,
            interval_width=0.95,
        )
        m.fit(df_p, iter=300)
        future = m.make_future_dataframe(periods=horizon, freq="W")
        forecast = m.predict(future)
        tail = forecast.tail(horizon)
        return {
            "predictions": np.maximum(tail["yhat"].values, 0),
            "lower": np.maximum(tail["yhat_lower"].values, 0),
            "upper": tail["yhat_upper"].values,
            "model": "prophet",
        }
    except Exception as e:
        logger.warning(f"Prophet failed: {e}. Falling back to holt_winters.")
        return _holt_winters(values, horizon)


def _forecast_series(series: pd.Series, horizon: int) -> dict | None:
    values = series.values.astype(float)
    n = len(values)
    if n < FORECAST_MIN_HISTORY_WEEKS:
        return None
    if _is_sparse(values):
        return _croston(values, horizon)
    if n < 8:
        return _exp_smoothing(values, horizon)
    elif n < 24:
        return _holt_winters(values, horizon)
    else:
        return _prophet(values, horizon)


def _get_weekly_skill_data_global(engine) -> pd.DataFrame:
    query = text("""
        SELECT
            week_label,
            year,
            week,
            skill_name,
            skill_domain,
            job_category,
            SUM(posting_count) AS posting_count
        FROM mv_weekly_skill_demand
        GROUP BY week_label, year, week, skill_name, skill_domain, job_category
        ORDER BY year, week;
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=list(cols))


def _get_future_week_labels(last_year: int, last_week: int, horizon: int) -> list:
    d = datetime.date.fromisocalendar(last_year, last_week, 1)
    labels = []
    for i in range(1, horizon + 1):
        d_future = d + datetime.timedelta(weeks=i)
        iso = d_future.isocalendar()
        yr, wk = iso[0], iso[1]
        labels.append((yr, wk, f"{yr}-W{wk:02d}"))
    return labels


def run_forecasting(engine=None, horizon: int = None):
    if engine is None:
        engine = get_engine()
    if horizon is None:
        horizon = FORECAST_HORIZON_WEEKS

    logger.warning("Loading global weekly skill demand data...")
    df = _get_weekly_skill_data_global(engine)
    if df.empty:
        logger.warning("No data in mv_weekly_skill_demand. Skipping forecasting.")
        return

    df = df[df["job_category"] != "Other"].copy()

    group_cols = ["skill_name", "job_category"]
    groups = df.groupby(group_cols)
    total = len(groups)
    logger.warning(f"Global forecasting: {total} skill×category combinations, horizon={horizon} weeks")

    results = []
    skipped = 0

    for i, (key, grp) in enumerate(groups):
        grp_sorted = grp.sort_values(["year", "week"]).reset_index(drop=True)
        series = grp_sorted["posting_count"].astype(float)
        result = _forecast_series(series, horizon)
        if result is None:
            skipped += 1
            continue
        values = series.values.astype(float)
        trend_score = _compute_trend_score(values)
        last_year = int(grp_sorted.iloc[-1]["year"])
        last_week = int(grp_sorted.iloc[-1]["week"])
        future_labels = _get_future_week_labels(last_year, last_week, horizon)
        for j, (yr, wk, wk_label) in enumerate(future_labels):
            results.append({
                "skill_name": key[0],
                "job_category": key[1],
                "forecast_week_label": wk_label,
                "forecast_year": yr,
                "forecast_week": wk,
                "predicted_count": float(result["predictions"][j]),
                "lower_bound": float(result["lower"][j]),
                "upper_bound": float(result["upper"][j]),
                "trend_score": trend_score,
                "model_name": result["model"],
            })
        if (i + 1) % 50 == 0:
            logger.warning(f"  Progress: {i+1}/{total} (skipped: {skipped})")

    logger.warning(f"Skipped {skipped}/{total} groups (history < {FORECAST_MIN_HISTORY_WEEKS} week(s)).")

    if not results:
        logger.warning("No forecasts generated.")
        return

    forecast_df = pd.DataFrame(results)
    logger.warning(f"Generated {len(forecast_df)} forecast rows. Loading to DB...")

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
            row["forecast_week_label"],
            int(row["forecast_year"]),
            int(row["forecast_week"]),
            float(row["predicted_count"]),
            float(row["lower_bound"]),
            float(row["upper_bound"]),
            float(row["trend_score"]),
            row["model_name"],
        ))

    if not rows:
        logger.warning("No forecast rows to insert (skill_id lookup all failed).")
        return

    import psycopg2.extras
    conn = get_connection()
    try:
        cur = conn.cursor()
        generated_at = datetime.datetime.utcnow()
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO forecast_skill_demand
                (skill_id, job_category, forecast_week_label,
                 forecast_year, forecast_week, predicted_count, lower_bound,
                 upper_bound, trend_score, model_name, generated_at)
            VALUES %s
            ON CONFLICT (skill_id, job_category, forecast_week_label)
            DO UPDATE SET
                predicted_count = EXCLUDED.predicted_count,
                lower_bound     = EXCLUDED.lower_bound,
                upper_bound     = EXCLUDED.upper_bound,
                trend_score     = EXCLUDED.trend_score,
                model_name      = EXCLUDED.model_name,
                generated_at    = EXCLUDED.generated_at;
            """,
            [r + (generated_at,) for r in rows],
            page_size=500,
        )
        conn.commit()

        cur2 = conn.cursor()
        cur2.execute("""
            DELETE FROM forecast_skill_demand
            WHERE generated_at < %s;
        """, (generated_at,))
        conn.commit()
        logger.warning(f"Forecast: {len(rows)} rows inserted/updated, old generations cleaned.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
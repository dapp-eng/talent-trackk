import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sqlalchemy import text

warnings.filterwarnings("ignore")

from db import get_cursor, get_engine, get_connection
from config import FORECAST_HORIZON_WEEKS, FORECAST_MIN_HISTORY_WEEKS

logger = logging.getLogger(__name__)


def _is_sparse(values: np.ndarray, threshold: float = 0.60) -> bool:
    if len(values) == 0:
        return False
    return (np.sum(values == 0) / len(values)) > threshold


def _linear_trend(values: np.ndarray, horizon: int) -> dict:
    n = len(values)
    if n == 1:
        pred_val = max(float(values[0]), 0.0)
        preds = np.full(horizon, pred_val)
        return {"predictions": preds, "lower": preds.copy(), "upper": preds.copy(), "model": "linear_trend"}
    x = np.arange(n)
    coeffs = np.polyfit(x, values, 1)
    x_future = np.arange(n, n + horizon)
    preds = np.maximum(np.polyval(coeffs, x_future), 0)
    residuals = values - np.polyval(coeffs, x)
    std = float(np.std(residuals))
    return {
        "predictions": preds,
        "lower": np.maximum(preds - 1.96 * std, 0),
        "upper": preds + 1.96 * std,
        "model": "linear_trend",
    }


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
        z = alpha * demand   + (1 - alpha) * z
        p = alpha * interval + (1 - alpha) * p
    forecast_val = max(z / p if p > 0 else 0.0, 0.0)
    nz_vals = values[non_zero_idx]
    std  = float(np.std(nz_vals)) if len(nz_vals) > 1 else 0.0
    preds = np.full(horizon, forecast_val)
    lower = np.maximum(preds - 1.96 * std, 0)
    upper = preds + 1.96 * std
    return {"predictions": preds, "lower": lower, "upper": upper, "model": "croston"}


def _auto_arima(values: np.ndarray, horizon: int) -> dict:
    try:
        from pmdarima import auto_arima as pm_auto_arima
        model = pm_auto_arima(
            values,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,
            seasonal=False,
            information_criterion="aic",
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
        )
        preds, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
        preds = np.maximum(preds, 0)
        lower = np.maximum(conf_int[:, 0], 0)
        upper = conf_int[:, 1]
        return {"predictions": preds, "lower": lower, "upper": upper, "model": "auto_arima"}
    except Exception as e:
        logger.warning(f"auto_arima failed: {e}. Falling back to linear_trend.")
        return _linear_trend(values, horizon)


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
        logger.warning(f"Holt-Winters failed: {e}. Falling back to linear_trend.")
        return _linear_trend(values, horizon)


def _prophet(values: np.ndarray, horizon: int) -> dict:
    try:
        from prophet import Prophet
        n    = len(values)
        df_p = pd.DataFrame({
            "ds": pd.date_range(start="2024-01-01", periods=n, freq="W"),
            "y":  values,
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
        return _linear_trend(values, horizon)
    elif n < 16:
        return _auto_arima(values, horizon)
    elif n < 52:
        return _holt_winters(values, horizon)
    else:
        return _prophet(values, horizon)


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
    return pd.DataFrame(rows, columns=list(cols))


def _get_future_week_labels(last_year: int, last_week: int, horizon: int) -> list:
    labels = []
    yr, wk = last_year, last_week
    for _ in range(horizon):
        wk += 1
        if wk > 52:
            wk  = 1
            yr += 1
        labels.append((yr, wk, f"{yr}-W{wk:02d}"))
    return labels


def run_forecasting(engine=None, horizon: int = None):
    if engine is None:
        engine  = get_engine()
    if horizon is None:
        horizon = FORECAST_HORIZON_WEEKS

    logger.info("Loading weekly skill demand data...")
    df = _get_weekly_skill_data(engine)
    if df.empty:
        logger.warning("No data in mv_weekly_skill_demand. Skipping forecasting.")
        return

    group_cols = ["skill_name", "job_category", "global_region"]
    groups = df.groupby(group_cols)
    total = len(groups)
    logger.info(
        f"Forecasting {total} combinations, horizon={horizon} weeks, "
        f"min_history={FORECAST_MIN_HISTORY_WEEKS} week(s)"
    )

    results = []
    skipped = 0

    for i, (key, grp) in enumerate(groups):
        grp_sorted = grp.sort_values(["year", "week"]).reset_index(drop=True)
        series = grp_sorted["posting_count"].astype(float)

        result = _forecast_series(series, horizon)
        if result is None:
            skipped += 1
            continue

        last_year = int(grp_sorted.iloc[-1]["year"])
        last_week = int(grp_sorted.iloc[-1]["week"])

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
            logger.info(f"  Progress: {i+1}/{total} (skipped: {skipped})")

    logger.info(
        f"Skipped {skipped}/{total} groups "
        f"(history < {FORECAST_MIN_HISTORY_WEEKS} week(s))."
    )

    if not results:
        logger.warning("No forecasts generated.")
        return

    forecast_df = pd.DataFrame(results)
    logger.info(f"Generated {len(forecast_df)} forecast rows. Loading to DB...")

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
        logger.warning("No forecast rows to insert (skill_id lookup all failed).")
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
        logger.info(f"Forecast: {len(rows)} rows inserted.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
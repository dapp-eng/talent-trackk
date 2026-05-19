import re
import unicodedata
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

REGION_MAP = {
    "usa": "North America", "united states": "North America",
    "us": "North America", "america": "North America",
    "canada": "North America", "mexico": "North America",
    "united kingdom": "Europe", "uk": "Europe", "england": "Europe",
    "germany": "Europe", "france": "Europe", "netherlands": "Europe",
    "spain": "Europe", "italy": "Europe", "sweden": "Europe",
    "norway": "Europe", "denmark": "Europe", "finland": "Europe",
    "switzerland": "Europe", "austria": "Europe", "belgium": "Europe",
    "portugal": "Europe", "ireland": "Europe", "poland": "Europe",
    "czech": "Europe", "romania": "Europe", "ukraine": "Europe",
    "hungary": "Europe", "greece": "Europe",
    "singapore": "Southeast Asia", "indonesia": "Southeast Asia",
    "malaysia": "Southeast Asia", "vietnam": "Southeast Asia",
    "thailand": "Southeast Asia", "philippines": "Southeast Asia",
    "myanmar": "Southeast Asia", "cambodia": "Southeast Asia",
    "india": "South Asia", "pakistan": "South Asia",
    "bangladesh": "South Asia", "sri lanka": "South Asia",
    "australia": "Oceania", "new zealand": "Oceania",
    "china": "East Asia", "japan": "East Asia",
    "south korea": "East Asia", "taiwan": "East Asia",
    "hong kong": "East Asia",
    "brazil": "South America", "argentina": "South America",
    "colombia": "South America", "chile": "South America",
    "peru": "South America", "ecuador": "South America",
    "costa rica": "South America",
    "south africa": "Africa", "nigeria": "Africa",
    "kenya": "Africa", "egypt": "Africa",
    "ghana": "Africa", "ethiopia": "Africa",
    "uae": "Middle East", "dubai": "Middle East",
    "saudi arabia": "Middle East", "israel": "Middle East",
    "turkey": "Middle East", "jordan": "Middle East",
    "remote": "Remote/Global", "worldwide": "Remote/Global",
    "global": "Remote/Global",
}

JOB_LEVEL_KEYWORDS = {
    "Junior":  ["junior", "entry", "entry-level", "associate", "intern",
                "internship", "graduate", "jr", "early career", "trainee"],
    "Mid":     ["mid", "intermediate", "mid-level", "ii", " ii ", "level 2",
                "experienced", "mid level"],
    "Senior":  ["senior", "sr", "lead", "principal", "staff", "iii", " iii ",
                "level 3", "iv", "level 4", "expert", "seasoned"],
    "Manager": ["manager", "director", "head of", "vp ", "vice president",
                "chief", "cto", "cdo", "ciso", "coo", "president", "executive"],
}

JOB_CATEGORY_KEYWORDS = {
    "Data":             ["data scientist", "data analyst", "data engineer",
                         "data architect", "analytics engineer", "bi analyst",
                         "business intelligence", "data steward", "data modeler"],
    "Machine Learning": ["machine learning", "ml engineer", "ai engineer",
                         "deep learning", "nlp engineer", "computer vision",
                         "artificial intelligence", "llm engineer"],
    "Cloud":            ["cloud engineer", "cloud architect", "solutions architect",
                         "platform engineer", "infrastructure engineer",
                         "aws engineer", "gcp engineer", "azure engineer"],
    "DevOps":           ["devops", "site reliability", "sre", "mlops",
                         "platform engineer", "release engineer", "ci/cd"],
    "FinTech":          ["fintech", "quant", "quantitative", "algorithmic",
                         "financial engineer", "risk analyst", "trading"],
    "Engineering":      ["software engineer", "backend", "frontend", "fullstack",
                         "full stack", "full-stack", "mobile engineer",
                         "ios", "android", "java developer", "python developer",
                         "golang", "rust engineer", "scala", "kotlin",
                         "react developer", "vue developer", "node developer",
                         "game developer", "embedded engineer", "robotics"],
    "Management":       ["manager", "director", "head of", "vp of",
                         "chief", "engineering manager", "product manager",
                         "project manager", "scrum master"],
    "Design":           ["ux", "ui ", "designer", "product designer",
                         "graphic designer", "visual designer"],
    "Security":         ["cybersecurity", "security engineer", "network engineer",
                         "penetration tester", "soc analyst", "information security"],
    "Database":         ["database administrator", "dba", "sql developer",
                         "etl developer", "data warehouse", "database engineer"],
}

SALARY_BOUNDS_BY_LEVEL = {
    "Junior": (20000,  130000),
    "Mid": (40000,  220000),
    "Senior": (70000,  400000),
    "Manager": (80000,  600000),
    "Unknown": (15000,  500000),
}


def clean_text(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    if not text.strip():
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s\.,\-\+\#\/\(\)\:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def parse_date(val) -> pd.Timestamp:
    if val is None:
        return pd.NaT
    if isinstance(val, float) and np.isnan(val):
        return pd.NaT
    val_str = str(val).strip().lower()
    if not val_str or val_str in ("nan", "none", "nat", ""):
        return pd.NaT

    relative_map = [
        (r"^today$", 0),
        (r"^yesterday$", 1),
        (r"^(\d+)\s+days?\s+ago$", "days"),
        (r"^(\d+)\s+hours?\s+ago$", "hours"),
        (r"^(\d+)\s+weeks?\s+ago$", "weeks"),
        (r"^(\d+)\s+months?\s+ago$", "months"),
        (r"^just\s+posted$", 0),
        (r"^30\+\s+days\s+ago$", 30),
        (r"^over\s+30\s+days\s+ago$", 30),
    ]
    now = pd.Timestamp.now().normalize()
    for pattern, offset in relative_map:
        m = re.search(pattern, val_str)
        if m:
            if isinstance(offset, int):
                return now - pd.Timedelta(days=offset)
            elif offset == "days":
                return now - pd.Timedelta(days=int(m.group(1)))
            elif offset == "hours":
                return now - pd.Timedelta(hours=int(m.group(1)))
            elif offset == "weeks":
                return now - pd.Timedelta(weeks=int(m.group(1)))
            elif offset == "months":
                return now - pd.DateOffset(months=int(m.group(1)))

    try:
        parsed = pd.to_datetime(val, infer_datetime_format=True, utc=False)
        if pd.isna(parsed):
            return pd.NaT
        return parsed
    except Exception:
        return pd.NaT


def parse_location(loc) -> tuple:
    if loc is None or (isinstance(loc, float) and np.isnan(loc)):
        return None, None, "Unknown", "Other"
    loc_str = str(loc).strip()
    if not loc_str or loc_str.lower() in ("nan", "none", ""):
        return None, None, "Unknown", "Other"

    loc_lower = loc_str.lower()
    region = "Other"
    for keyword, reg in REGION_MAP.items():
        if keyword in loc_lower:
            region = reg
            break

    parts = [p.strip() for p in re.split(r"[,\|]+", loc_str) if p.strip()]
    if len(parts) == 0:
        return None, None, "Unknown", region
    elif len(parts) == 1:
        return None, None, parts[0], region
    elif len(parts) == 2:
        return parts[0], None, parts[1], region
    else:
        return parts[0], parts[1], parts[-1], region


def infer_job_level(title) -> str:
    if not title or (isinstance(title, float) and np.isnan(title)):
        return "Unknown"
    t = str(title).lower()
    for level, keywords in JOB_LEVEL_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return level
    return "Mid"


def infer_job_category(title) -> str:
    if not title or (isinstance(title, float) and np.isnan(title)):
        return "Other"
    t = str(title).lower()
    for category, keywords in JOB_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return category
    return "Other"


def normalize_salary_col(series: pd.Series) -> pd.Series:
    def _parse(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v) if np.isfinite(v) else np.nan
        s = str(v).replace(",", "").replace("$", "").replace("£", "").replace("€", "").strip()
        m = re.search(r"[\d]+(?:\.\d+)?", s)
        if m:
            val = float(m.group())
            if "k" in s.lower():
                val *= 1000
            return val
        return np.nan
    return series.apply(_parse)


def normalize_salary(df: pd.DataFrame) -> pd.DataFrame:
    df["salary_min"] = normalize_salary_col(df.get("salary_min", pd.Series(dtype=float)))
    df["salary_max"] = normalize_salary_col(df.get("salary_max", pd.Series(dtype=float)))

    df["salary_min"] = df["salary_min"].where(df["salary_min"] > 0)
    df["salary_max"] = df["salary_max"].where(df["salary_max"] > 0)

    swap_mask = (df["salary_min"].notna() & df["salary_max"].notna() &
                 (df["salary_min"] > df["salary_max"]))
    df.loc[swap_mask, ["salary_min", "salary_max"]] = (
        df.loc[swap_mask, ["salary_max", "salary_min"]].values
    )

    df["salary_max"] = df["salary_max"].where(
        df["salary_max"] >= df["salary_min"].fillna(0))

    level_col = df.get("job_level", pd.Series(["Unknown"] * len(df)))
    for level, (lo, hi) in SALARY_BOUNDS_BY_LEVEL.items():
        mask = (level_col == level) if isinstance(level_col, pd.Series) else True
        df.loc[mask, "salary_min"] = df.loc[mask, "salary_min"].where(
            df.loc[mask, "salary_min"].between(lo, hi))
        df.loc[mask, "salary_max"] = df.loc[mask, "salary_max"].where(
            df.loc[mask, "salary_max"].between(lo, hi))

    df["has_salary"] = (df["salary_min"].notna() | df["salary_max"].notna())
    return df


def normalize_remote(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if np.isnan(val) if isinstance(val, float) else False:
            return False
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ["true", "yes", "1", "remote",
                                        "work from home", "wfh", "fully remote"]
    return False


def normalize_platform(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Unknown"
    mapping = {
        "linkedin": "LinkedIn", "indeed": "Indeed",
        "glassdoor": "Glassdoor", "zip_recruiter": "ZipRecruiter",
        "ziprecruiter": "ZipRecruiter", "google": "Google Jobs",
        "google jobs": "Google Jobs", "google_jobs": "Google Jobs",
        "kaggle": "Kaggle Historical", "kaggle_2024": "Kaggle Historical",
    }
    return mapping.get(str(val).lower().strip(), str(val).strip().title() or "Unknown")


def preprocess(df: pd.DataFrame, source_label: str = "unknown") -> pd.DataFrame:
    df = df.copy()
    logger.info(f"Preprocessing {len(df)} rows, source={source_label}")

    title_col = df.get("title", pd.Series(dtype=str))
    df["title_clean"] = title_col.apply(clean_text)
    df["description_clean"] = df.get("description", pd.Series(dtype=str)).apply(clean_text)
    df["company_clean"] = df.get("company", pd.Series(dtype=str)).apply(
        lambda x: clean_text(x).title() if clean_text(x) else "Unknown"
    )

    df["date_parsed"] = df.get("date_posted", pd.Series(dtype=object)).apply(parse_date)
    df["date_parsed"] = pd.to_datetime(df["date_parsed"], utc=False, errors="coerce")
    df["date_parsed"] = df["date_parsed"].apply(
        lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, "tzinfo") and x.tzinfo is not None else x
    )

    before = len(df)
    df = df[df["date_parsed"].notna()].copy()
    df = df[df["date_parsed"] >= pd.Timestamp("2022-01-01")].copy()
    df = df[df["date_parsed"] <= pd.Timestamp.now() + pd.Timedelta(days=1)].copy()
    logger.info(f"  Date filter: {before} → {len(df)} rows")

    df["date_parsed"] = df["date_parsed"].dt.normalize()

    loc_parsed = df.get("location", pd.Series(dtype=str)).apply(parse_location)
    df["loc_city"] = loc_parsed.apply(lambda x: x[0])
    df["loc_province"] = loc_parsed.apply(lambda x: x[1])
    df["loc_country"]  = loc_parsed.apply(lambda x: x[2])
    df["global_region"] = loc_parsed.apply(lambda x: x[3])

    df["job_level"] = title_col.apply(infer_job_level)
    df["job_category"] = title_col.apply(infer_job_category)

    df = normalize_salary(df)
    df["is_remote"] = df.get("is_remote", pd.Series(dtype=object)).apply(normalize_remote)
    df["platform_norm"] = df.get("platform", pd.Series(dtype=str)).apply(normalize_platform)
    df["source_label"] = source_label

    if "source_hash" not in df.columns or df["source_hash"].isna().all():
        import hashlib
        def _make_hash(row):
            raw = f"{row.get('company_clean','')}-{row.get('title_clean','')}-{str(row.get('date_parsed',''))}"
            return hashlib.md5(raw.encode()).hexdigest()
        df["source_hash"] = df.apply(_make_hash, axis=1)

    before = len(df)
    df = df.drop_duplicates(subset=["source_hash"])
    logger.info(f"  Dedup: {before} → {len(df)} rows")

    df = df[df["title_clean"].str.len() > 1].copy()
    df = df[df["description_clean"].str.len() >= 20].copy()
    logger.info(f"  Quality filter done: {len(df)} rows remain")

    return df.reset_index(drop=True)


def preprocess_file(input_path: str, source_label: str = "unknown") -> Path:
    p = Path(input_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    df = preprocess(df, source_label=source_label)

    out_path = DATA_PROCESSED_DIR / (p.stem + "_preprocessed.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Preprocessed output: {len(df)} rows → {out_path}")
    return out_path

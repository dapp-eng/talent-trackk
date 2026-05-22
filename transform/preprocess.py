import re
import hashlib
import unicodedata
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from nltk.corpus import stopwords as nltk_stopwords
from config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

_NLTK_STOPWORDS = {}
_NLTK_LANG_MAP = {
    "arabic":     "ar", "danish":   "da", "dutch":     "nl",
    "english":    "en", "finnish":  "fi", "french":    "fr",
    "german":     "de", "greek":    "el", "hungarian": "hu",
    "indonesian": "id", "italian":  "it", "norwegian": "no",
    "portuguese": "pt", "romanian": "ro", "russian":   "ru",
    "spanish":    "es", "swedish":  "sv", "turkish":   "tr",
}

def _load_stopwords():
    global _NLTK_STOPWORDS
    if _NLTK_STOPWORDS:
        return
    for lang_name, iso in _NLTK_LANG_MAP.items():
        try:
            words = set(nltk_stopwords.words(lang_name))
            _NLTK_STOPWORDS[iso] = words
        except Exception:
            pass

_NON_LATIN_RANGES = [
    (0x0400, 0x04FF),
    (0x0600, 0x06FF),
    (0x0900, 0x097F),
    (0x4E00, 0x9FFF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7AF),
    (0x0E00, 0x0E7F),
    (0x0370, 0x03FF),
    (0x0500, 0x052F),
]

def _has_non_latin_script(text: str) -> bool:
    count = 0
    total = 0
    for ch in text[:300]:
        if ch.isalpha():
            total += 1
            cp = ord(ch)
            for lo, hi in _NON_LATIN_RANGES:
                if lo <= cp <= hi:
                    count += 1
                    break
    if total == 0:
        return False
    return (count / total) > 0.3

def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return "unknown"

    _load_stopwords()

    if _has_non_latin_script(text):
        sample = text[:200]
        for ch in sample:
            cp = ord(ch)
            if 0x0600 <= cp <= 0x06FF:
                return "ar"
            if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF:
                return "zh"
            if 0xAC00 <= cp <= 0xD7AF:
                return "ko"
            if 0x0400 <= cp <= 0x04FF:
                return "ru"
            if 0x0900 <= cp <= 0x097F:
                return "hi"
            if 0x0E00 <= cp <= 0x0E7F:
                return "th"
            if 0x0370 <= cp <= 0x03FF:
                return "el"
        return "non-latin"

    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()[:1000]))
    if not words:
        return "unknown"

    best_lang = "unknown"
    best_score = 0.0
    for iso, sw_set in _NLTK_STOPWORDS.items():
        hits = len(words & sw_set)
        score = hits / max(len(words), 1)
        if score > best_score:
            best_score = score
            best_lang = iso

    if best_score < 0.05:
        return "unknown"
    return best_lang


REGION_MAP = {
    "usa": "North America", "united states": "North America",
    "us": "North America", "america": "North America",
    "canada": "North America", "mexico": "North America",
    "méxico": "North America",

    "united kingdom": "Europe", "uk": "Europe", "england": "Europe",
    "germany": "Europe", "deutschland": "Europe", "allemagne": "Europe",
    "france": "Europe", "frankrijk": "Europe", "frankreich": "Europe",
    "netherlands": "Europe", "nederland": "Europe", "pays-bas": "Europe",
    "spain": "Europe", "españa": "Europe", "espagne": "Europe",
    "italy": "Europe", "italia": "Europe", "italie": "Europe",
    "sweden": "Europe", "sverige": "Europe", "suède": "Europe",
    "norway": "Europe", "norge": "Europe", "norvège": "Europe",
    "denmark": "Europe", "danmark": "Europe", "danemark": "Europe",
    "finland": "Europe", "suomi": "Europe", "finlande": "Europe",
    "switzerland": "Europe", "schweiz": "Europe", "suisse": "Europe",
    "austria": "Europe", "österreich": "Europe", "autriche": "Europe",
    "belgium": "Europe", "belgique": "Europe", "belgië": "Europe",
    "portugal": "Europe",
    "ireland": "Europe", "irlande": "Europe", "irland": "Europe",
    "poland": "Europe", "polska": "Europe", "pologne": "Europe",
    "czech": "Europe", "tschechien": "Europe", "tchéquie": "Europe",
    "romania": "Europe", "românia": "Europe", "roumanie": "Europe",
    "ukraine": "Europe", "україна": "Europe",
    "hungary": "Europe", "magyarország": "Europe", "hongrie": "Europe",
    "greece": "Europe", "ελλάδα": "Europe", "grèce": "Europe",

    "singapore": "Southeast Asia", "singapour": "Southeast Asia",
    "indonesia": "Southeast Asia", "indonésie": "Southeast Asia",
    "malaysia": "Southeast Asia", "malaisie": "Southeast Asia",
    "vietnam": "Southeast Asia", "viêt nam": "Southeast Asia",
    "thailand": "Southeast Asia", "thaïlande": "Southeast Asia",
    "philippines": "Southeast Asia",
    "myanmar": "Southeast Asia", "birmanie": "Southeast Asia",
    "cambodia": "Southeast Asia", "cambodge": "Southeast Asia",

    "india": "South Asia", "inde": "South Asia", "indien": "South Asia",
    "pakistan": "South Asia",
    "bangladesh": "South Asia",
    "sri lanka": "South Asia",

    "australia": "Oceania", "australie": "Oceania", "australien": "Oceania",
    "new zealand": "Oceania", "nouvelle-zélande": "Oceania",

    "china": "East Asia", "chine": "East Asia", "china vr": "East Asia",
    "japan": "East Asia", "japon": "East Asia",
    "south korea": "East Asia", "corée du sud": "East Asia", "südkorea": "East Asia",
    "taiwan": "East Asia",
    "hong kong": "East Asia",

    "brazil": "South America", "brasil": "South America", "brésil": "South America",
    "argentina": "South America", "argentine": "South America",
    "colombia": "South America", "colombie": "South America",
    "chile": "South America", "chili": "South America",
    "peru": "South America", "pérou": "South America",
    "ecuador": "South America", "équateur": "South America",
    "costa rica": "South America",

    "south africa": "Africa", "afrique du sud": "Africa", "südafrika": "Africa",
    "nigeria": "Africa", "nigéria": "Africa",
    "kenya": "Africa",
    "egypt": "Africa", "égypte": "Africa", "ägypten": "Africa",
    "ghana": "Africa",
    "ethiopia": "Africa", "éthiopie": "Africa",

    "uae": "Middle East", "dubai": "Middle East",
    "united arab emirates": "Middle East", "émirats arabes unis": "Middle East",
    "saudi arabia": "Middle East", "arabie saoudite": "Middle East",
    "israel": "Middle East", "israël": "Middle East",
    "turkey": "Middle East", "türkiye": "Middle East",
    "jordan": "Middle East", "jordanie": "Middle East",

    "remote": "Remote/Global", "worldwide": "Remote/Global",
    "global": "Remote/Global", "anywhere": "Remote/Global",
    "partout": "Remote/Global", "weltweit": "Remote/Global",
}

REGION_MAP_SORTED = sorted(REGION_MAP.items(), key=lambda x: len(x[0]), reverse=True)

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
    "Junior": (15000, 150000),
    "Mid": (25000, 250000),
    "Senior": (40000, 500000),
    "Manager": (50000, 700000),
    "Unknown": (10000, 700000),
}

PAY_PERIOD_MULTIPLIER = {
    "hourly": 2080,
    "hour": 2080,
    "daily": 260,
    "day": 260,
    "weekly": 52,
    "week": 52,
    "monthly": 12,
    "month": 12,
    "yearly": 1,
    "year": 1,
    "annual": 1,
    "annually": 1,
}

HOURLY_SALARY_THRESHOLD = 500


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
    for keyword, reg in REGION_MAP_SORTED:
        if re.search(r"\b" + re.escape(keyword) + r"\b", loc_lower):
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
    return "Unknown"


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


def normalize_pay_period(df: pd.DataFrame) -> pd.DataFrame:
    period_col = None
    for candidate in ["pay_period", "salary_period", "compensation_type", "pay_type"]:
        if candidate in df.columns:
            period_col = candidate
            break

    if period_col is None:
        min_vals = df["salary_min"].dropna()
        max_vals = df["salary_max"].dropna()
        all_vals = pd.concat([min_vals, max_vals])
        if len(all_vals) > 0:
            likely_hourly = (all_vals < HOURLY_SALARY_THRESHOLD).mean()
            if likely_hourly > 0.5:
                logger.warning(f"  No pay_period column found but >50% values look hourly — applying x2080 to all")
                df["salary_min"] = df["salary_min"] * 2080
                df["salary_max"] = df["salary_max"] * 2080
                df["salary_period_norm"] = "hourly_inferred"
            else:
                df["salary_period_norm"] = "yearly_assumed"
        return df

    def _multiplier(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 1
        return PAY_PERIOD_MULTIPLIER.get(str(val).strip().lower(), 1)

    multipliers = df[period_col].apply(_multiplier)
    df["salary_period_norm"] = df[period_col].apply(
        lambda v: str(v).strip().lower() if pd.notna(v) else "unknown"
    )

    needs_conversion = multipliers != 1
    converted_count = needs_conversion.sum()
    if converted_count > 0:
        logger.warning(f"  Pay period normalization: converting {converted_count} rows to yearly")

    df["salary_min"] = df["salary_min"].where(~needs_conversion, df["salary_min"] * multipliers)
    df["salary_max"] = df["salary_max"].where(~needs_conversion, df["salary_max"] * multipliers)
    return df


def normalize_salary(df: pd.DataFrame) -> pd.DataFrame:
    df["salary_min"] = normalize_salary_col(df.get("salary_min", pd.Series(dtype=float)))
    df["salary_max"] = normalize_salary_col(df.get("salary_max", pd.Series(dtype=float)))

    df["salary_min"] = df["salary_min"].where(df["salary_min"] > 0)
    df["salary_max"] = df["salary_max"].where(df["salary_max"] > 0)

    df = normalize_pay_period(df)

    swap_mask = (df["salary_min"].notna() & df["salary_max"].notna() &
                 (df["salary_min"] > df["salary_max"]))
    df.loc[swap_mask, ["salary_min", "salary_max"]] = (
        df.loc[swap_mask, ["salary_max", "salary_min"]].values
    )

    df["salary_max"] = df["salary_max"].where(
        df["salary_max"] >= df["salary_min"].fillna(0))

    level_col = df.get("job_level", pd.Series(["Unknown"] * len(df)))
    total_nulled = 0
    for level, (lo, hi) in SALARY_BOUNDS_BY_LEVEL.items():
        mask = level_col == level
        before_min = df.loc[mask, "salary_min"].notna().sum()
        before_max = df.loc[mask, "salary_max"].notna().sum()
        df.loc[mask, "salary_min"] = df.loc[mask, "salary_min"].where(
            df.loc[mask, "salary_min"].between(lo, hi))
        df.loc[mask, "salary_max"] = df.loc[mask, "salary_max"].where(
            df.loc[mask, "salary_max"].between(lo, hi))
        nulled = (before_min - df.loc[mask, "salary_min"].notna().sum()) + \
                 (before_max - df.loc[mask, "salary_max"].notna().sum())
        if nulled > 0:
            logger.warning(f"  Salary bounds [{level}] nulled {nulled} values outside ({lo}, {hi})")
        total_nulled += nulled

    if total_nulled > 0:
        logger.warning(f"  Salary bounds total nulled: {total_nulled} values")

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


def _make_source_hash(row: pd.Series) -> str:
    company = str(row.get("company_clean") or "")
    title = str(row.get("title_clean") or "")
    date = str(row.get("date_parsed") or "")
    location = str(row.get("loc_country") or row.get("location") or "")
    raw = f"{company}|{title}|{date}|{location}"
    return hashlib.md5(raw.encode()).hexdigest()


def _tag_languages(df: pd.DataFrame) -> pd.DataFrame:
    logger.warning("  Detecting description languages...")
    df["description_lang"] = df["description_clean"].apply(detect_language)
    lang_counts = df["description_lang"].value_counts().to_dict()
    logger.warning(f"  Language distribution: {lang_counts}")
    non_english = df[~df["description_lang"].isin(["en", "unknown"])]["description_lang"].value_counts().to_dict()
    if non_english:
        logger.warning(f"  Non-English rows (kept, NER still runs): {non_english}")
    return df


def preprocess(df: pd.DataFrame, source_label: str = "unknown") -> pd.DataFrame:
    df = df.copy()
    logger.warning(f"Preprocessing {len(df)} rows, source={source_label}")

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
    after_nat = len(df)
    df = df[df["date_parsed"] >= pd.Timestamp("2023-01-01")].copy()
    after_old = len(df)
    df = df[df["date_parsed"] <= pd.Timestamp.now() + pd.Timedelta(days=1)].copy()
    after_future = len(df)
    logger.warning(
        f"  Date filter: {before} → {after_nat} (dropped {before - after_nat} NaT)"
        f" → {after_old} (dropped {after_nat - after_old} pre-2023)"
        f" → {after_future} (dropped {after_old - after_future} future)"
    )

    df["date_parsed"] = df["date_parsed"].dt.normalize()

    loc_parsed = df.get("location", pd.Series(dtype=str)).apply(parse_location)
    df["loc_city"] = loc_parsed.apply(lambda x: x[0])
    df["loc_province"] = loc_parsed.apply(lambda x: x[1])
    df["loc_country"] = loc_parsed.apply(lambda x: x[2])
    df["global_region"] = loc_parsed.apply(lambda x: x[3])

    df["job_level"] = title_col.apply(infer_job_level)
    df["job_category"] = title_col.apply(infer_job_category)

    df = normalize_salary(df)
    df["is_remote"] = df.get("is_remote", pd.Series(dtype=object)).apply(normalize_remote)
    df["platform_norm"] = df.get("platform", pd.Series(dtype=str)).apply(normalize_platform)
    df["source_label"] = source_label

    df["source_hash"] = df.apply(_make_source_hash, axis=1)

    before = len(df)
    df = df.drop_duplicates(subset=["source_hash"])
    logger.warning(f"  Internal dedup: {before} → {len(df)} rows (removed {before - len(df)})")

    before = len(df)
    df = df[df["title_clean"].str.len() > 1].copy()
    after_title = len(df)
    df = df[df["description_clean"].str.len() >= 10].copy()
    after_desc = len(df)
    logger.warning(
        f"  Quality filter: {before} → {after_title} (dropped {before - after_title} short title)"
        f" → {after_desc} (dropped {after_title - after_desc} short desc)"
    )

    df = _tag_languages(df)

    logger.warning(f"  Preprocessing done: {len(df)} rows remain")
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
    logger.warning(f"Preprocessed output: {len(df)} rows → {out_path}")
    return out_path
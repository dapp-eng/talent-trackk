import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

_DATA_ROOT = Path(os.environ.get("TALENTTRACK_DATA_DIR", "/tmp/talent-trackk"))

DATA_RAW_DIR = _DATA_ROOT / "raw"
DATA_PROCESSED_DIR = _DATA_ROOT / "processed"
MODELS_DIR = _DATA_ROOT / "models"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET_PATH = BASE_DIR / "data" / "raw" / "kaggle_jobs_2024.csv"
PERIODIC_RAW_PATH   = DATA_RAW_DIR / "periodic"

load_dotenv(BASE_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "port": int(os.getenv("PGPORT", 5432)),
    "dbname": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
}

if not all([DB_CONFIG["host"], DB_CONFIG["dbname"], DB_CONFIG["user"], DB_CONFIG["password"]]):
    raise EnvironmentError(
        "Credentials DB tidak lengkap. Pastikan file .env ada di folder talent-trackk/ "
        "dan berisi PGHOST, PGDATABASE, PGUSER, PGPASSWORD."
    )

DB_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

JOBBERT_MODEL = "jjzha/jobbert-base-cased"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

JOBSPY_SEARCH_TERMS = [
    "software engineer", "data engineer", "data scientist", "machine learning engineer",
    "cloud engineer", "devops engineer", "backend engineer", "frontend engineer",
    "fullstack developer", "mobile developer", "ios developer", "android developer",
    "site reliability engineer", "platform engineer", "infrastructure engineer",
    "data analyst", "business analyst", "business intelligence analyst",
    "product manager", "project manager", "scrum master",
    "cybersecurity analyst", "security engineer", "network engineer",
    "database administrator", "sql developer", "etl developer",
    "ai engineer", "nlp engineer", "computer vision engineer",
    "fintech engineer", "quantitative analyst", "financial analyst",
    "ux designer", "ui designer", "graphic designer",
    "marketing analyst", "growth engineer", "data architect",
    "solutions architect", "technical lead", "engineering manager",
    "embedded engineer", "robotics engineer", "game developer",
    "java developer", "python developer", "golang developer",
    "rust engineer", "scala engineer", "kotlin developer",
    "react developer", "vue developer", "angular developer",
    "node developer", "php developer", "ruby developer",
]

JOBSPY_GLOBAL_LOCATIONS = [
    "United States", "United Kingdom", "Canada", "Australia", "Germany",
    "France", "Netherlands", "Sweden", "Switzerland", "Norway",
    "Singapore", "Japan", "South Korea", "India", "Hong Kong",
    "Brazil", "Mexico", "Argentina", "Colombia",
    "South Africa", "Nigeria", "Kenya",
    "Indonesia", "Malaysia", "Thailand", "Philippines", "Vietnam",
    "UAE", "Israel", "Poland", "Spain", "Italy",
    "New Zealand", "Portugal", "Ireland", "Belgium", "Denmark",
    "Czech Republic", "Romania", "Pakistan", "Bangladesh",
    "Taiwan", "China", "Saudi Arabia", "Turkey", "Egypt",
    "Ukraine", "Finland", "Austria", "Hungary", "Greece",
    "Chile", "Peru", "Ecuador", "Costa Rica",
    "Remote", "Worldwide",
]

JOBSPY_RESULTS_PER_SEARCH = 30
JOBSPY_SITES = ["linkedin", "indeed", "glassdoor", "google", "zip_recruiter"]
JOBSPY_HOURS_OLD = 72

JOBBERT_MAX_TOKENS = 512
JOBBERT_CHUNK_OVERLAP = 64
EMBEDDING_DIM = 768
SBERT_DIM = 384

FORECAST_HORIZON_WEEKS = 8
FORECAST_MIN_HISTORY_WEEKS = 12
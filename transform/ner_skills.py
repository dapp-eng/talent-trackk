import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_PROCESSED_DIR

SKILL_VOCABULARY = {
    "Python": ("Programming Language", "General"),
    "SQL": ("Programming Language", "Data"),
    "R": ("Programming Language", "Data"),
    "Scala": ("Programming Language", "Data"),
    "Java": ("Programming Language", "General"),
    "Go": ("Programming Language", "General"),
    "Rust": ("Programming Language", "General"),
    "JavaScript": ("Programming Language", "Web"),
    "TypeScript": ("Programming Language", "Web"),
    "C++": ("Programming Language", "Systems"),
    "C#": ("Programming Language", "Systems"),
    "Spark": ("Big Data", "Data Engineering"),
    "Hadoop": ("Big Data", "Data Engineering"),
    "Kafka": ("Messaging", "Data Engineering"),
    "Flink": ("Stream Processing", "Data Engineering"),
    "Airflow": ("Orchestration", "Data Engineering"),
    "dbt": ("Transformation", "Data Engineering"),
    "Databricks": ("Platform", "Data Engineering"),
    "Snowflake": ("Data Warehouse", "Data Engineering"),
    "Redshift": ("Data Warehouse", "Data Engineering"),
    "BigQuery": ("Data Warehouse", "Data Engineering"),
    "PostgreSQL": ("Database", "Data"),
    "MySQL": ("Database", "Data"),
    "MongoDB": ("Database", "Data"),
    "Cassandra": ("Database", "Data"),
    "Elasticsearch": ("Search", "Data"),
    "Redis": ("Cache", "Data Engineering"),
    "AWS": ("Cloud", "Cloud Infrastructure"),
    "GCP": ("Cloud", "Cloud Infrastructure"),
    "Azure": ("Cloud", "Cloud Infrastructure"),
    "Kubernetes": ("Container Orchestration", "DevOps"),
    "Docker": ("Containerization", "DevOps"),
    "Terraform": ("IaC", "DevOps"),
    "Ansible": ("Configuration Management", "DevOps"),
    "Jenkins": ("CI/CD", "DevOps"),
    "GitHub Actions": ("CI/CD", "DevOps"),
    "TensorFlow": ("ML Framework", "Machine Learning"),
    "PyTorch": ("ML Framework", "Machine Learning"),
    "scikit-learn": ("ML Library", "Machine Learning"),
    "Hugging Face": ("NLP Framework", "Machine Learning"),
    "LangChain": ("LLM Framework", "Machine Learning"),
    "XGBoost": ("ML Library", "Machine Learning"),
    "LightGBM": ("ML Library", "Machine Learning"),
    "Pandas": ("Data Library", "Data"),
    "NumPy": ("Data Library", "Data"),
    "Power BI": ("BI Tool", "Business Intelligence"),
    "Tableau": ("BI Tool", "Business Intelligence"),
    "Looker": ("BI Tool", "Business Intelligence"),
    "Metabase": ("BI Tool", "Business Intelligence"),
    "Excel": ("Spreadsheet", "Business Intelligence"),
    "MLflow": ("MLOps", "Machine Learning"),
    "Kubeflow": ("MLOps", "Machine Learning"),
    "FastAPI": ("Web Framework", "Engineering"),
    "Django": ("Web Framework", "Engineering"),
    "Flask": ("Web Framework", "Engineering"),
    "React": ("Frontend", "Engineering"),
    "Vue": ("Frontend", "Engineering"),
    "Node.js": ("Runtime", "Engineering"),
    "REST API": ("Architecture", "Engineering"),
    "GraphQL": ("API", "Engineering"),
    "microservices": ("Architecture", "Engineering"),
    "CI/CD": ("DevOps Practice", "DevOps"),
    "Git": ("Version Control", "General"),
    "Linux": ("OS", "Systems"),
    "Statistics": ("Math", "Data"),
    "Machine Learning": ("Concept", "Machine Learning"),
    "Deep Learning": ("Concept", "Machine Learning"),
    "NLP": ("Concept", "Machine Learning"),
    "Computer Vision": ("Concept", "Machine Learning"),
    "Time Series": ("Concept", "Data"),
    "Forecasting": ("Concept", "Data"),
    "A/B Testing": ("Analytics", "Data"),
    "Agile": ("Methodology", "General"),
    "Scrum": ("Methodology", "General"),
    "Communication": ("Soft Skill", "General"),
    "Leadership": ("Soft Skill", "General"),
    "Problem Solving": ("Soft Skill", "General"),
    "Team Player": ("Soft Skill", "General"),
    "S3": ("Cloud Storage", "Cloud Infrastructure"),
    "Lambda": ("Serverless", "Cloud Infrastructure"),
    "EC2": ("Compute", "Cloud Infrastructure"),
    "SageMaker": ("ML Platform", "Machine Learning"),
    "Vertex AI": ("ML Platform", "Machine Learning"),
    "Azure ML": ("ML Platform", "Machine Learning"),
    "Prometheus": ("Monitoring", "DevOps"),
    "Grafana": ("Monitoring", "DevOps"),
}

_COMPILED_PATTERNS = None


def _get_patterns():
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS is None:
        _COMPILED_PATTERNS = {}
        for skill in SKILL_VOCABULARY:
            escaped = re.escape(skill)
            _COMPILED_PATTERNS[skill] = re.compile(
                r"(?<![a-zA-Z0-9\-])" + escaped + r"(?![a-zA-Z0-9\-])",
                re.IGNORECASE,
            )
    return _COMPILED_PATTERNS


def extract_skills_from_text(text: str) -> list:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    patterns = _get_patterns()
    found = []
    for skill, pattern in patterns.items():
        if pattern.search(text):
            found.append(skill)
    return found


def extract_skills_dataframe(df: pd.DataFrame,
                              text_col: str = "description_clean") -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        skills = extract_skills_from_text(row.get(text_col, ""))
        src_hash = row.get("source_hash", str(idx))
        for skill in skills:
            stype, sdomain = SKILL_VOCABULARY.get(skill, ("Other", "Other"))
            records.append({
                "source_hash": src_hash,
                "skill_name": skill,
                "skill_type": stype,
                "skill_domain": sdomain,
                "extraction_confidence": 0.90,
            })
    return pd.DataFrame(records)


def run_ner(preprocessed_path: str) -> Path:
    df = pd.read_parquet(preprocessed_path)
    print(f"Running NER on {len(df)} rows...")
    skills_df = extract_skills_dataframe(df)
    out_path = DATA_PROCESSED_DIR / (Path(preprocessed_path).stem + "_skills.parquet")
    skills_df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"NER done: {len(skills_df)} skill records → {out_path}")
    return out_path

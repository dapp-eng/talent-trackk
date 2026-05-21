# TalentTrack - Job Market Intelligence Data Warehouse

ETL pipeline for global job market analysis with embeddings and forecasting.

## Project Structure

```
talent-trackk/
├── talenttrack_dag.py          # Airflow DAG orchestration
├── config.py                   # Configuration & paths
├── db.py                       # PostgreSQL utilities
├── init_setup.py               # One-time DB initialization
├── requirements_local.txt      # Dependencies (Local not server)
├── sql/ddl.sql                 # Database schema
├── extract/                    # Data extraction
│   ├── extract_kaggle.py       # Kaggle dataset loading
│   └── extract_periodic.py     # JobSpy web scraping
├── transform/                  # Data transformation
│   ├── preprocess.py           # Cleaning & normalization
│   ├── embed.py                # JobBERT / SBERT embeddings
│   ├── ner_skills.py           # Skill extraction
│   └── dedup.py                # Deduplication
├── load/                       # Database loading
│   ├── load_dimensions.py      # Dimension tables
│   └── load_facts.py           # Fact & bridge tables
├── analysis/                   # Analytics
│   ├── forecasting.py          # Time series forecasting
│   └── olap_queries.py         # CUBE queries
├── atoti_app/                  # BI interface
│   └── datamart.ipynb          # Analytics notebook
└── data/
    └── raw/                    # Raw data storage
```

## Pipeline Features

- **Extraction**: Kaggle dataset + JobSpy real-time scraping
- **Transformation**: Text cleaning, date parsing, salary normalization, location mapping
- **Embeddings**: JobBERT chunked encoding + SBERT sentence embeddings
- **Deduplication**: MD5 hash + cosine similarity filtering
- **Loading**: Batch insert with conflict handling (source_hash, date)
- **Analysis**: Forecasting, CUBE aggregations
- **BI**: Atoti dashboards
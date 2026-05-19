CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS dim_time (
    time_id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    week INT NOT NULL,
    month INT NOT NULL,
    quarter INT NOT NULL,
    year INT NOT NULL,
    week_label TEXT GENERATED ALWAYS AS (
                    year::TEXT || '-W' || LPAD(week::TEXT, 2, '0')
                ) STORED
);

CREATE INDEX IF NOT EXISTS idx_dim_time_year_month ON dim_time (year, month);
CREATE INDEX IF NOT EXISTS idx_dim_time_week_label ON dim_time (week_label);

CREATE TABLE IF NOT EXISTS dim_location (
    location_id SERIAL PRIMARY KEY,
    city TEXT,
    province_state TEXT,
    country TEXT NOT NULL DEFAULT 'Unknown',
    global_region TEXT NOT NULL DEFAULT 'Other',
    UNIQUE (city, province_state, country, global_region)
);

CREATE INDEX IF NOT EXISTS idx_dim_location_country ON dim_location (country);
CREATE INDEX IF NOT EXISTS idx_dim_location_region ON dim_location (global_region);

CREATE TABLE IF NOT EXISTS dim_company (
    company_id SERIAL PRIMARY KEY,
    company_name TEXT NOT NULL UNIQUE,
    industry_sector TEXT,
    company_size_category TEXT
);

CREATE INDEX IF NOT EXISTS idx_dim_company_name ON dim_company USING gin (company_name gin_trgm_ops);

CREATE TABLE IF NOT EXISTS dim_position (
    position_id SERIAL PRIMARY KEY,
    normalized_job_title TEXT NOT NULL,
    job_level TEXT NOT NULL DEFAULT 'Unknown',
    job_category TEXT NOT NULL DEFAULT 'Other',
    UNIQUE (normalized_job_title, job_level, job_category)
);

CREATE INDEX IF NOT EXISTS idx_dim_position_category ON dim_position (job_category);
CREATE INDEX IF NOT EXISTS idx_dim_position_level ON dim_position (job_level);

CREATE TABLE IF NOT EXISTS dim_platform (
    platform_id SERIAL PRIMARY KEY,
    platform_name TEXT NOT NULL UNIQUE,
    regional_focus TEXT
);

CREATE TABLE IF NOT EXISTS dim_skill (
    skill_id SERIAL PRIMARY KEY,
    skill_name TEXT NOT NULL UNIQUE,
    skill_type TEXT,
    skill_domain TEXT
);

CREATE INDEX IF NOT EXISTS idx_dim_skill_type   ON dim_skill (skill_type);
CREATE INDEX IF NOT EXISTS idx_dim_skill_domain ON dim_skill (skill_domain);

CREATE TABLE IF NOT EXISTS fact_job_posting (
    job_id BIGSERIAL,
    time_id INT NOT NULL REFERENCES dim_time(time_id),
    location_id INT REFERENCES dim_location(location_id),
    company_id INT REFERENCES dim_company(company_id),
    position_id INT REFERENCES dim_position(position_id),
    platform_id INT REFERENCES dim_platform(platform_id),
    posting_count INT DEFAULT 1,
    job_age_days INT,
    has_salary BOOLEAN DEFAULT FALSE,
    is_remote BOOLEAN DEFAULT FALSE,
    salary_min NUMERIC(15,2),
    salary_max NUMERIC(15,2),
    source_hash TEXT NOT NULL,
    raw_title TEXT,
    PRIMARY KEY (job_id, time_id),
    UNIQUE (source_hash, time_id)
) PARTITION BY RANGE (time_id);

CREATE TABLE IF NOT EXISTS fact_job_posting_2024 PARTITION OF fact_job_posting
    FOR VALUES FROM (1) TO (367);

CREATE TABLE IF NOT EXISTS fact_job_posting_2025 PARTITION OF fact_job_posting
    FOR VALUES FROM (367) TO (732);

CREATE TABLE IF NOT EXISTS fact_job_posting_2026 PARTITION OF fact_job_posting
    FOR VALUES FROM (732) TO (1097);

CREATE TABLE IF NOT EXISTS fact_job_posting_future PARTITION OF fact_job_posting
    FOR VALUES FROM (1097) TO (MAXVALUE);

CREATE INDEX IF NOT EXISTS idx_fact_time ON fact_job_posting (time_id);
CREATE INDEX IF NOT EXISTS idx_fact_location ON fact_job_posting (location_id);
CREATE INDEX IF NOT EXISTS idx_fact_company ON fact_job_posting (company_id);
CREATE INDEX IF NOT EXISTS idx_fact_position ON fact_job_posting (position_id);
CREATE INDEX IF NOT EXISTS idx_fact_platform ON fact_job_posting (platform_id);
CREATE INDEX IF NOT EXISTS idx_fact_source_hash ON fact_job_posting (source_hash);

CREATE TABLE IF NOT EXISTS bridge_job_skill (
    job_id BIGINT NOT NULL,
    skill_id INT NOT NULL REFERENCES dim_skill(skill_id),
    extraction_confidence NUMERIC(5,4) DEFAULT 0.9,
    PRIMARY KEY (job_id, skill_id)
);

CREATE INDEX IF NOT EXISTS idx_bridge_skill ON bridge_job_skill (skill_id);
CREATE INDEX IF NOT EXISTS idx_bridge_job   ON bridge_job_skill (job_id);

CREATE TABLE IF NOT EXISTS job_embeddings (
    job_id BIGINT PRIMARY KEY,
    jobbert_vector vector(768),
    sbert_vector vector(384)
);

CREATE INDEX IF NOT EXISTS idx_job_embeddings_jobbert
    ON job_embeddings USING ivfflat (jobbert_vector vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_job_embeddings_sbert
    ON job_embeddings USING ivfflat (sbert_vector vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS forecast_skill_demand (
    forecast_id BIGSERIAL PRIMARY KEY,
    skill_id INT REFERENCES dim_skill(skill_id),
    job_category TEXT,
    global_region TEXT,
    forecast_week_label TEXT NOT NULL,
    forecast_year INT NOT NULL,
    forecast_week INT NOT NULL,
    predicted_count NUMERIC(12,4),
    lower_bound NUMERIC(12,4),
    upper_bound NUMERIC(12,4),
    model_name TEXT,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (skill_id, job_category, global_region, forecast_week_label)
);

CREATE INDEX IF NOT EXISTS idx_forecast_skill ON forecast_skill_demand (skill_id);
CREATE INDEX IF NOT EXISTS idx_forecast_week  ON forecast_skill_demand (forecast_week_label);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_weekly_skill_demand AS
SELECT
    dt.year,
    dt.week,
    dt.week_label,
    ds.skill_name,
    ds.skill_type,
    ds.skill_domain,
    dp.job_category,
    dl.global_region,
    dl.country,
    COUNT(DISTINCT f.job_id) AS posting_count,
    AVG(f.salary_min) AS avg_salary_min,
    AVG(f.salary_max) AS avg_salary_max
FROM fact_job_posting f
JOIN bridge_job_skill b ON f.job_id = b.job_id
JOIN dim_skill ds ON b.skill_id = ds.skill_id
JOIN dim_time dt ON f.time_id = dt.time_id
JOIN dim_position dp ON f.position_id = dp.position_id
JOIN dim_location dl ON f.location_id = dl.location_id
GROUP BY dt.year, dt.week, dt.week_label,
         ds.skill_name, ds.skill_type, ds.skill_domain,
         dp.job_category, dl.global_region, dl.country
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_weekly_skill
    ON mv_weekly_skill_demand (week_label, skill_name, job_category, country);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_platform_monthly AS
SELECT
    dt.year,
    dt.month,
    dpl.platform_name,
    dp.job_category,
    dl.global_region,
    COUNT(DISTINCT f.job_id) AS posting_count,
    SUM(CASE WHEN f.is_remote THEN 1 ELSE 0 END) AS remote_count,
    SUM(CASE WHEN f.has_salary THEN 1 ELSE 0 END) AS with_salary_count
FROM fact_job_posting f
JOIN dim_time dt ON f.time_id = dt.time_id
JOIN dim_platform dpl ON f.platform_id = dpl.platform_id
JOIN dim_position dp ON f.position_id = dp.position_id
JOIN dim_location dl ON f.location_id = dl.location_id
GROUP BY dt.year, dt.month, dpl.platform_name, dp.job_category, dl.global_region
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_platform_monthly
    ON mv_platform_monthly (year, month, platform_name, job_category, global_region);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_company_hiring AS
SELECT
    dc.company_name,
    dc.industry_sector,
    dc.company_size_category,
    dt.year,
    dt.quarter,
    dl.global_region,
    COUNT(DISTINCT f.job_id) AS total_postings,
    AVG(f.salary_max) AS avg_salary_max
FROM fact_job_posting f
JOIN dim_company dc ON f.company_id = dc.company_id
JOIN dim_time dt ON f.time_id = dt.time_id
JOIN dim_location dl ON f.location_id = dl.location_id
GROUP BY dc.company_name, dc.industry_sector, dc.company_size_category,
         dt.year, dt.quarter, dl.global_region
WITH DATA;

CREATE INDEX IF NOT EXISTS idx_mv_company_sector
    ON mv_company_hiring (industry_sector, year, quarter);

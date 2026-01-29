import os
from pathlib import Path
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv

import func_bit as mvf
import config_bit as cfg

from tqdm import tqdm
tqdm.pandas()

# ========================== 0. Базовые настройки ==========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

cfg.REPORT_FOLDER.mkdir(exist_ok=True)
cfg.RESULT_FOLDER.mkdir(exist_ok=True)

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 50)

SCRAPING_DATE = cfg.SCRAPING_DATE
six_months_ago = SCRAPING_DATE - timedelta(days=180)

# ========================== 1. Загрузка LinkedIn ==========================
def load_linkedin():
    file_path = Path("data") / "dataset_linkedin-jobs-scraper-no-login-required_2025-12-14.xlsx"
    df = pd.read_excel(file_path)

    df = df.rename(columns={
        "company_name": "company",
        "description_text": "description",
        "employment_type": "contractType",
        "seniority_level": "experienceLevel"
    })

    df["posted_at"] = df["published_at"].apply(mvf.parse_date)
    df = df[df["posted_at"].notna()]

    df = df.drop_duplicates() \
           .drop_duplicates(subset=["job_title", "company", "location"], keep="first")

    df["system"] = "LinkedIn"
    print(f"[LinkedIn] после фильтрации по дате и дубликатам: {len(df)}")
    return df

# ========================== 2. Загрузка Stepstone ==========================
def load_stepstone():
    file_path = Path("data") / "dataset_stepstone-scraper-fast-reliable-4-1k_2025-12-15_19-18-23-513.xlsx"
    df = pd.read_excel(file_path)

    df = df.rename(columns={
        "title": "job_title",
        "url": "job_url",
        "companyProfileUrl": "company_url",
        "salary": "salary_range"
    })

    df["posted_at"] = pd.to_datetime(df["datePostedISO"], errors="coerce")
    df["posted_at"] = df["posted_at"].dt.tz_localize(None)

    df = df[df["posted_at"].notna() & (df["posted_at"] >= six_months_ago)]

    df = df.drop_duplicates() \
           .drop_duplicates(subset=["job_title", "company", "location"], keep="first")

    # experienceLevel
    df["experienceLevel"] = df.apply(
        lambda row: mvf.extract_experience(
            text=f"{row.get('job_title', '')} {row.get('description', '')}",
            experienceLevel=row.get("employmentType", "")
        ),
        axis=1
    )

    # soft-классификация
    df_results = df.progress_apply(mvf.classify_soft, axis=1)
    df = pd.concat([df, df_results], axis=1)

    df["system"] = "Stepstone"
    print(f"[Stepstone] после фильтрации по дате и дубликатам: {len(df)}")
    return df

# ========================== 3. Подготовка обоих датафреймов ==========================
df_linkedin = load_linkedin()
df_stepstone = load_stepstone()

# Приводим к общему виду (минимум нужных колонок)
COMMON_COLS = [
    "system",
    "search_term",
    "job_title",
    "job_url",
    "experienceLevel",
    "company",
    "company_url",
    "location",
    "posted_at",
    "salary_range",
    "description"
]

# # LinkedIn может не иметь search_term — если нужно, добавь заранее в пайплайне
# if "search_term" not in df_linkedin.columns:
#     df_linkedin["search_term"] = mvf.match_search_terms(df_linkedin["job_title"], cfg.JOB_TITLE)

df_linkedin_clean = df_linkedin.copy()
df_stepstone_clean = df_stepstone.copy()

# фильтрация по дате для LinkedIn
df_linkedin_clean = df_linkedin_clean[
    df_linkedin_clean["posted_at"].notna() & (df_linkedin_clean["posted_at"] >= six_months_ago)
]

# фильтрация по experienceLevel (junior уровни)
EXCLUDE_LEVELS = {"Executive", "Director", "Mid-Senior level", "Not Applicable", "3+ years"}
df_linkedin_clean = df_linkedin_clean[~df_linkedin_clean["experienceLevel"].isin(EXCLUDE_LEVELS)].copy()
df_stepstone_clean = df_stepstone_clean[~df_stepstone_clean["experienceLevel"].isin(EXCLUDE_LEVELS)].copy()

# очистка job_title
df_linkedin_clean["job_title_clean"] = mvf.clean_job_titles(df_linkedin_clean["job_title"].tolist())
df_stepstone_clean["job_title_clean"] = mvf.clean_job_titles(df_stepstone_clean["job_title"].tolist())

# ========================== 4. Анализ скиллов ==========================
entry_levels_linkedin = df_linkedin_clean["experienceLevel"].unique().tolist()
entry_levels_stepstone = df_stepstone_clean["experienceLevel"].unique().tolist()

hard_linkedin, soft_linkedin = mvf.analyze_skills_by_levels(
    df_linkedin_clean,
    entry_levels_linkedin,
    cfg.HARD_SKILLS,
    cfg.SOFT_SKILLS
)

hard_stepstone, soft_stepstone = mvf.analyze_skills_by_levels(
    df_stepstone_clean,
    entry_levels_stepstone,
    cfg.HARD_SKILLS,
    cfg.SOFT_SKILLS
)

# ========================== 5. Полный сравнительный отчёт ==========================
mvf.build_full_comparison_report(
    df_linkedin_clean=df_linkedin_clean,
    df_stepstone_clean=df_stepstone_clean,
    hard_linkedin=hard_linkedin,
    soft_linkedin=soft_linkedin,
    hard_stepstone=hard_stepstone,
    soft_stepstone=soft_stepstone,
    job_titles=cfg.JOB_TITLE,
    scraping_date=SCRAPING_DATE,
    report_folder=cfg.REPORT_FOLDER
)

# ========================== 6. Сохранение объединённых результатов ==========================
df_linkedin_out = df_linkedin_clean[COMMON_COLS].copy()
df_stepstone_out = df_stepstone_clean[COMMON_COLS].copy()

df_linkedin_out["posted_at"] = pd.to_datetime(df_linkedin_out["posted_at"]).dt.date
df_stepstone_out["posted_at"] = pd.to_datetime(df_stepstone_out["posted_at"]).dt.date

out_linkedin = cfg.RESULT_FOLDER / "LinkedIn_clean_vacancies_res.xlsx"
out_stepstone = cfg.RESULT_FOLDER / "Stepstone_clean_vacancies_res.xlsx"

df_linkedin_out.to_excel(out_linkedin, index=False)
df_stepstone_out.to_excel(out_stepstone, index=False)

print(f"[OK] LinkedIn результат сохранён: {out_linkedin}")
print(f"[OK] Stepstone результат сохранён: {out_stepstone}")
print("\nГотово: сравнительный отчёт LinkedIn vs Stepstone собран.")

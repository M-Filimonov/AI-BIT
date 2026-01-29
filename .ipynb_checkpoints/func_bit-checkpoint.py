import os
import pandas as pd
import re
import networkx as nx
import json
import time
import plotly.graph_objects as go

from IPython.display import display
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from typing import Tuple, Dict, Any, List, Optional, Any
from pathlib import Path
from openpyxl import Workbook
from pandas import DataFrame
from collections import Counter

import logging
logger = logging.getLogger(__name__)


# Загрузка переменных и словарей
from config_ec import (
    DATA_FOLDER, REPORT_FOLDER, RESULT_FOLDER, 
    DEDUPLICATION_MODE,KEEP,
    EMPLOYMENT_TYPES, WORK_TYPES,
    HARD_SKILLS, SOFT_SKILLS, LANG_PATTERNS, LANG_LEVELS,LANG_LEVEL_DESCRIPTIONS,
    GRADE_KEYWORDS, DIRECTION_PRIORITY, E_COMMERCE_TERMS,
    CURRENCY_RATES, PERIOD_MULTIPLIERS
)

# # Create folders if they don't exist
REPORT_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)


########################################
# process_vacancies   функция обработки вакансий AI
########################################
def process_vacancies(df: pd.DataFrame, pause_sec: float = 2.0) -> pd.DataFrame:
    direction_list = []
    ecommerce_list = []
    hard_skills = []
    soft_skills = []
    salary_info = []
    
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка вакансий"):
        full_text = " ".join(str(row[col]) for col in text_columns if pd.notnull(row[col]))
    
        direction = classify_direction(full_text)
        ecommerce_type = classify_ecommerce_or_offline(full_text)
        skills = extract_skills_and_salary(full_text)
    
        direction_list.append(direction)
        ecommerce_list.append(ecommerce_type)
        hard_skills.append(skills["hard_skills"])
        soft_skills.append(skills["soft_skills"])
        salary_info.append(skills["salary_info"])
    
        time.sleep(pause_sec)
    
    df["direction_llm"] = direction_list
    df["ecommerce_or_offline"] = ecommerce_list
    df["hard_skills"] = hard_skills
    df["soft_skills"] = soft_skills
    df["salary_info"] = salary_info
    return df

########################################
# classify_direction  Классификация направления (Marketplace, Marketing, Sales) AI
########################################

def classify_direction(text: str) -> str:
    prompt = (
        "Ты аналитик вакансий. По описанию вакансии определи, к какой группе eCommerce она относится: "
        "Marketplace, Marketing или Sales. Ответ должен быть только одно слово — название группы.\n\n"
        f"Описание:\n{text}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Ошибка:", e)
        return "Error"


########################################
# classify_ecommerce_or_offline Определение eCommerce или offline с AI
########################################
def classify_ecommerce_or_offline(text: str) -> str:
    prompt = (
        "Ты аналитик вакансий. Определи, относится ли вакансия к сфере eCommerce или к оффлайн-торговле. "
        "Ответ должен быть только одно слово: eCommerce или offline.\n\n"
        f"Описание:\n{text}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Ошибка:", e)
        return "Error"


########################################
# extract_skills_and_salary Извлечение хард/софт скиллов и зарплаты с AI
########################################

def extract_skills_and_salary(text: str, current_salary: str, client) -> Dict[str, str]:
    """
    Анализирует описание вакансии с помощью OpenAI и извлекает:
    - Hard skills (ориентируясь на список HARD_SKILLS)
    - Soft skills (ориентируясь на список SOFT_SKILLS)
    - Salary (если поле salary пустое, ищет в description)

    Parameters
    ----------
    text : str
        Описание вакансии.
    current_salary : str
        Текущее значение salary из DataFrame.
    client : OpenAI
        Инициализированный клиент OpenAI.

    Returns
    -------
    dict
        {
            "hard_skills": "...",
            "soft_skills": "...",
            "salary": "..."
        }
    """

    # prompt = (
    #     "Проанализируй описание вакансии.\n"
    #     f"Hard skills (список): {', '.join(HARD_SKILLS)}\n"
    #     f"Soft skills (список): {', '.join(SOFT_SKILLS)}\n"
    #     f"Текущее значение salary: {current_salary}\n"
    #     f"Описание:\n{text}\n"
    #     "Если нет упоминаний из списка — верни 'not specified'.\n"
    #     "Если зарплата не указана — верни 'not specified'.\n"
    #     "Если указана конкурентная оплата (competitive, Top-Bezahlung) — верни 'competitive'.\n"
    #     "Формат зарплаты: €35K/yr - €40K/yr."
    # )

    prompt = (
    "Ты аналитик вакансий. Проанализируй описание и верни строго структурированный результат.\n"
    f"Hard skills: выбирай только из списка [{', '.join(HARD_SKILLS)}]. "
    "Не добавляй soft skills и не придумывай новые. "
    "Записывай только точное название из списка, без пояснений.\n"
    f"Soft skills: выбирай только из списка [{', '.join(SOFT_SKILLS)}]. "
    "Не добавляй hard skills и не придумывай новые. "
    "Записывай только точное название из списка, без пояснений.\n"
    "Salary: если поле salary пустое, найди в описании; если не пустое, верни текущее. "
    "Формат зарплаты: €35K/yr - €40K/yr. "
    "Если указана конкурентная оплата (competitive, Top-Bezahlung) — верни 'competitive'. "
    "Если зарплата не указана — верни 'not specified'.\n\n"
    f"Текущее значение salary: {current_salary}\n"
    f"Описание:\n{text}\n"
    "Формат ответа:\n"
    "Hard skills: <список через запятую или 'not specified'>\n"
    "Soft skills: <список через запятую или 'not specified'>\n"
    "Salary: <значение>"
)


    result = {"hard_skills": "not specified", "soft_skills": "not specified", "salary": current_salary}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            output = response.choices[0].message.content.strip()
            lines = output.splitlines()

            for line in lines:
                if line.lower().startswith("hard skills:"):
                    result["hard_skills"] = line.partition(":")[2].strip()
                elif line.lower().startswith("soft skills:"):
                    result["soft_skills"] = line.partition(":")[2].strip()
                elif line.lower().startswith("salary:"):
                    extracted_salary = line.partition(":")[2].strip()
                    if not current_salary.strip():
                        result["salary"] = extracted_salary

            return result

        except Exception as e:
            error_msg = str(e).lower()

            if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                wait_time = 2 ** attempt
                print(f"⚠️ Rate limit достигнут. Повтор через {wait_time} сек...")
                time.sleep(wait_time)

            elif "request timed out" in error_msg:
                wait_time = 2 ** attempt
                print(f"⏳ Запрос превысил время ожидания. Повтор через {wait_time} сек...")
                time.sleep(wait_time)

            else:
                print("❌ Ошибка:", e)
                return {"hard_skills": "Error", "soft_skills": "Error", "salary": current_salary}

    # Если все попытки неудачны
    return {"hard_skills": "Error", "soft_skills": "Error", "salary": current_salary}

########################################
# extract_skienrich_dataframe_with_skills_and_salary
########################################

def enrich_dataframe_with_skills_and_salary(df: pd.DataFrame, client) -> pd.DataFrame:
    """
    Обрабатывает DataFrame построчно с прогресс‑баром.
    Добавляет hard_skills, soft_skills и обновляет salary.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с колонками description и salary.
    client : OpenAI
        Инициализированный клиент OpenAI.

    Returns
    -------
    pd.DataFrame
        DataFrame с новыми колонками hard_skills, soft_skills и обновлённым salary.
    """
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Извлечение skills и salary"):
        res = extract_skills_and_salary(row["description"], row["salary"], client)
        results.append(res)

    parsed = pd.DataFrame(results)
    df = df.copy()
    df["hard_skills"] = parsed["hard_skills"]
    df["soft_skills"] = parsed["soft_skills"]
    df["salary"] = parsed["salary"]

    return df


########################################
# cluster_job_titles Кластеризация названий вакансий
########################################
def cluster_job_titles(df: pd.DataFrame) -> pd.DataFrame:
    def clean_title(title):
        title = str(title).lower()
        title = re.sub(r"[^a-zA-Z0-9\s]", " ", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title

    def remove_stopwords(text):
        return " ".join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])

    df["job_title_clean"] = df["job_title"].apply(clean_title).apply(remove_stopwords)
    title_counts = Counter(df["job_title_clean"])
    total = sum(title_counts.values())

    def cluster_titles(titles, threshold=0.8):
        clusters = {}
        for title in titles:
            found = False
            for key in clusters:
                if get_close_matches(title, [key], n=1, cutoff=threshold):
                    clusters[key].append(title)
                    found = True
                    break
            if not found:
                clusters[title] = [title]
        return clusters

    clusters = cluster_titles(title_counts.keys())
    cluster_summary = []
    for cluster_name, variants in clusters.items():
        count = sum(title_counts[v] for v in variants)
        percent = round(count / total * 100, 2)
        cluster_summary.append({
            "cluster_name": cluster_name,
            "variants": ", ".join(variants),
            "count": count,
            "percent": percent
        })

    return pd.DataFrame(cluster_summary).sort_values(by="count", ascending=False)


########################################
# list_file_names
########################################
def list_file_names(folder: str = DATA_FOLDER) -> list[str]:
    """
    Возвращает отсортированный список XLSX-файлов в указанной папке.
    Если папка не существует — возвращает пустой список.
    """
    if not os.path.exists(folder):
        print(f"Папка не найдена: {folder}")
        return []

    files = [fn for fn in os.listdir(folder) if fn.lower().endswith(".xlsx")]
    return sorted(files)

    
########################################
# load_and_merge_excel_files
########################################
def load_and_merge_excel_files(file_list: list[str], folder: str, output_path: str) -> pd.DataFrame:
    """
    Загружает все Excel-файлы из списка, приводит их к единому набору колонок,
    объединяет в один DataFrame и сохраняет в указанный файл.
    """
    dataframes = []
    all_columns = set()

    # Сбор всех уникальных колонок
    for fname in tqdm(file_list, desc="Сбор всех уникальных колонок..."):
        path = os.path.join(folder, fname)
        try:
            df = pd.read_excel(path)
            all_columns.update(df.columns.str.strip())
        except Exception as e:
            print(f"Ошибка при загрузке {fname}: {e}")

    all_columns = sorted(all_columns)

    # Загрузка и приведение к общему формату
    for fname in tqdm(file_list, desc="Объединение всех Excel файлов данных в один..."):
        path = os.path.join(folder, fname)
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()
            df = df.reindex(columns=all_columns)
            df["source_file"] = fname
            dataframes.append(df)
        except Exception as e:
            print(f"Ошибка при обработке {fname}: {e}")

    # Объединение всех DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Всего объединено строк: {len(combined_df)}")

    # Сохранение в Excel
    combined_df.to_excel(output_path, index=False)
    print(f"Полное объединение сохранено в файл: {output_path}")

    return combined_df

########################################
#  clean_and_enrich_dataframe
########################################
import re
import pandas as pd
from typing import Any, Dict, Optional
from tqdm import tqdm
#from func_ec import normalize_text, clean_title

def clean_and_enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка, обогащение и фильтрация e-commerce вакансий.
    Дополнительно:
    - расширенный парсинг зарплаты, нормализация в годовой EUR
    - извлечение experienceLevel, contractType, work_type из описания
    - добавление поля required_experience (например, "3 years")
    """

    def fill_primary_from_secondary(df: pd.DataFrame, primary: str, secondary: str) -> None:
        """Заполняет значения в колонке `primary`, если они отсутствуют, используя данные из `secondary`."""
        if primary in df.columns and secondary in df.columns:
            mask = df[primary].isna() & df[secondary].notna()
            df.loc[mask, primary] = df.loc[mask, secondary]

    def extract_info_from_insights(value: Any) -> Dict[str, Optional[str]]:
        """Извлекает contractType, work_type и salary_text из поля job_insights."""
        if pd.isna(value) or not str(value).strip():
            return {"contractType": None, "work_type": None, "salary_text": None}
        txt = normalize_text(value)
        emp = next((et.title() for et in EMPLOYMENT_TYPES if et in txt), None)
        work = next(("On-site" if wt == "on-site" else wt.title() for wt in WORK_TYPES if wt in txt), None)
        salary_text = str(value) if any(k in txt for k in ["€", "eur", "$", "usd", "gbp", "/yr", "/year", "/mo", "/month", "/hr", "/hour", "k"]) else None
        return {"contractType": emp, "work_type": work, "salary_text": salary_text}

    def fill_from_insights(row: pd.Series) -> pd.Series:
        """Заполняет contractType, work_type и salary из распарсенных job_insights."""
        if not row.get("contractType"):
            for col in ["job_insights/0_contractType", "job_insights/1_contractType", "job_insights/2_contractType"]:
                val = row.get(col)
                if pd.notna(val) and val:
                    row["contractType"] = val
                    break
        if not row.get("work_type"):
            for col in ["job_insights/0_work_type", "job_insights/1_work_type", "job_insights/2_work_type"]:
                val = row.get(col)
                if pd.notna(val) and val:
                    row["work_type"] = val
                    break
        if (not row.get("salary")) or pd.isna(row.get("salary")):
            for col in ["job_insights/0_salary_text", "job_insights/1_salary_text", "job_insights/2_salary_text"]:
                val = row.get(col)
                if pd.notna(val) and val:
                    row["salary"] = val
                    break
        return row

    def detect_direction(title_clean: str, description_norm: str) -> Optional[str]:
        """Определяет направление вакансии на основе словаря E_COMMERCE_TERMS."""
        txt = f"{normalize_text(title_clean)} {normalize_text(description_norm)}"
        found = set()
        for dir_name, terms in E_COMMERCE_TERMS.items():
            for term in terms:
                if term.lower() in txt:
                    found.add(dir_name)
                    break
        for d in DIRECTION_PRIORITY:
            if d in found:
                return d
        return None

    def enrich_from_description(row: pd.Series) -> pd.Series:
        """
        Дополняет experienceLevel, contractType, work_type и required_experience из описания вакансии.
        """
        desc = normalize_text(row.get("description", ""))

        # Уровень опыта
        if not row.get("experienceLevel"):
            if "intern" in desc or "praktikum" in desc or "trainee" in desc:
                row["experienceLevel"] = "Intern"
            elif "junior" in desc:
                row["experienceLevel"] = "Junior"
            elif "senior" in desc:
                row["experienceLevel"] = "Senior"
            elif any(k in desc for k in ["mid-level", "intermediate", "professional"]):
                row["experienceLevel"] = "Mid"
            elif any(k in desc for k in ["lead", "principal", "head", "manager"]):
                row["experienceLevel"] = "Lead"

        # Тип контракта
        if not row.get("contractType"):
            if "vollzeit" in desc or "full-time" in desc:
                row["contractType"] = "Full-time"
            elif "teilzeit" in desc or "part-time" in desc:
                row["contractType"] = "Part-time"
            elif "befristet" in desc or "temporary" in desc:
                row["contractType"] = "Fixed-term"
            elif "werkstudent" in desc or "working student" in desc:
                row["contractType"] = "Working Student"
            elif "freelance" in desc or "contractor" in desc:
                row["contractType"] = "Freelance"
            elif "praktikum" in desc or "internship" in desc:
                row["contractType"] = "Internship"

        # Формат работы
        if not row.get("work_type"):
            if "remote" in desc or "home office" in desc:
                row["work_type"] = "Remote"
            elif "hybrid" in desc:
                row["work_type"] = "Hybrid"
            elif "on-site" in desc or "vor ort" in desc:
                row["work_type"] = "On-site"

        # Необходимый опыт (например, "2+ years")
        exp_match = re.search(r"(\d+)\+?\s*(year|years|jahr|jahre)\b", desc)
        if exp_match:
            row["required_experience"] = f"{exp_match.group(1)} years"

        return row

    # === Основная логика ===
    fill_primary_from_secondary(df, "company", "companyName")
    fill_primary_from_secondary(df, "company_url", "companyUrl")
    fill_primary_from_secondary(df, "job_title", "title")
    fill_primary_from_secondary(df, "posted_at", "publishedAt")
    fill_primary_from_secondary(df, "job_url", "applyUrl")
    fill_primary_from_secondary(df, "job_url", "jobUrl")

    # Инициализация итоговых полей
    for col in ["contractType", "work_type", "salary", "experienceLevel", "required_experience"]:
        if col not in df.columns:
            df[col] = None

    # Парсинг job_insights
    for col in ["job_insights/0", "job_insights/1", "job_insights/2"]:
        parsed = df[col].apply(extract_info_from_insights).apply(pd.Series)
        df[f"{col}_contractType"] = parsed["contractType"]
        df[f"{col}_work_type"] = parsed["work_type"]
        df[f"{col}_salary_text"] = parsed["salary_text"]

    tqdm.pandas(desc="Извлечение подсказок из job_insights...")
    df = df.progress_apply(fill_from_insights, axis=1)

    # Нормализация заголовка и описания
    df["title_clean"] = df["job_title"].apply(clean_title)
    df["description_norm"] = df["description"].apply(normalize_text)

    # Определение направления
    df["direction"] = df.apply(
        lambda row: detect_direction(row.get("title_clean"), row.get("description_norm")),
        axis=1
    )
    df = df[df["direction"].notna()].copy()

    # Обогащение из описания
    tqdm.pandas(desc="Дозаполнение level/contract/work_type/experience из description...")
    df = df.progress_apply(enrich_from_description, axis=1)

    # Нормализация локаций
    df = clean_and_enrich_germany_locations(df)

    return df





########################################
#  deduplicate_dataframe & report
########################################

# безопасный импорт rapidfuzz
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None
    print("rapidfuzz не установлен — fuzzy-этап будет пропущен. Установи rapidfuzz для улучшения результатов.")

def deduplicate_dataframe(df: pd.DataFrame,
                          result_folder: Path,
                          mode: str = "canonical",
                          keep: str = "first",
                          title_threshold: int = 90,
                          max_pairs: int = 2000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Удаляет дубликаты из датафрейма.
    mode = "simple" → drop_duplicates по безопасным колонкам
    mode = "canonical" → выбор лучшей строки из кластеров по приоритету
    keep = "first" или "last" → какую строку оставлять при удалении дубликатов в режиме simple
    """

    df = df.copy()
    for col in ["job_title", "company", "location", "job_url", "job_id", "description", "posted_at", "salary"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    report: Dict[str, Any] = {"total_rows": len(df)}

    # безопасные колонки (исключаем dict/list)
    safe_cols = [c for c in df.columns if df[c].map(lambda x: not isinstance(x, (dict, list))).all()]
    full_dup_mask = df.duplicated(subset=safe_cols, keep=False)
    report["full_exact_duplicates_count"] = int(full_dup_mask.sum())

    # дубликаты по job_url и job_id
    report["dup_by_job_url_count"] = int(df["job_url"].duplicated(keep=False).sum()) if "job_url" in df.columns else 0
    report["dup_by_job_id_count"] = int(df["job_id"].duplicated(keep=False).sum()) if "job_id" in df.columns else 0

    # нормализация
    def normalize(s: str) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s\-]", "", s)
        return s

    df["company_norm"] = df["company"].apply(normalize)
    df["location_norm"] = df["location"].apply(normalize)
    df["title_norm"] = df["job_title"].apply(normalize)
    df["block_key"] = df["company_norm"].str[:30] + "|" + df["location_norm"].str[:30]

    # fuzzy-сравнение
    matches = []
    for block, group in df.groupby("block_key"):
        idxs = group.index.tolist()
        if len(idxs) < 2:
            continue
        pairs = list(combinations(idxs, 2))[:max_pairs]
        for i, j in pairs:
            s1, s2 = df.at[i, "title_norm"], df.at[j, "title_norm"]
            if s1 and s2:
                score = fuzz.token_set_ratio(s1, s2)
                if score >= title_threshold:
                    matches.append((i, j, score))

    report["fuzzy_candidate_pairs"] = len(matches)

    # граф
    G = nx.Graph()
    G.add_nodes_from(df.index)

    for val, group in df.groupby("job_url"):
        if val:
            idxs = group.index.tolist()
            for i, j in combinations(idxs, 2):
                G.add_edge(i, j, reason="job_url")

    for val, group in df.groupby("job_id"):
        if val and val != "0":
            idxs = group.index.tolist()
            for i, j in combinations(idxs, 2):
                G.add_edge(i, j, reason="job_id")

    for i, j, score in matches:
        G.add_edge(i, j, reason=f"fuzzy_{score}")

    # кластеры
    components = list(nx.connected_components(G))
    dup_clusters = [c for c in components if len(c) > 1]
    report["duplicate_clusters_count"] = len(dup_clusters)
    report["rows_in_duplicate_clusters"] = sum(len(c) for c in dup_clusters)

    # примеры
    examples = []
    for c in dup_clusters[:10]:
        sample = []
        for idx in sorted(c):
            sample.append({
                "index": int(idx),
                "job_title": df.at[idx, "job_title"],
                "company": df.at[idx, "company"],
                "location": df.at[idx, "location"],
                "job_url": df.at[idx, "job_url"],
                "job_id": df.at[idx, "job_id"]
            })
        examples.append(sample)
    report["examples"] = examples

    # метки
    group_id = {}
    for gid, comp in enumerate(dup_clusters, start=1):
        for idx in comp:
            group_id[idx] = gid

    df["is_duplicate"] = df.index.map(lambda i: i in group_id)
    df["duplicate_group_id"] = df.index.map(lambda i: group_id.get(i, None))

    # применение режима
    if mode == "simple":
        df_clean = df.drop_duplicates(subset=safe_cols, keep=keep)
        if "duplicate_group_id" in df_clean.columns:
            df_clean = df_clean.sort_values("duplicate_group_id").drop_duplicates(
                subset=["duplicate_group_id"], keep=keep
            )
        report["keep_strategy"] = keep

    elif mode == "canonical":
        selected_rows = []
        for comp in dup_clusters:
            group = df.loc[list(comp)].copy()
            group["has_url"] = group["job_url"].apply(lambda x: bool(str(x).strip()))
            group["has_salary"] = group["salary"].apply(lambda x: bool(str(x).strip()))
            group["desc_len"] = group["description"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
            group["posted_dt"] = pd.to_datetime(group["posted_at"], errors="coerce")
            best = group.sort_values(
                by=["has_url", "has_salary", "desc_len", "posted_dt"],
                ascending=[False, False, False, False]
            ).iloc[0]
            selected_rows.append(best)
        unique_rows = df[~df.index.isin(group_id.keys())]
        df_clean = pd.concat([unique_rows, pd.DataFrame(selected_rows)], ignore_index=True)
        report["keep_strategy"] = "priority"

    else:
        raise ValueError(f"Unknown deduplication mode: {mode}")
    
    # Финальный список колонок
    list_of_columns_to_leave_final = [
        "direction", "job_title", "job_url", "company",
        "experienceLevel", "contractType", "work_type",
        "sector", "salary", "description", "skills",
        "location", "city", "state", "country", "posted_at"
    ]
    df_clean = df_clean[[c for c in list_of_columns_to_leave_final if c in df_clean.columns]].copy()
   
    # Сохраняем отчёт в JSON
    save_report_json(report, result_folder)
    
    return df_clean, report


########################################
#  save_report_json
########################################
def save_report_json(report: Dict[str, Any], result_folder: Path, filename: str = "dedup_report.json") -> Path:
    """Сохраняет отчёт в JSON с пояснениями."""
    report_explained = {
        "total_rows": "Общее количество строк в датафрейме до удаления дубликатов",
        "full_exact_duplicates_count": "Количество полных дубликатов (строки совпадают по всем полям)",
        "duplicate_clusters_count": "Количество кластеров дубликатов (группы похожих вакансий)",
        "rows_in_duplicate_clusters": "Количество строк, попавших в кластеры дубликатов",
        "examples": "Примеры кластеров дубликатов с индексами и основными полями"
    }

    report_with_notes = {
        "explanations": report_explained,
        "report": report,        
    }

    result_folder.mkdir(parents=True, exist_ok=True)
    output_path = result_folder / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_with_notes, f, ensure_ascii=False, indent=4)

    print(f"Отчёт по дубликатам сохранён в {output_path}")
    return output_path


########################################
#  clean_and_enrich_germany_locations
########################################
def clean_and_enrich_germany_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка и обогащение DataFrame:
    - оставляет только записи по Германии
    - нормализует специальные регионы и федеральные земли
    - разбивает location на city, state, country
    """

    # 1. Фильтруем только строки, где явно указана Germany
    df_germany = df[df["location"].str.contains("Germany", na=False)].copy()

    # 2. Словарь нормализации регионов
    normalize_map = {
        "Ruhr Region": "Ruhr, North Rhine-Westphalia, Germany",
        "Stuttgart Region": "Stuttgart, Baden-Württemberg, Germany",
        "Cologne Bonn Region": "Cologne Bonn, North Rhine-Westphalia, Germany",
        "Greater Munich Metropolitan Area": "Munich, Bavaria, Germany",
        "Berlin Metropolitan Area": "Berlin, Berlin, Germany",
        "Greater Hamburg Area": "Hamburg, Hamburg, Germany",
        "Frankfurt Rhine-Main Metropolitan Area": "Frankfurt, Hesse, Germany",
        "Hannover-Braunschweig-Göttingen-Wolfsburg Region": "Hannover Region, Lower Saxony, Germany",
        "Rhein-Neckar Metropolitan Region": "Rhein-Neckar, Baden-Württemberg/Hesse, Germany",
        "Greater Dusseldorf Area": "Dusseldorf, North Rhine-Westphalia, Germany",
    }
    df_germany["location"] = df_germany["location"].replace(normalize_map)

    # 3. Словарь нормализации федеральных земель
    state_normalization = {
        "Bayern": "Bavaria",
        "Sachsen": "Saxony",
        "Thüringen": "Thuringia",
        "Hessen": "Hesse",
        "Mecklenburg-Vorpommern": "Mecklenburg-West Pomerania",
        "Niedersachsen": "Lower Saxony",
        "Nordrhein-Westfalen": "North Rhine-Westphalia",
        "Rheinland-Pfalz": "Rhineland-Palatinate",
        "Saarland": "Saarland",
        "Schleswig-Holstein": "Schleswig-Holstein",
        "Baden-Württemberg": "Baden-Württemberg",
        "Berlin": "Berlin",
        "Hamburg": "Hamburg",
        "Bremen": "Bremen",
        "Brandenburg": "Brandenburg",
    }

    # 4. Функция для аккуратного разбиения
    def split_location(loc: str):
        parts = [p.strip() for p in loc.split(",")]
        if len(parts) == 1:
            return None, None, parts[0]              # только страна
        elif len(parts) == 2:
            # если первая часть — федеральная земля, то это state
            state = state_normalization.get(parts[0], parts[0])
            return None, state, parts[1] if parts[1] else "Germany"
        elif len(parts) >= 3:
            city = parts[0]
            state = state_normalization.get(parts[1], parts[1])
            country = parts[2]
            return city, state, country
        else:
            return None, None, None

    # 5. Применяем
    df_germany[["city", "state", "country"]] = df_germany["location"].apply(lambda x: pd.Series(split_location(x)))

    return df_germany



########################################
#  analyze_jobs_by_state
########################################
def analyze_jobs_by_state(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Анализ количества вакансий по федеральным землям Германии.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame после очистки и обогащения (с колонкой 'state').
    top_n : int, optional
        Количество топовых регионов для вывода (по умолчанию 10).

    Returns
    -------
    pd.DataFrame
        Таблица с количеством вакансий по федеральным землям,
        отсортированная по убыванию.
    """
    if "state" not in df.columns:
        raise ValueError("В DataFrame отсутствует колонка 'state'. Сначала вызовите clean_and_enrich_germany_locations.")

    # считаем количество вакансий по state
    state_counts = df["state"].value_counts().reset_index()
    state_counts.columns = ["state", "vacancy_count"]

    # возвращаем топ-N
    return state_counts.head(top_n)



########################################
#  normalize_salary_field
########################################

def normalize_salary_field(df: pd.DataFrame, salary_col: str = "salary") -> pd.DataFrame:
    """
    Нормализация поля salary:
    - извлекает min/max зарплату
    - определяет валюту (EUR, USD, GBP)
    - приводит все значения к годовому формату
    - конвертирует в EUR

    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с колонкой salary.
    salary_col : str, optional
        Название колонки с зарплатой (по умолчанию "salary").

    Returns
    -------
    pd.DataFrame
        DataFrame с новыми колонками:
        - salary_min_eur
        - salary_max_eur
        - salary_currency
        - salary_period
    """

    def parse_salary(s: str) -> Dict[str, Optional[float]]:
        if not s or not str(s).strip():
            return {"salary_min_eur": None, "salary_max_eur": None,
                    "salary_currency": None, "salary_period": None}

        txt = s.lower().replace(",", "").replace(" ", "")

        # Определяем валюту
        if "€" in txt or "eur" in txt:
            currency = "EUR"
        elif "$" in txt or "usd" in txt:
            currency = "USD"
        elif "£" in txt or "gbp" in txt:
            currency = "GBP"
        else:
            currency = "EUR"

        # Определяем период
        if "/hr" in txt or "hour" in txt:
            period = "hour"
        elif "/month" in txt or "mo" in txt:
            period = "month"
        else:
            period = "year"

        # Извлекаем числа
        numbers = re.findall(r"[\d\.]+k?", txt)
        values = []
        for num in numbers:
            if "k" in num:
                values.append(float(num.replace("k", "")) * 1000)
            else:
                values.append(float(num))

        if not values:
            return {"salary_min_eur": None, "salary_max_eur": None,
                    "salary_currency": currency, "salary_period": period}

        salary_min = min(values)
        salary_max = max(values)

        # перевод в год
        multiplier = PERIOD_MULTIPLIERS[period]
        salary_min_year = salary_min * multiplier
        salary_max_year = salary_max * multiplier

        # конвертация в EUR
        rate = CURRENCY_RATES[currency]
        salary_min_eur = round(salary_min_year * rate, 2)
        salary_max_eur = round(salary_max_year * rate, 2)

        return {
            "salary_min_eur": salary_min_eur,
            "salary_max_eur": salary_max_eur,
            "salary_currency": currency,
            "salary_period": period
        }

    parsed = df[salary_col].apply(parse_salary).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    return df





####################################################################################
# Утилиты очистки / извлечения
####################################################################################

def clean_title(title: str) -> str:
    '''
    Функция clean_title предназначена для очистки и нормализации заголовков вакансий,
    особенно в немецком контексте. Она удаляет гендерные обозначения (m/w/d),
    скобки с сокращениями, лишние символы и приводит текст к унифицированному виду.
    Это важно для последующей классификации, кластеризации и сравнения названий должностей.
    '''
    # Проверка: если значение отсутствует (NaN), вернуть пустую строку
    if pd.isna(title):
        return ""

    # Преобразование в строку
    s = str(title)

    # Удаление скобок, содержащих m/w/d (например, "(m/w/d)", "(m / w / d)", "(m/w/d - remote)")
    s = re.sub(r"\([^\)]*m\s*\/\s*w\s*\/\s*d[^\)]*\)", " ", s, flags=re.I)

    # Удаление одиночных вхождений m/w/d без скобок
    s = re.sub(r"\b[mM]\s*\/\s*[wW]\s*\/\s*[dD]\b", " ", s)

    # Удаление всех остальных скобок и их содержимого
    s = re.sub(r"\(.*?\)", " ", s)

    # Удаление всех символов, кроме:
    # - латинских и расширенных латинских букв (включая немецкие: ä, ö, ü, ß)
    # - цифр
    # - пробелов, дефисов, амперсандов, точек и слэшей
    s = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s\-/&\.]", " ", s)

    # Замена множественных пробелов на один и удаление пробелов по краям
    s = re.sub(r"\s+", " ", s).strip()

    # Приведение к нижнему регистру
    return s.lower()


def normalize_text(s: str) -> str:
    '''
    Функция normalize_text предназначена для нормализации текстовых данных —
    она очищает строку от лишних пробелов, приводит её к нижнему регистру и
    обрабатывает пропущенные значения (NaN).     
    '''
    # Проверка: если значение отсутствует (NaN), вернуть пустую строку
    if pd.isna(s):
        return ""

    # Преобразование в строку и приведение к нижнему регистру
    t = str(s).lower()

    # Замена множественных пробелов (включая табуляции и переводы строк) на один пробел
    # Удаление пробелов в начале и конце строки
    t = re.sub(r"\s+", " ", t).strip()

    # Возврат нормализованного текста
    return t




def extract_grade(title_clean: str) -> Optional[str]:
    '''
    Функция extract_grade предназначена для определения уровня должности (grade)
    на основе ключевых слов в заголовке вакансии. Она ищет совпадения с заранее
    заданными паттернами, чтобы классифицировать вакансию как, например,
    "junior", "senior", "lead", "intern" и т.д.
    '''
    # Приводим заголовок к нижнему регистру для унификации
    t = title_clean.lower()

    # Проходим по словарю ключевых слов для каждого уровня
    for grade, keywords in GRADE_KEYWORDS.items():
        for k in keywords:
            # Ищем регулярное выражение в заголовке
            if re.search(k, t):
                return grade  # Возвращаем первый найденный уровень
    return None  # Если ничего не найдено, возвращаем None


########################################
# 2. Парсер зарплат (упрощённый, расширяемый)
########################################

number_re = re.compile(r"[\d\.,]+")

def parse_salary_field(s: str) -> Dict[str, Optional[float]]:
    '''
    Функция parse_salary_field предназначена для извлечения информации о
    зарплате из текстового поля вакансии. Она определяет:
    Минимальную и максимальную зарплату
    Валюту (EUR или USD)
    Период (в год, месяц или час)
    '''
    # Инициализация выходного словаря с None
    out = {"salary_min": None, "salary_max": None, "currency": None, "period": None}

    # Проверка: если значение отсутствует или пустое — вернуть пустой результат
    if pd.isna(s) or not str(s).strip():
        return out

    # Приведение к строке и нижнему регистру
    txt = str(s).lower()

    # Определение валюты
    if "€" in txt or "eur" in txt:
        out["currency"] = "EUR"
    elif "$" in txt or "usd" in txt:
        out["currency"] = "USD"

    # Определение периода (английские и немецкие варианты)
    if any(k in txt for k in ["per year", "per annum", "year", "jahres", "jahresgehalt", "pa"]):
        out["period"] = "year"
    elif any(k in txt for k in ["per month", "monat", "/month"]):
        out["period"] = "month"
    elif any(k in txt for k in ["per hour", "stunde", "/hour", "hour"]):
        out["period"] = "hour"

    # Извлечение чисел с помощью регулярного выражения (предполагается, что number_re определён заранее)
    nums = number_re.findall(txt.replace(" ", ""))

    # Преобразование чисел: удаление точек (как разделителей тысяч), замена запятых на точки (десятичные)
    nums = [n.replace(".", "").replace(",", ".") for n in nums]

    # Преобразование в float с обработкой ошибок
    nums_f = []
    for n in nums:
        try:
            val = float(n)
            nums_f.append(val)
        except:
            continue

    # Заполнение salary_min и salary_max
    if len(nums_f) == 1:
        out["salary_min"] = nums_f[0]
        out["salary_max"] = nums_f[0]
    elif len(nums_f) >= 2:
        out["salary_min"] = min(nums_f[0], nums_f[1])
        out["salary_max"] = max(nums_f[0], nums_f[1])

    return out


def to_annual_eur(salary_min, period, currency):
    '''
    Функция to_annual_eur преобразует зарплату, указанную в часовом
    или месячном формате, в годовой эквивалент в евро.
    Она предполагает, что валюта уже приведена к EUR (или игнорируется),
    и использует эвристику для пересчёта почасовой ставки.
    '''
    # Проверка: если зарплата отсутствует или равна None — вернуть None
    if pd.isna(salary_min) or salary_min is None:
        return None

    val = salary_min  # Базовое значение

    # Преобразование в годовой эквивалент в зависимости от периода
    if period == "month":
        annual = val * 12  # 12 месяцев в году
    elif period == "hour":
        annual = val * 160 * 12  # Эвристика: 160 рабочих часов в месяц × 12 месяцев
    else:
        annual = val  # Если уже годовая ставка

    # Преобразование валюты не реализовано — предполагается, что значение уже в EUR
    return annual


########################################
# 3. Experience extraction
########################################

# Регулярные выражения для поиска количества лет опыта
exp_re_en = re.compile(r"(\d+)\s*\+?\s*(years|year|yrs|y)\b", flags=re.I)
exp_re_de = re.compile(r"(\d+)\s*\+?\s*(jahre|jahr|j)\b", flags=re.I)

def extract_experience_years(text: str) -> Optional[int]:
    '''
    Функция extract_experience_years предназначена для извлечения 
    количества лет опыта из текстового описания вакансии или профиля.
    Она поддерживает английские и немецкие формулировки, 
    а также использует эвристику по ключевым словам ("senior", "junior", "trainee").
    '''
    
    # Проверка: если текст пустой — вернуть None
    if not text:
        return None

    # Поиск английского шаблона (например, "3 years", "5+ yrs")
    m = exp_re_en.search(text)
    if m:
        return int(m.group(1))

    # Поиск немецкого шаблона (например, "2 Jahre", "4+ Jahr")
    m2 = exp_re_de.search(text)
    if m2:
        return int(m2.group(1))

    # Эвристика: если встречается "senior", предполагаем 5 лет опыта
    if "senior" in text:
        return 5

    # Эвристика: если встречается "junior" или "trainee", предполагаем 0 лет
    if "junior" in text or "trainee" in text:
        return 0

    # Если ничего не найдено — вернуть None
    return None


########################################
# 4. Language detection
########################################

def detect_languages(text: str) -> Dict[str, Optional[str]]:
    '''
    Функция detect_languages предназначена для определения языков,
    упомянутых или используемых в тексте, а также для извлечения
    подсказки об уровне владения языком (например, B2, C1).
    '''
    
    # Инициализация выходного словаря
    out = {"german": False, "english": False, "lang_level_hint": None}

    # Проверка: если текст пустой — вернуть пустой результат
    if not text:
        return out

    # Приведение текста к нижнему регистру
    t = text.lower()

    # Поиск языков по шаблонам
    for lname, pats in LANG_PATTERNS.items():
        for p in pats:
            if re.search(p, t):
                out[lname] = True  # Устанавливаем флаг языка
                break  # Переход к следующему языку

    # Поиск уровня владения языком
    for lvl, pats in LANG_LEVELS.items():
        for p in pats:
            if re.search(p, t):
                out["lang_level_hint"] = lvl.upper()  # Например, "B2", "C1"
                break
        if out["lang_level_hint"]:
            break  # Прерываем, если уровень уже найден

    return out


########################################
# 5. Skills extraction
########################################

def extract_skills(text: str, skills_list: List[str]) -> List[str]:
    '''
    Функция extract_skills предназначена для выделения навыков из текстового описания
    (например, вакансии или резюме) на основе заданного списка. 
    Она ищет точные вхождения каждого навыка в тексте и возвращает 
    отсортированный список уникальных совпадений.
    '''
    # Проверка: если текст пустой — вернуть пустой список
    if not text:
        return []

    # Приведение текста к нижнему регистру для унификации
    t = text.lower()

    # Список для хранения найденных навыков
    found = []

    # Поиск каждого навыка в тексте
    for s in skills_list:
        if s in t:
            found.append(s)

    # Удаление дубликатов и сортировка
    return sorted(set(found))


########################################
# 6. Titles clusters / top formulations
########################################

def top_titles_by_direction(df_proc: pd.DataFrame, direction: str, topn: int = 30) -> pd.DataFrame:
    '''
    Функция top_titles_by_direction предназначена для вывода самых популярных должностей
    (title_clean) в рамках заданного направления (direction) из обработанного датафрейма 
    df_proc. Она возвращает таблицу с абсолютным и относительным количеством упоминаний 
    каждой должности.
    '''
    
    # Фильтруем строки по заданному направлению
    subset = df_proc[df_proc["direction"] == direction]

    # Считаем количество уникальных заголовков вакансий
    titles = subset["title_clean"].fillna("").value_counts()

    # Общее количество заголовков в выбранном направлении
    total = titles.sum()

    # Выбираем top-N самых частых заголовков
    top = titles.head(topn).reset_index()

    # Переименовываем столбцы
    top.columns = ["title_clean", "count"]

    # Добавляем процентное соотношение от общего количества
    top["percent"] = (top["count"] / total * 100).round(2)

    # Добавляем колонку с направлением
    top["direction"] = direction

    # Возвращаем итоговую таблицу
    return top

########################################
# 
########################################
def log_step(name: str, start_time: float):
    duration = round(time.time() - start_time, 2)
    logger.info(f"{name} completed in {duration} seconds.")
    
def run_ecommerce_pipeline(df: pd.DataFrame, verbose: bool = True, logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    """
    Главная функция пайплайна для анализа e-commerce вакансий.

    Parameters
    ----------
    df : pd.DataFrame
        Входной датафрейм (например, df_pre), который уже прошёл предварительную очистку:
        - direction, city, state, country — нормализованы
        - hard_skills, soft_skills — извлечены
        - salary_min_eur, salary_max_eur, salary_currency, salary_period — рассчитаны
    verbose : bool, optional
        Если True, выводит дополнительную информацию в логах (по умолчанию True).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Словарь с результатами анализа, включающий:
        - "counts": количество вакансий по направлениям и уровням опыта
        - "salary_agg": агрегированные данные по зарплатам (min/max/median/mean)
        - "experience": статистика по опыту (experienceLevel, exp_years)
        - "languages": требования по языкам (если поля german/english присутствуют)
        - "hard_skills": частотность hard skills по направлениям
        - "soft_skills": частотность soft skills по направлениям
        - "companies": профили компаний (кол-во вакансий, направления, медианная зарплата, опыт)
        - "title_clusters": кластеры заголовков вакансий
        - "df_processed": итоговый датафрейм после обогащения
        - "keep_columns": список колонок, которые рекомендуется оставить
        - "drop_suggestions": список колонок, которые можно удалить

    Notes
    -----
    Функция выполняет следующие шаги:
    1. Очистка заголовков вакансий и нормализация описаний.
    2. Обогащение полей experienceLevel, contractType, work_type из описаний вакансий.
    3. Формирование агрегатов по зарплатам, опыту, языкам.
    4. Подсчёт частотности hard/soft skills по направлениям.
    5. Построение профилей компаний.
    6. Кластеризация заголовков вакансий.
    7. Формирование итогового словаря reports.
    """

    start_total = time.time()
    logger.info("Pipeline run_ecommerce_pipeline started.")
    df_proc = df.copy()

    # Этап 1: Очистка заголовков и нормализация описания
    t0 = time.time()
    df_proc["title_clean"] = df_proc["job_title"].apply(clean_title)
    df_proc["description_norm"] = df_proc["description"].apply(normalize_text)
    if verbose:
        log_step("Этап 1: Очистка заголовков и нормализация описания", t0)

    # Этап 2: Обогащение experienceLevel, contractType, work_type
    def enrich_from_description(row: pd.Series) -> pd.Series:
        desc = row.get("description_norm", "")

        if not row.get("experienceLevel") or row["experienceLevel"] in ["", "not specified"]:
            if "intern" in desc or "praktikum" in desc or "trainee" in desc:
                row["experienceLevel"] = "Intern"
            elif "junior" in desc:
                row["experienceLevel"] = "Junior"
            elif "senior" in desc:
                row["experienceLevel"] = "Senior"
            elif any(k in desc for k in ["mid-level", "intermediate", "professional"]):
                row["experienceLevel"] = "Mid"
            elif any(k in desc for k in ["lead", "principal", "head", "manager"]):
                row["experienceLevel"] = "Lead"

        if not row.get("contractType") or row["contractType"] in ["", "not specified"]:
            if "vollzeit" in desc or "full-time" in desc:
                row["contractType"] = "Full-time"
            elif "teilzeit" in desc or "part-time" in desc:
                row["contractType"] = "Part-time"
            elif "befristet" in desc or "temporary" in desc:
                row["contractType"] = "Fixed-term"
            elif "werkstudent" in desc or "working student" in desc:
                row["contractType"] = "Working Student"
            elif "freelance" in desc or "contractor" in desc:
                row["contractType"] = "Freelance"
            elif "praktikum" in desc or "internship" in desc:
                row["contractType"] = "Internship"

        if not row.get("work_type") or row["work_type"] in ["", "not specified"]:
            if "remote" in desc or "home office" in desc:
                row["work_type"] = "Remote"
            elif "hybrid" in desc:
                row["work_type"] = "Hybrid"
            elif "on-site" in desc or "vor ort" in desc:
                row["work_type"] = "On-site"

        return row

    t0 = time.time()
    tqdm.pandas(desc="Обогащение experienceLevel, contractType, work_type")
    df_proc = df_proc.progress_apply(enrich_from_description, axis=1)
    if verbose:
        log_step("Этап 2: Обогащение experienceLevel, contractType, work_type", t0)

    # Этап 3: Агрегации
    t0 = time.time()
    counts = df_proc.groupby(["direction", "experienceLevel"]).size().reset_index(name="count")
    total_by_direction = df_proc["direction"].value_counts().rename_axis("direction").reset_index(name="total")

    salary_agg = df_proc.groupby(["direction", "experienceLevel"]).agg(
        vacancies_count=("salary_min_eur", "count"),
        median_salary_min=("salary_min_eur", "median"),
        median_salary_max=("salary_max_eur", "median"),
        mean_salary_min=("salary_min_eur", "mean"),
        mean_salary_max=("salary_max_eur", "mean")
    ).reset_index()

    exp_stats = df_proc.groupby(["direction", "experienceLevel"]).agg(
        count=("experienceLevel", "count"),
        median_exp=("exp_years", "median") if "exp_years" in df_proc.columns else ("experienceLevel", "count"),
        mean_exp=("exp_years", "mean") if "exp_years" in df_proc.columns else ("experienceLevel", "count")
    ).reset_index()

    lang_stats = pd.DataFrame()
    if "german" in df_proc.columns and "english" in df_proc.columns:
        lang_stats = df_proc.groupby("direction").agg(
            total_vacancies=("direction", "count"),
            german_required=("german", "sum"),
            english_required=("english", "sum")
        ).reset_index()

    def skills_counter(series_of_lists):
        c = Counter()
        for lst in series_of_lists:
            if isinstance(lst, list):
                c.update(lst)
            elif isinstance(lst, str) and lst not in ["", "not specified"]:
                c.update([s.strip() for s in lst.split(",") if s.strip()])
        return pd.DataFrame(c.most_common(), columns=["skill", "count"])

    hard_skills_by_dir = {}
    soft_skills_by_dir = {}
    for d in DIRECTION_PRIORITY:
        subset = df_proc[df_proc["direction"] == d]
        hard_skills_by_dir[d] = skills_counter(subset["hard_skills"])
        soft_skills_by_dir[d] = skills_counter(subset["soft_skills"])

    df_proc["company_norm"] = df_proc["company"].apply(normalize_text)
    company_profiles = df_proc.groupby("company_norm").agg(
        vacancies_count=("company_norm", "count"),
        directions=("direction", lambda s: ", ".join(sorted(set(s.dropna())))),
        median_salary=("salary_min_eur", "median"),
        avg_exp_years=("exp_years", "median") if "exp_years" in df_proc.columns else ("experienceLevel", "count")
    ).sort_values("vacancies_count", ascending=False).reset_index()

    title_clusters_list = []
    for d in DIRECTION_PRIORITY:
        top = top_titles_by_direction(df_proc, d, topn=50)
        title_clusters_list.append(top)
    title_clusters = pd.concat(title_clusters_list, ignore_index=True) if title_clusters_list else pd.DataFrame()

    keep_cols = [
        "job_title", "title_clean", "description", "description_norm", "company", "company_norm",
        "location", "city", "state", "country", "posted_at", "job_url", "job_id",
        "direction", "experienceLevel", "contractType", "work_type",
        "salary", "salary_min_eur", "salary_max_eur", "salary_currency", "salary_period",
        "hard_skills", "soft_skills"
    ]
    drop_suggestions = [c for c in df_proc.columns if c not in keep_cols]

    # Формирование отчётов
    reports = {
        "counts": counts.merge(total_by_direction, on="direction", how="left"),
        "salary_agg": salary_agg,
        "experience": exp_stats,
        "languages": lang_stats,
        "hard_skills": hard_skills_by_dir,
        "soft_skills": soft_skills_by_dir,
        "companies": company_profiles,
        "title_clusters": title_clusters,
        "df_processed": df_proc,
        "keep_columns": keep_cols,
        "drop_suggestions": drop_suggestions
    }

    log_step("Всего пайплайн", start_total)
    return reports

def show_reports(reports: dict, top_n: int = 10) -> None:
    """
    Удобный просмотр содержимого словаря отчётов (reports).

    Parameters
    ----------
    reports : dict
        Словарь с результатами пайплайна. Ключи — названия отчётов,
        значения могут быть DataFrame, Series, dict, list или другие типы.
    top_n : int, optional
        Количество строк/элементов для отображения (по умолчанию 10).

    Returns
    -------
    None
        Функция выводит содержимое отчётов в консоль и через display(),
        ничего не возвращает.
    
    Notes
    -----
    - Для DataFrame показывается размер (shape) и первые строки.
      Если таблица маленькая (<= top_n строк), выводится целиком.
    - Для Series показывается длина и первые элементы.
    - Для dict выводятся ключи и содержимое по каждому ключу.
    - Для list показываются первые элементы.
    - Для других типов выводится repr().
    """

    for key, value in reports.items():
        print(f"\n=== {key} ===")

        # Пустое значение
        if value is None:
            print("(None)")
            continue

        # DataFrame
        if isinstance(value, pd.DataFrame):
            print(f"DataFrame shape: {value.shape}")
            display(value.head(top_n))
            if value.shape[0] <= top_n:
                print("Full content:")
                display(value)
            continue

        # Series
        if isinstance(value, pd.Series):
            print(f"Series length: {len(value)}")
            display(value.head(top_n))
            continue

        # Dict (например hard_skills_by_dir)
        if isinstance(value, dict):
            print(f"Dict with {len(value)} keys: {list(value.keys())[:20]}")
            for subk, subv in value.items():
                print(f"\n-- {key}['{subk}'] --")
                if isinstance(subv, pd.DataFrame):
                    print(f"  DataFrame shape: {subv.shape}")
                    display(subv.head(top_n))
                elif isinstance(subv, pd.Series):
                    print(f"  Series length: {len(subv)}")
                    display(subv.head(top_n))
                else:
                    # Частые случаи: список, Counter, простые значения
                    display(subv)
            continue

        # List
        if isinstance(value, list):
            print(f"List length: {len(value)}")
            display(value[:top_n])
            continue

        # Другие типы
        print(repr(value))

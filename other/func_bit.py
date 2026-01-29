import os
import pandas as pd
import numpy as np
import re
import networkx as nx
import json
import time
#import plotly.graph_objects as go
#import plotly.express as px

from rapidfuzz import fuzz

import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from openpyxl.drawing.image import Image


from collections import Counter
from collections import defaultdict

from IPython.display import display
from tqdm import tqdm
from itertools import combinations

from typing import Tuple, Dict, Any, List, Optional, Any
from pathlib import Path
from openpyxl import Workbook
from pandas import DataFrame
from datetime import datetime, timedelta

# Загрузка переменных и словарей
from config_bit import (    
    DATA_FOLDER, REPORT_FOLDER, RESULT_FOLDER,     
    SCRAPING_DATE, SCRAPING_SYST_NAME, DEDUPLICATION_MODE, KEEP,
    AI_SPETIALIST_TERMS, JOB_TITLE,
    EMPLOYMENT_TYPES, WORK_TYPES,
    HARD_SKILLS, SOFT_SKILLS, LANG_PATTERNS, LANG_LEVELS, LANG_LEVEL_DESCRIPTIONS,
    GRADE_KEYWORDS, DIRECTION_PRIORITY, 
    CURRENCY_RATES, PERIOD_MULTIPLIERS,
    STEPSTONE_SEARCH_TITLES, STEPSTONE_SYNONYMS, STEPSTONE_KEYWORDS
)

# # Create folders if they don't exist
REPORT_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)


# ============================
# 1. Классификация направления
# ============================

def classify_direction(description_text: str, terms_dict: Dict[str, List[str]]) -> str:
    """
    Классифицирует описание вакансии по направлению.

    Args:
        description_text (str): текст описания вакансии.
        terms_dict (dict[str, list[str]]): словарь направлений и ключевых слов.

    Returns:
        str: направление или "Unclassified".
    """
    text_lower = str(description_text).lower()
    for direction, keywords in terms_dict.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return direction
    return "Unclassified"

# ============================
# 2. Парсинг даты
# ============================

_UNIT_TO_DELTA = {
    "minute": lambda n: timedelta(minutes=n),
    "minutes": lambda n: timedelta(minutes=n),
    "hour": lambda n: timedelta(hours=n),
    "hours": lambda n: timedelta(hours=n),
    "day": lambda n: timedelta(days=n),
    "days": lambda n: timedelta(days=n),
    "week": lambda n: timedelta(weeks=n),
    "weeks": lambda n: timedelta(weeks=n),
    "month": lambda n: timedelta(days=30 * n),   # упрощённо
    "months": lambda n: timedelta(days=30 * n),
    "year": lambda n: timedelta(days=365 * n),   # упрощённо
    "years": lambda n: timedelta(days=365 * n),
}

_TIME_RE = re.compile(r"(?P<num>\d+)\s*(?P<unit>minutes?|hours?|days?|weeks?|months?|years?)", flags=re.I)

def parse_date(published_at: str) -> Optional[datetime]:
    """
    Преобразует строки вида '3 days ago', '2 weeks ago' и т.п. в datetime.

    Args:
        published_at (str): строка с датой публикации.
        SCRAPING_DATE (datetime): дата скрепинга из config_bit.py

    Returns:
        Optional[datetime]: вычисленная дата публикации или None.
    """
    if not published_at:
        return None

    s = str(published_at).strip().lower()
    m = _TIME_RE.search(s)
    if not m:
        return None

    try:
        num = int(m.group("num"))
        unit = m.group("unit").lower()
        delta_fn = _UNIT_TO_DELTA.get(unit)
        if not delta_fn:
            return None
        return SCRAPING_DATE - delta_fn(num)
    except Exception:
        return None


########################################
# analyze_description
########################################

def analyze_description(text, current_salary, client) -> dict:
    # гарантируем строковый тип
    text = "" if pd.isna(text) else str(text)
    current_salary = "" if pd.isna(current_salary) else str(current_salary)

    result = {
        "directions_AI": "not specified",
        "hard_skills": "not specified",
        "soft_skills": "not specified",
        "salary": current_salary or "not specified"
    }

    prompt = (
        "Ты аналитик вакансий. Проанализируй описание и верни строго структурированный результат.\n"
        f"Directions_AI: выбери одно из [{', '.join(DIRECTION_PRIORITY)}]. "
        "Если нет явных признаков — верни 'not specified'.\n"
        f"Hard skills: выбирай только из списка [{', '.join(HARD_SKILLS)}]. "
        "Не добавляй soft skills и не придумывай новые.\n"
        f"Soft skills: выбирай только из списка [{', '.join(SOFT_SKILLS)}]. "
        "Не добавляй hard skills и не придумывай новые.\n"
        "Salary: если поле salary пустое, найди в описании; если не пустое, верни текущее. "
        "Формат зарплаты: €35K/yr - €40K/yr. "
        "Если указана конкурентная оплата — верни 'competitive'. "
        "Если зарплата не указана — верни 'not specified'.\n\n"
        f"Текущее значение salary: {current_salary}\n"
        f"Описание:\n{text}\n"
        "Формат ответа:\n"
        "Directions_AI: <значение>\n"
        "Hard skills: <список>\n"
        "Soft skills: <список>\n"
        "Salary: <значение>"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
        for line in output.splitlines():
            if line.lower().startswith("directions_ai:"):
                result["directions_AI"] = line.partition(":")[2].strip()
            elif line.lower().startswith("hard skills:"):
                result["hard_skills"] = line.partition(":")[2].strip()
            elif line.lower().startswith("soft skills:"):
                result["soft_skills"] = line.partition(":")[2].strip()
            elif line.lower().startswith("salary:"):
                extracted_salary = line.partition(":")[2].strip()
                if not current_salary.strip():
                    result["salary"] = extracted_salary
    except Exception as e:
        print("❌ Ошибка при анализе:", e)

    return result


########################################
# enrich_dataframe_with_skills_and_salary
########################################

def enrich_dataframe_with_skills_and_salary(df: pd.DataFrame, client) -> pd.DataFrame:
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Извлечение skills и salary"):
        res = extract_skills_and_salary(row["description_text"], row.get("salary_range", ""), client)
        # гарантируем наличие ключей
        res.setdefault("hard_skills", "not specified")
        res.setdefault("soft_skills", "not specified")
        res.setdefault("salary", row.get("salary_range", ""))
        results.append(res)

    parsed = pd.DataFrame(results)

    df = df.copy()
    df["hard_skills"] = parsed.get("hard_skills", "not specified")
    df["soft_skills"] = parsed.get("soft_skills", "not specified")
    df["salary_range"] = parsed.get("salary", df["salary_range"])

    return df




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
        if not loc or pd.isna(loc):
            return ("", "", "")
        parts = [p.strip() for p in str(loc).split(",")]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            return parts[0], parts[1], ""
        elif len(parts) == 1:
            return parts[0], "", ""
        else:
            return ("", "", "")


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
    """
    Очищает и нормализует заголовки вакансий (особенно в немецком контексте).

    Args:
        title (str): исходный заголовок вакансии.

    Returns:
        str: нормализованный заголовок.
    """
    if pd.isna(title):
        return ""

    s = str(title)

    # Удаление (m/w/d) и подобных
    s = re.sub(r"[^[^]*m\s*\/\s*w\s*\/\s*d[^\)]*\)", " ", s, flags=re.I)
    s = re.sub(r"\b[mM]\s*\/\s*[wW]\s*\/\s*[dD]\b", " ", s)

    # Удаление любых скобок
    s = re.sub(r".∗?.*?", " ", s)

    # Разрешённые символы
    s = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s\-/&\.]", " ", s)

    # Нормализация пробелов
    s = re.sub(r"\s+", " ", s).strip()

    return s.lower()


# ============================
# Нормализация текста
# ============================

def normalize_text(s: str) -> str:
    """
    Нормализует текст: убирает лишние пробелы, приводит к нижнему регистру.

    Args:
        s (str): исходный текст.

    Returns:
        str: нормализованный текст.
    """
    if pd.isna(s):
        return ""

    t = str(s).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t




# ########################################
# Извлечение уровня должности
# ########################################

def extract_grade(title_clean: str, grade_keywords: Dict[str, List[str]]) -> Optional[str]:
    """
    Определяет уровень должности (junior, senior, lead и т.д.) по ключевым словам.

    Args:
        title_clean (str): нормализованный заголовок вакансии.
        grade_keywords (dict[str, list[str]]): словарь ключевых слов для уровней.

    Returns:
        Optional[str]: уровень должности или None.
    """
    t = title_clean.lower()
    for grade, keywords in grade_keywords.items():
        for k in keywords:
            if re.search(k, t):
                return grade
    return None


########################################
# 2. Парсер зарплат (упрощённый, расширяемый)
########################################
number_re = re.compile(r"[\d\.,]+")

def parse_salary_field(s: str) -> Dict[str, Optional[float]]:
    """
    Извлекает информацию о зарплате из текстового поля вакансии.

    Args:
        s (str): строка с зарплатой.

    Returns:
        dict: словарь с ключами:
            - salary_min (float | None): минимальная зарплата
            - salary_max (float | None): максимальная зарплата
            - currency (str | None): валюта ("EUR", "USD")
            - period (str | None): период ("year", "month", "hour")
    """
    out = {"salary_min": None, "salary_max": None, "currency": None, "period": None}

    if pd.isna(s) or not str(s).strip():
        return out

    txt = str(s).lower()

    # Определение валюты
    if "€" in txt or "eur" in txt:
        out["currency"] = "EUR"
    elif "$" in txt or "usd" in txt:
        out["currency"] = "USD"

    # Определение периода
    if any(k in txt for k in ["per year", "per annum", "year", "jahres", "jahresgehalt", "pa"]):
        out["period"] = "year"
    elif any(k in txt for k in ["per month", "monat", "/month"]):
        out["period"] = "month"
    elif any(k in txt for k in ["per hour", "stunde", "/hour", "hour"]):
        out["period"] = "hour"

    # Извлечение чисел
    nums = number_re.findall(txt.replace(" ", ""))
    nums = [n.replace(".", "").replace(",", ".") for n in nums]

    nums_f = []
    for n in nums:
        try:
            nums_f.append(float(n))
        except ValueError:
            continue

    if len(nums_f) == 1:
        out["salary_min"] = nums_f[0]
        out["salary_max"] = nums_f[0]
    elif len(nums_f) >= 2:
        out["salary_min"] = min(nums_f[0], nums_f[1])
        out["salary_max"] = max(nums_f[0], nums_f[1])

    return out

########################################
# to_annual_eur зарплату в годовой эквивалент в евро
########################################
def to_annual_eur(salary_min: Optional[float], period: Optional[str], currency: Optional[str]) -> Optional[float]:
    """
    Преобразует зарплату в годовой эквивалент в евро.

    Args:
        salary_min (float | None): минимальная зарплата.
        period (str | None): период ("year", "month", "hour").
        currency (str | None): валюта ("EUR", "USD").

    Returns:
        float | None: годовой эквивалент зарплаты в евро.
    """
    if salary_min is None:
        return None

    val = salary_min

    if period == "month":
        annual = val * 12
    elif period == "hour":
        annual = val * 160 * 12  # эвристика: 160 часов × 12 месяцев
    else:
        annual = val

    # конвертация валюты (пример: USD → EUR)
    if currency and currency != "EUR":
        # предполагается словарь CURRENCY_RATES
        rate = CURRENCY_RATES.get(currency, 1.0)
        return annual * rate

    return annual

########################################
# Experience extraction
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


# ============================
# Извлечение навыков
# ============================

def _flex_pattern(term: str) -> str:
    """
    Создаёт регекс для термина, устойчивый к пробелам/дефисам.

    Args:
        term (str): навык.

    Returns:
        str: регулярное выражение.
    """
    parts = re.split(r"[ \-]+", term.strip())
    core = r"[\\s\\-]+".join(re.escape(p) for p in parts if p)
    return rf"\b{core}\b"

def extract_skills(text: str, skills_list: List[str]) -> List[str]:
    """
    Извлекает навыки из текста.

    Args:
        text (str): описание вакансии.
        skills_list (list[str]): список навыков для поиска.

    Returns:
        list[str]: найденные навыки.
    """
    if not text:
        return []

    t = str(text)
    canonical_map = {s.lower(): s for s in skills_list}
    found = set()

    # Специальный кейс для Make
    if re.search(r"\bmake\b", t, flags=re.I) and re.search(
        r"(automation|workflow|integration|platform|no[\s\-]?code|low[\s\-]?code|zapier|n8n|uipath|power\s*automate)",
        t,
        flags=re.I,
    ):
        if "make" in canonical_map:
            found.add(canonical_map["make"])

    # Общий матчинг
    for skill in skills_list:
        sl = skill.lower()
        if sl == "make":
            continue
        pat = _flex_pattern(skill)
        if re.search(pat, t, flags=re.I):
            found.add(canonical_map[sl])

    return sorted(found)


########################################
# Titles clusters / top formulations
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
# clean_job_titles
########################################

# основные локации DACH‑региона (Германия, Австрия, Швейцария)
LOCATIONS = {
    # Германия
    "Berlin", "Hamburg", "Munich", "München", "Cologne", "Köln", "Frankfurt",
    "Stuttgart", "Düsseldorf", "Leipzig", "Dresden", "Hannover", "Nuremberg", "Nürnberg",
    "Bremen", "Essen", "Dortmund", "Bonn", "Mannheim", "Karlsruhe", "Mainz",
    "Aachen", "Wiesbaden", "Freiburg", "Regensburg",

    # Австрия
    "Vienna", "Wien", "Graz", "Linz", "Salzburg", "Innsbruck", "Klagenfurt",
    "Villach", "St. Pölten", "Wels",

    # Швейцария
    "Zurich", "Zürich", "Geneva", "Genf", "Basel", "Bern", "Lausanne",
    "Lucerne", "Luzern", "St. Gallen", "Winterthur", "Lugano"
}


def clean_job_titles(titles: list[str]) -> list[str]:
    cleaned = []
    
    gender_pattern = re.compile(
        r"\(?\b(?:m|w|d|f|x)(?:\s*[/\-•]\s*(?:m|w|d|f|x)){1,2}\b\)?",
        flags=re.IGNORECASE
    )

    for title in titles:
        if pd.isna(title) or not str(title).strip():
            cleaned.append("")
            continue

        t = str(title).strip()

        # 1. Удаляем гендерные маркеры (m/w/d), (w/m/d), m-w-d, m•w•d, m w d
        t = gender_pattern.sub("", t)

        # 2. Analyst*in → Analyst
        t = re.sub(r"(\w+)\*in\b", r"\1", t, flags=re.IGNORECASE)

        # 3. Удаляем @ Company
        t = re.sub(r"@.*", "", t)

        # 4. Удаляем ":in"
        t = re.sub(r":in\b", "", t, flags=re.IGNORECASE)

        # 5. Удаляем "für das ..."
        t = re.sub(r"für das.*", "", t, flags=re.IGNORECASE)

        # 6. Удаляем ведущие C++ и подобные маркеры
        t = re.sub(r"^C\+\+\s*", "", t)

        # 7. Разбор " - "
        parts = t.split(" - ")
        if len(parts) > 1:
            left, right = parts[0].strip(), parts[1].strip()
            if left in LOCATIONS or re.match(r"^[A-Z]{2,}$", left):
                t = right
            else:
                t = left

        # 8. Удаляем хвосты в скобках
        t = re.sub(r"\s*\([^)]*\)$", "", t)

        # 9. Обрезаем по разделителям
        for sep in ["...", "|", ",", "–", ":"]:
            t = t.split(sep)[0]

        # 10. Финальная очистка
        t = re.sub(r"\s+", " ", t).strip(" -–,;/").strip()

        cleaned.append(t)

    return cleaned



########################################
# plot_vacancy_dynamics
########################################

# ========================== Функция: динамика вакансий ==========================
def plot_vacancy_dynamics(df_clean, job_title, six_months_ago, scraping_date, report_folder: Path, syst_name):
    """Строит динамику вакансий по месяцам и сохраняет интерактивный HTML в REPORT_FOLDER."""
    
    # список месяцев (последние 6)
    all_months = pd.date_range(start=six_months_ago, end=df_clean["posted_at"].max(), freq="MS")
    all_months_str = all_months.strftime("%m-%Y")

    # базовая таблица
    base = pd.DataFrame([(m, v) for m in all_months_str for v in job_title], 
                        columns=["posted_at", "search_term"])

    # реальные значения
    actual = (
        df_clean[df_clean["posted_at"] >= six_months_ago]
        .groupby([df_clean["posted_at"].dt.to_period("M").astype(str), "search_term"])
        .size()
        .reset_index(name="count")
    )
    actual["posted_at"] = pd.to_datetime(actual["posted_at"]).dt.strftime("%m-%Y")

    # объединение
    merged = pd.merge(base, actual, on=["posted_at", "search_term"], how="left").fillna(0)
    merged["count"] = merged["count"].astype(int)

    # строка "All vacancies"
    all_vacancies = (
        merged.groupby("posted_at")["count"].sum().reset_index()
        .assign(search_term="All vacancies")
    )
    plot_data = pd.concat([merged, all_vacancies], ignore_index=True)

    # график
    fig = px.bar(
        plot_data[plot_data["search_term"] == "All vacancies"],
        x="posted_at",
        y="count",
        text="count",
        title=f"Динамика вакансий {syst_name} за полгода от даты скрепинга ({scraping_date.strftime('%d.%m.%Y')})",
        labels={"posted_at": "Месяц", "count": ""},
        color_discrete_sequence=["#4472C4"]
    )

    # выпадающий список
    buttons = []
    for term in ["All vacancies"] + job_title:
        filtered = plot_data[plot_data["search_term"] == term]
        buttons.append(
            dict(
                label=term,
                method="update",
                args=[
                    {
                        "x": [filtered["posted_at"]],
                        "y": [filtered["count"]],
                        "text": [filtered["count"]],
                        "marker": {"color": "#4472C4"},
                        "type": "bar"
                    },
                    {
                        "title": {
                            "text": f"Динамика вакансий {syst_name} за полгода от даты скрепинга ({scraping_date.strftime('%d.%m.%Y')})",
                            "font": dict(family="Arial Bold, sans-serif", size=16, color="darkblue")
                        }
                    }
                ]
            )
        )

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0,
            y=1.05,
            xanchor="left",
            yanchor="top"
        )],
        yaxis=dict(showticklabels=False),
        xaxis=dict(tickangle=-45, tickmode="array", tickvals=all_months_str, ticktext=all_months_str),
    )
    fig.update_traces(textposition="outside")

    # сохранение
    chart_file_name = f"04_{SCRAPING_SYST_NAME}_vacancies_dynamics.html"
    output_path = report_folder / chart_file_name
    fig.write_html(output_path, auto_open=False)
    print(f"Интерактивная диаграмма по динамике вакансий сохранёна: {output_path}")
    return fig

########################################
# analyze_junior_skills Функция: анализ скиллов джунов
########################################
 
def analyze_skills_by_levels(
    df_clean: pd.DataFrame,
    levels: List[str],
    hard_skills_dict: List[str],
    soft_skills_dict: List[str]
) -> Tuple[Counter, Counter]:
    """
    Извлекает HARD и SOFT skills для указанных уровней (например Entry, Associate, Internship).

    Args:
        df_clean (pd.DataFrame): очищенный датафрейм с колонками 'experienceLevel' и 'description'.
        levels (list[str]): список уровней для фильтрации (например ['Entry level','Associate','Internship']).
        hard_skills_dict (list[str]): список hard skills для поиска.
        soft_skills_dict (list[str]): список soft skills для поиска.

    Returns:
        Tuple[Counter, Counter]: счётчики hard и soft skills.
    """
    # фильтруем по указанным уровням
    subset = df_clean[df_clean["experienceLevel"].isin(levels)].copy()

    # извлекаем навыки
    subset["hard_skills"] = subset["description"].apply(lambda x: extract_skills(str(x), hard_skills_dict))
    subset["soft_skills"] = subset["description"].apply(lambda x: extract_skills(str(x), soft_skills_dict))

    # считаем частоты
    hard_counter = Counter([s for skills in subset["hard_skills"] for s in skills])
    soft_counter = Counter([s for skills in subset["soft_skills"] for s in skills])

    return hard_counter, soft_counter


########################################
# plot_junior_skills_interactive визуализация ТОП скиллов джунов
########################################
 
def plot_junior_skills_interactive(hard_counter, soft_counter, scraping_date, report_folder: Path, syst_name: str):
    """Строит одну диаграмму с переключателем HARD/SOFT skills и сохраняет в REPORT_FOLDER."""

    # преобразуем в DataFrame топ-10
    hard_df = pd.DataFrame(hard_counter.most_common(10), columns=["skill", "count"])
    soft_df = pd.DataFrame(soft_counter.most_common(10), columns=["skill", "count"])

    # базовый график (по умолчанию HARD)
    fig = px.bar(
        hard_df,
        x="skill",
        y="count",
        text="count",
        title=f"Top HARD-skills джунов ({syst_name}, за пол года от {scraping_date.strftime('%d.%m.%Y')})",
        labels={"skill": "Навык", "count": "Количество"},
        color_discrete_sequence=["#4472C4"]
    )
    fig.update_traces(textposition="outside")

    # кнопки переключения
    buttons = [
        dict(
            label="HARD skills",
            method="update",
            args=[
                {"x": [hard_df["skill"]], "y": [hard_df["count"]], "text": [hard_df["count"]],
                 "marker": {"color": "#4472C4"}, "type": "bar"},
                {"title": {"text": f"Top HARD-skills джунов ({syst_name},  за пол года от {scraping_date.strftime('%d.%m.%Y')})",
                           "font": dict(family="Arial Bold, sans-serif", size=16, color="darkblue")}}
            ]
        ),
        dict(
            label="SOFT skills",
            method="update",
            args=[
                {"x": [soft_df["skill"]], "y": [soft_df["count"]], "text": [soft_df["count"]],
                 "marker": {"color": "#70AD47"}, "type": "bar"},
                {"title": {"text": f"Top SOFT-skills джунов ({syst_name}, за пол года от {scraping_date.strftime('%d.%m.%Y')})",
                           "font": dict(family="Arial Bold, sans-serif", size=16, color="darkgreen")}}
            ]
        )
    ]

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=buttons,
            direction="down",  
            showactive=True,
            x=1.05,
            y=1.0,
            xanchor="left",
            yanchor="top"
        )],
        xaxis_tickangle=-30
    )


    # сохранение
    chart_file_name = f"05_{SCRAPING_SYST_NAME}_junior_skills_interactive.html"
    output_path = report_folder / chart_file_name
    fig.write_html(output_path, auto_open=False)
    print(f"Интерактивная диаграмма Top HARD-,SOFTl-skills сохранена в: {output_path}")
    #fig.show()
    return fig


########################################
# plot_vacancy_distribution
########################################

def plot_vacancy_distribution_seaborn(
    df_clean: pd.DataFrame,
    job_title_list: list,
    scraping_syst_name: str,
    report_folder: Path
):
    chart_name = (
        f"{scraping_syst_name}_распределение вакансий по поисковым тайтлам "
        f"после фильтрации для JUNIOR LEVEL"
    )

    counts = df_clean["search_term"].value_counts()
    filtered_counts = counts.reindex(job_title_list, fill_value=0).reset_index()
    filtered_counts.columns = ["job_title", "count"]

    palette = get_system_palette(scraping_syst_name, n_colors=len(filtered_counts))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=filtered_counts,
        x="job_title",
        y="count",
        hue="job_title",
        palette=palette,
        legend=False,
        ax=ax
    )

    for i, v in enumerate(filtered_counts["count"]):
        ax.text(i, v + max(filtered_counts["count"]) * 0.02, str(v),
                ha="center", va="bottom", fontsize=10)

    ax.set_title(chart_name, fontsize=14, fontweight="bold")
    ax.set_xlabel("Поисковый тайтл")
    ax.set_ylabel("Количество вакансий")
    ax.set_xticks(range(len(filtered_counts)))
    ax.set_xticklabels(filtered_counts["job_title"], rotation=45, ha="right")

    plt.tight_layout()

    chart_file_name = f"03_{scraping_syst_name}_vacancy_distribution.png"
    output_path = report_folder / chart_file_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Диаграмма распределения вакансий сохранена: {output_path}")
    return fig




########################################
# plot_experience_distribution
########################################

def plot_experience_distribution_seaborn(
    df_clean: pd.DataFrame,
    report_folder: Path,
    filename: str,
    title_name: str,
    syst_name: str
):
    # группировка
    distribution = (
        df_clean["experienceLevel"]
        .value_counts()
        .rename_axis("experienceLevel")
        .reset_index(name="count")
    )

    palette = get_system_palette(syst_name, n_colors=len(distribution))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=distribution,
        x="experienceLevel",
        y="count",
        hue="experienceLevel",
        palette=palette,
        legend=False,
        ax=ax
    )

    for i, v in enumerate(distribution["count"]):
        ax.text(i, v + max(distribution["count"]) * 0.02, str(v),
                ha="center", va="bottom", fontsize=10)

    ax.set_title(title_name, fontsize=14, fontweight="bold")
    ax.set_xlabel("Уровень опыта")
    ax.set_ylabel("Количество вакансий")
    ax.set_xticks(range(len(distribution)))
    ax.set_xticklabels(distribution["experienceLevel"], rotation=30, ha="right")

    plt.tight_layout()

    output_path = report_folder / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Диаграмма '{title_name}' сохранена: {output_path}")
    return fig


########################################
# plot_vacancy_dynamics_separate
########################################

def plot_vacancy_dynamics_separate_seaborn(
    df_clean: pd.DataFrame,
    job_titles: list[str],
    six_months_ago,
    scraping_date,
    syst_name: str
):
    df_work = df_clean.copy()
    df_work["posted_at"] = pd.to_datetime(df_work["posted_at"], errors="coerce")
    df_work["posted_at"] = df_work["posted_at"].dt.tz_localize(None)

    six_months_ago_ts = pd.to_datetime(six_months_ago)
    max_date = df_work["posted_at"].max()

    all_months = pd.date_range(start=six_months_ago_ts, end=max_date, freq="MS")
    all_months_str = all_months.strftime("%m-%Y")

    base = pd.DataFrame(
        [(m, v) for m in all_months_str for v in job_titles],
        columns=["posted_at", "search_term"]
    )

    actual = (
        df_work[df_work["posted_at"] >= six_months_ago_ts]
        .assign(posted_at=df_work["posted_at"].dt.to_period("M").astype(str))
        .groupby(["posted_at", "search_term"])
        .size()
        .reset_index(name="count")
    )
    actual["posted_at"] = pd.to_datetime(actual["posted_at"]).dt.strftime("%m-%Y")

    merged = pd.merge(base, actual, on=["posted_at", "search_term"], how="left").fillna(0)
    merged["count"] = merged["count"].astype(int)

    all_vacancies = (
        merged.groupby("posted_at")["count"]
        .sum()
        .reset_index()
        .assign(search_term="All vacancies")
    )

    plot_data = pd.concat([merged, all_vacancies], ignore_index=True)

    sns.set_theme(style="whitegrid")

    # один цвет на систему
    if syst_name.lower() == "linkedin":
        bar_color = "#6A0DAD"
    else:
        bar_color = "#1F4E79"

    figs = {}

    for term in ["All vacancies"] + job_titles:
        filtered = plot_data[plot_data["search_term"] == term]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=filtered,
            x="posted_at",
            y="count",
            color=bar_color,
            ax=ax
        )

        for i, v in enumerate(filtered["count"]):
            ax.text(i, v + (max(filtered["count"]) * 0.02 if max(filtered["count"]) > 0 else 0.1),
                    str(v), ha="center", va="bottom", fontsize=9)

        ax.set_title(
            f"Динамика вакансий ({term}) {syst_name}\nза полгода от {scraping_date.strftime('%d.%m.%Y')}",
            fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Количество вакансий")

        ax.set_xticks(range(len(all_months_str)))
        ax.set_xticklabels(all_months_str, rotation=45, ha="right")

        plt.tight_layout()
        figs[term] = fig

    return figs



########################################
# plot_junior_skills_separate
########################################

def plot_junior_skills_separate_seaborn(
    hard_counter,
    soft_counter,
    scraping_date,
    syst_name: str
):
    hard_df = pd.DataFrame(hard_counter.most_common(10), columns=["skill", "count"])
    soft_df = pd.DataFrame(soft_counter.most_common(10), columns=["skill", "count"])

    sns.set_theme(style="whitegrid")

    # цвета под систему
    if syst_name.lower() == "linkedin":
        color_hard = "#6A0DAD"
        color_soft = "#9B30FF"
    else:
        color_hard = "#1F4E79"
        color_soft = "#4472C4"

    # HARD
    fig_hard, ax_hard = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=hard_df,
        x="skill",
        y="count",
        color=color_hard,
        ax=ax_hard
    )

    for i, v in enumerate(hard_df["count"]):
        ax_hard.text(i, v + max(hard_df["count"]) * 0.02, str(v),
                     ha="center", va="bottom", fontsize=9)

    ax_hard.set_title(
        f"Top HARD-skills джунов ({syst_name}, за пол года от {scraping_date.strftime('%d.%m.%Y')})",
        fontsize=14, fontweight="bold"
    )
    ax_hard.set_xlabel("Навык")
    ax_hard.set_ylabel("Количество")
    ax_hard.set_xticks(range(len(hard_df)))
    ax_hard.set_xticklabels(hard_df["skill"], rotation=30, ha="right")
    plt.tight_layout()

    # SOFT
    fig_soft, ax_soft = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=soft_df,
        x="skill",
        y="count",
        color=color_soft,
        ax=ax_soft
    )

    for i, v in enumerate(soft_df["count"]):
        ax_soft.text(i, v + max(soft_df["count"]) * 0.02, str(v),
                     ha="center", va="bottom", fontsize=9)

    ax_soft.set_title(
        f"Top SOFT-skills джунов ({syst_name}, за пол года от {scraping_date.strftime('%d.%m.%Y')})",
        fontsize=14, fontweight="bold"
    )
    ax_soft.set_xlabel("Навык")
    ax_soft.set_ylabel("Количество")
    ax_soft.set_xticks(range(len(soft_df)))
    ax_soft.set_xticklabels(soft_df["skill"], rotation=30, ha="right")
    plt.tight_layout()

    return fig_hard, fig_soft




# ============================================================
# Stepstone‑функции EXPERIENCE CLASSIFIER
# ============================================================

def extract_experience(text, experienceLevel=None):
    """
    Определяет требуемый уровень опыта для вакансии.

    Логика работы:
    1. Анализирует текст вакансии (title + description):
       - Если встречаются признаки студенческих/учебных позиций 
         (Werkstudent, Praktikum, Ausbildung, Studium, Trainee и т.п.) → "no_experience".
       - Если встречаются фразы про первые профессиональные навыки 
         ("erste Erfahrungen", "erste Berufserfahrung", "mindestens 1 Jahr") → "1-2_years".
       - Если встречаются фразы про многолетний опыт 
         ("mehrjährige Berufserfahrung", "mindestens 3 Jahre", "5 Jahre") → "3+_years".

    2. Если в тексте нет признаков опыта, анализирует поле employmentType:
       - Если оно относится к учебным/студенческим категориям 
         (Ausbildung, Studium, Praktikum, Werkstudent, Trainee) → "no_experience".
       - "Feste Anstellung" НЕ считается признаком опыта → возвращает "unknown".

    3. Если ни текст, ни employmentType не дают информации → "unknown".

    Параметры:
        text (str): текст вакансии (title + description).
        experienceLevel (str): значение поля employmentType.

    Возвращает:
        str: одна из категорий:
             "no_experience", "1-2_years", "3+_years", "unknown".
    """

    
    # 1. Анализ текста вакансии
    if not isinstance(text, str):
        text = ""
    t = text.lower()

    # 0 — опыт не требуется
    no_exp_patterns = [
        "werkstudent", "praktikum", "ausbildung", "studium",
        "duales studium", "trainee", "berufseinstieg",
        "studentenjob", "studentenjobs", "werkstudenten",
        "bachelor-/master-/diplom-arbeiten"
    ]
    if any(p in t for p in no_exp_patterns):
        return "Entry level"

    # 1–2 года опыта
    mid_exp_patterns = [
        "erste erfahrungen", "erste berufserfahrung",
        "erste praktische erfahrungen",
        "mindestens einjährige", "mindestens 1 jahr",
        "1 jahr", "1-2 jahre", "1 bis 2 jahre"
    ]
    if any(p in t for p in mid_exp_patterns):
        return "Associate"

    # 3+ лет опыта
    high_exp_patterns = [
        "mehrjährige", "mehrere jahre", "mindestens 3 jahre",
        "mindestens drei jahre", "mindestens 5 jahre",
        "5 jahre", "langjährige"
    ]
    if any(p in t for p in high_exp_patterns):
        return "3+ years"


    # 2. Анализ employmentType
    if isinstance(experienceLevel, str):
        et = experienceLevel.lower()

        # категории без опыта
        if any(p in et for p in no_exp_patterns):
            return "Entry level"

        # trainee → no experience
        if "trainee" in et:
            return "Intership"

        # Ausbildung, Studium → no experience
        if "ausbildung" in et or "studium" in et:
            return "Entry level"

        # Werkstudent → no experience
        if "werkstudent" in et:
            return "Entry level"

        # Praktikum → no experience
        if "praktikum" in et:
            return "Intership"

        # Feste Anstellung → НЕ опыт → вернуть unknown
        if "feste anstellung" in et:
            return "unknown"

    
    # 3. Если ничего не найдено    
    return "unknown"


# ============================================================
# Stepstone‑функции SOFT CLASSIFIER FOR SEARCH TITLES
# ============================================================

def classify_soft(row):
    """
    Выполняет мягкую классификацию вакансии по одному из заранее заданных search_title.

    Логика работы:
    1. Объединяет title и description в единый текст и приводит его к нижнему регистру.
    2. Для каждого search_title вычисляет три независимых сигнала:
       
       • fuzzy_score — степень текстового сходства между вакансией и названием search_title
         (используются partial_ratio, token_set_ratio и WRatio из RapidFuzz).

       • synonym_score — количество совпавших синонимов, специфичных для данного search_title.

       • keyword_score — количество совпавших ключевых слов, характерных для search_title.

    3. Комбинирует сигналы в итоговый final_score по формуле:
           final_score = 0.5*fuzzy + 0.3*synonyms + 0.2*keywords

       Такой подход делает классификацию мягкой и устойчивой:
       fuzzy даёт общий смысл, синонимы усиливают точность, ключевые слова добавляют контекст.

    4. Выбирает search_title с максимальным final_score.

    Возвращает:
        pandas.Series с полями:
            - search_title: выбранная категория
            - final_score: итоговый взвешенный скор
            - fuzzy_score: степень текстового сходства
            - synonym_score: количество совпавших синонимов
            - keyword_score: количество совпавших ключевых слов
            - matched_synonyms: список найденных синонимов

    Функция обеспечивает прозрачную и объяснимую классификацию,
    позволяя анализировать вклад каждого сигнала в итоговое решение.
    """
    
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()

    scores = []
    fuzzy_scores = []
    synonym_scores = []
    keyword_scores = []
    matched_synonyms = []

    for st in STEPSTONE_SEARCH_TITLES:

        fuzzy_score = max(
            fuzz.partial_ratio(text, st.lower()),
            fuzz.token_set_ratio(text, st.lower()),
            fuzz.WRatio(text, st.lower())
        ) / 100

        syn_hits = [s for s in STEPSTONE_SYNONYMS.get(st, []) if s in text]
        syn_score = len(syn_hits)

        kw_score = sum(1 for kw in STEPSTONE_KEYWORDS.get(st, []) if kw in text)

        final_score = 0.5 * fuzzy_score + 0.3 * syn_score + 0.2 * kw_score

        scores.append(final_score)
        fuzzy_scores.append(fuzzy_score)
        synonym_scores.append(syn_score)
        keyword_scores.append(kw_score)
        matched_synonyms.append(syn_hits)

    best_idx = int(np.argmax(scores))

    return pd.Series({
        "search_term": STEPSTONE_SEARCH_TITLES[best_idx],
        "final_score": scores[best_idx],
        "fuzzy_score": fuzzy_scores[best_idx],
        "synonym_score": synonym_scores[best_idx],
        "keyword_score": keyword_scores[best_idx],
        "matched_synonyms": matched_synonyms[best_idx]
    })



def get_system_palette(syst_name: str, n_colors: int):
    """
    Возвращает палитру нужной длины в зависимости от системы:
    - LinkedIn  → фиолетовая гамма
    - Stepstone → синяя гамма
    """
    syst = (syst_name or "").lower()
    if syst == "linkedin":
        return sns.color_palette("Purples", n_colors=n_colors)
    else:
        return sns.color_palette("Blues", n_colors=n_colors)

import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################################
#===================================Общий стиль и палитры
sns.set_theme(style="whitegrid")

SYSTEM_COLORS = {
    "LinkedIn": "#9B30FF",   # фиолетовый
    "Stepstone": "#4472C4",  # синий
}

def normalize_system_name(name: str) -> str:
    name = (name or "").strip()
    if name.lower().startswith("link"):
        return "LinkedIn"
    if name.lower().startswith("step"):
        return "Stepstone"
    return name or "Unknown"


# 1. Сравнение распределения уровней опыта
def plot_experience_distribution_compare(
    df_linkedin: pd.DataFrame,
    df_stepstone: pd.DataFrame,
    report_folder: Path,
    filename: str = "01_experience_distribution_compare.png",
    title: str = "Распределение вакансий по уровням опыта: LinkedIn vs Stepstone"
):
    df_l = df_linkedin.copy()
    df_s = df_stepstone.copy()

    df_l["system"] = "LinkedIn"
    df_s["system"] = "Stepstone"

    df_all = pd.concat([df_l, df_s], ignore_index=True)

    dist = (
        df_all.groupby(["system", "experienceLevel"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.barplot(
        data=dist,
        x="experienceLevel",
        y="count",
        hue="system",
        palette=SYSTEM_COLORS,
        ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Уровень опыта")
    ax.set_ylabel("Количество вакансий")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    output_path = report_folder / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] {title} → {output_path}")
    return fig

# 2. Сравнение распределения по search_term
def plot_vacancy_distribution_compare(
    df_linkedin: pd.DataFrame,
    df_stepstone: pd.DataFrame,
    job_titles: list[str],
    report_folder: Path,
    filename: str = "02_vacancy_distribution_compare.png",
    title: str = "Распределение вакансий по поисковым тайтлам: LinkedIn vs Stepstone"
):
    df_l = df_linkedin.copy()
    df_s = df_stepstone.copy()

    df_l["system"] = "LinkedIn"
    df_s["system"] = "Stepstone"

    df_all = pd.concat([df_l, df_s], ignore_index=True)

    dist = (
        df_all[df_all["search_term"].isin(job_titles)]
        .groupby(["system", "search_term"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.barplot(
        data=dist,
        x="search_term",
        y="count",
        hue="system",
        palette=SYSTEM_COLORS,
        ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Поисковый тайтл")
    ax.set_ylabel("Количество вакансий")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    output_path = report_folder / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] {title} → {output_path}")
    return fig

# 3. Сравнение динамики вакансий по месяцам
def plot_vacancy_dynamics_compare(
    df_linkedin: pd.DataFrame,
    df_stepstone: pd.DataFrame,
    job_titles: list[str],
    report_folder: Path,
    filename: str = "03_vacancy_dynamics_compare.png",
    title: str = "Динамика вакансий по месяцам: LinkedIn vs Stepstone"
):
    df_l = df_linkedin.copy()
    df_s = df_stepstone.copy()

    df_l["system"] = "LinkedIn"
    df_s["system"] = "Stepstone"

    df_all = pd.concat([df_l, df_s], ignore_index=True)
    df_all["posted_at"] = pd.to_datetime(df_all["posted_at"], errors="coerce")
    df_all["month"] = df_all["posted_at"].dt.to_period("M").astype(str)

    dyn = (
        df_all[df_all["search_term"].isin(job_titles)]
        .groupby(["system", "month"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.lineplot(
        data=dyn,
        x="month",
        y="count",
        hue="system",
        marker="o",
        palette=SYSTEM_COLORS,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Количество вакансий")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    output_path = report_folder / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] {title} → {output_path}")
    return fig

# 4. Сравнение топ‑скиллов (HARD/ SOFT)
from collections import Counter

def plot_skills_compare(
    hard_linkedin: Counter,
    soft_linkedin: Counter,
    hard_stepstone: Counter,
    soft_stepstone: Counter,
    scraping_date,
    report_folder: Path,
    filename_prefix: str = "04_skills_compare"
):
    def build_df(counter: Counter, system: str, skill_type: str, top_n: int = 10):
        return pd.DataFrame(
            [(skill, count, system, skill_type) for skill, count in counter.most_common(top_n)],
            columns=["skill", "count", "system", "type"]
        )

    df_hard = pd.concat([
        build_df(hard_linkedin, "LinkedIn", "HARD"),
        build_df(hard_stepstone, "Stepstone", "HARD"),
    ], ignore_index=True)

    df_soft = pd.concat([
        build_df(soft_linkedin, "LinkedIn", "SOFT"),
        build_df(soft_stepstone, "Stepstone", "SOFT"),
    ], ignore_index=True)

    # HARD
    fig_hard, ax_hard = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=df_hard,
        x="skill",
        y="count",
        hue="system",
        palette=SYSTEM_COLORS,
        ax=ax_hard
    )
    ax_hard.set_title(
        f"Top HARD-skills джунов: LinkedIn vs Stepstone\nза полгода от {scraping_date.strftime('%d.%m.%Y')}",
        fontsize=14, fontweight="bold"
    )
    ax_hard.set_xlabel("Навык")
    ax_hard.set_ylabel("Количество")
    #ax_hard.set_xticklabels(ax_hard.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax_hard.get_xticklabels(), rotation=30, ha="right")
    
    for container in ax_hard.containers:
        ax_hard.bar_label(container, fmt="%d", padding=3, fontsize=9)
    plt.tight_layout()
    out_hard = report_folder / f"{filename_prefix}_hard.png"
    fig_hard.savefig(out_hard, dpi=200)
    plt.close(fig_hard)
    print(f"[OK] HARD-skills → {out_hard}")

    # SOFT
    fig_soft, ax_soft = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=df_soft,
        x="skill",
        y="count",
        hue="system",
        palette=SYSTEM_COLORS,
        ax=ax_soft
    )
    ax_soft.set_title(
        f"Top SOFT-skills джунов: LinkedIn vs Stepstone\nза полгода от {scraping_date.strftime('%d.%m.%Y')}",
        fontsize=14, fontweight="bold"
    )
    ax_soft.set_xlabel("Навык")
    ax_soft.set_ylabel("Количество")
    #ax_soft.set_xticklabels(ax_soft.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax_soft.get_xticklabels(), rotation=30, ha="right")
    
    for container in ax_soft.containers:
        ax_soft.bar_label(container, fmt="%d", padding=3, fontsize=9)
    plt.tight_layout()
    out_soft = report_folder / f"{filename_prefix}_soft.png"
    fig_soft.savefig(out_soft, dpi=200)
    plt.close(fig_soft)
    print(f"[OK] SOFT-skills → {out_soft}")

    return out_hard, out_soft


# 5. Сборка полного отчёта LinkedIn vs Stepstone
def build_full_comparison_report(
    df_linkedin_clean: pd.DataFrame,
    df_stepstone_clean: pd.DataFrame,
    hard_linkedin: Counter,
    soft_linkedin: Counter,
    hard_stepstone: Counter,
    soft_stepstone: Counter,
    job_titles: list[str],
    scraping_date,
    report_folder: Path
):
    report_folder.mkdir(exist_ok=True, parents=True)

    plot_experience_distribution_compare(
        df_linkedin_clean,
        df_stepstone_clean,
        report_folder,
        "01_experience_distribution_compare.png",
        "Распределение вакансий по уровням опыта: LinkedIn vs Stepstone"
    )

    plot_vacancy_distribution_compare(
        df_linkedin_clean,
        df_stepstone_clean,
        job_titles,
        report_folder,
        "02_vacancy_distribution_compare.png",
        "Распределение вакансий по поисковым тайтлам: LinkedIn vs Stepstone"
    )

    plot_vacancy_dynamics_compare(
        df_linkedin_clean,
        df_stepstone_clean,
        job_titles,
        report_folder,
        "03_vacancy_dynamics_compare.png",
        "Динамика вакансий по месяцам: LinkedIn vs Stepstone"
    )

    plot_skills_compare(
        hard_linkedin,
        soft_linkedin,
        hard_stepstone,
        soft_stepstone,
        scraping_date,
        report_folder,
        "04_skills_compare"
    )

    print("\n[OK] Полный отчёт LinkedIn vs Stepstone собран.")





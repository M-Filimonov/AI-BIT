# ------------------------------------------------------------
# 4. Top HARD-skills
# ------------------------------------------------------------

def top_hard_skills(df: pd.DataFrame, top_n=20):
    print(f"\n=== Arbeitsagentur: Top {top_n} HARD-skills ===")
    counter = Counter()
    for skills in df["hard_skills"].dropna():
        for s in skills.split(","):
            if s.strip():
                counter[s.strip()] += 1

    top = counter.most_common(top_n)
    for skill, count in top:
        print(f"{skill}: {count}")

    skills, counts = zip(*top)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=list(counts),
        y=list(skills),
        hue=list(skills),
        palette="Reds_r",
        legend=False
    )

    add_value_labels(ax)
    plt.title(f"Arbeitsagentur: Top {top_n} HARD-skills")
    plt.xlabel("Anzahl")
    plt.ylabel("Fähigkeit")
    save_plot("04_AA_top_hard_skills")


# ------------------------------------------------------------
# 5. Top SOFT-skills
# ------------------------------------------------------------

def top_soft_skills(df: pd.DataFrame, top_n=20):
    print(f"\n=== Arbeitsagentur: Top {top_n} SOFT-skills ===")
    counter = Counter()
    for skills in df["soft_skills"].dropna():
        for s in skills.split(","):
            if s.strip():
                counter[s.strip()] += 1

    top = counter.most_common(top_n)
    for skill, count in top:
        print(f"{skill}: {count}")

    skills, counts = zip(*top)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=list(counts),
        y=list(skills),
        hue=list(skills),
        palette="Purples_r",
        legend=False
    )

    add_value_labels(ax)
    plt.title(f"Arbeitsagentur: Top {top_n} SOFT-skills")
    plt.xlabel("Anzahl")
    plt.ylabel("Fähigkeit")
    save_plot("05_AA_top_soft_skills")

def search_term_distribution(df: pd.DataFrame):
    print("\n=== Arbeitsagentur: Verteilung der offenen Stellen nach Suchbegriffen ===")
    counts = df["search_term"].value_counts()
    print(counts)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette="Blues_d",
        legend=False
    )

    add_value_labels(ax)
    plt.title("Arbeitsagentur: Verteilung der offenen Stellen nach Suchbegriffen")
    plt.xlabel("Suchbegriff")
    plt.ylabel("Stellenanzahl")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    save_plot("02_AA_search_term_distribution")


def monthly_dynamics(df: pd.DataFrame):
    print("\n=== Arbeitsagentur: Dynamik der offenen Stellen im Monatsverlauf ===")
    df["month"] = df["posted_date"].dt.to_period("M")
    counts = df.groupby("month").size()
    print(counts)

    plt.figure(figsize=(10, 4))
    months = counts.index.to_timestamp()
    ax = sns.lineplot(x=months, y=counts.values, marker="o")
    plt.title("Arbeitsagentur: Dynamik im Monatsverlauf")
    plt.xlabel("Monat")
    plt.ylabel("Stellenanzahl")
    plt.xticks(rotation=45, ha="right", fontsize=8)

    for x, y in zip(months, counts.values):
        plt.text(x, y, str(y), fontsize=8, ha="center", va="bottom")

    save_plot("03_AA_monthly_dynamics")

def monthly_dynamics(df: pd.DataFrame):
    """
    Строит динамику количества вакансий по месяцам:
    1) общий график по всем вакансиям
    2) отдельные графики по уровням опыта (entry, advanced)
    """

    print("\n=== Arbeitsagentur: Dynamik der offenen Stellen im Monatsverlauf ===")

    # Универсальная подготовка
    df["month"] = df["posted_date"].dt.to_period("M")

    # ------------------------------------------------------------
    # 3.1. Динамика вакансий по месяцам - Общий график
    # ------------------------------------------------------------
    counts_all = df.groupby("month").size()
    print("\n--- Gesamt (alle Erfahrungsstufen) ---")
    print(counts_all)

    plt.figure(figsize=(10, 4))
    months = counts_all.index.to_timestamp()

    ax = sns.lineplot(
        x=months,
        y=counts_all.values,
        marker="o",
        color="steelblue"
    )

    plt.title("Arbeitsagentur: Dynamik im Monatsverlauf (Gesamt)")
    plt.xlabel("Monat")
    plt.ylabel("Stellenanzahl")
    plt.xticks(rotation=45, ha="right", fontsize=8)

    for x, y in zip(months, counts_all.values):
        plt.text(x, y, str(y), fontsize=8, ha="center", va="bottom")

    save_plot("03_AA_monthly_dynamics_all")

    # ------------------------------------------------------------
    # 3.2 Динамика вакансий по месяцам. Графики по уровням опыта
    # ------------------------------------------------------------
    exp_levels = df["experience_level"].dropna().unique()

    for level in exp_levels:
        subset = df[df["experience_level"] == level]

        if subset.empty:
            continue

        counts = subset.groupby("month").size()
        print(f"\n--- Erfahrungsstufe: {level.upper()} ---")
        print(counts)

        plt.figure(figsize=(10, 4))
        months = counts.index.to_timestamp()

        # Цвета для разных уровней
        color = "darkorange" if level == "entry" else "seagreen"

        ax = sns.lineplot(
            x=months,
            y=counts.values,
            marker="o",
            color=color
        )

        plt.title(f"Arbeitsagentur: Dynamik im Monatsverlauf ахк Erfahrungsniveau:({level.upper()})")
        plt.xlabel("Monat")
        plt.ylabel("Stellenanzahl")
        plt.xticks(rotation=45, ha="right", fontsize=8)

        for x, y in zip(months, counts.values):
            plt.text(x, y, str(y), fontsize=8, ha="center", va="bottom")

        save_plot(f"03_AA_monthly_dynamics_{level.upper()}")




def fetch_description_from_site(
    refnr: str
) -> Tuple[
    Optional[str], List[str], Dict[str, str], Optional[str], Optional[str], str, str
]:
    """
    Возвращает:
        html_full: структурированное описание (если есть)
        skills_found: навыки из HTML
        sections_dict: словарь секций
        sections_text: текст секций
        html_raw_text: fallback-текст
        html_quality: structured / unstructured / external / empty
        html_filename: имя файла в кэше
    """

    if not refnr:
        return None, [], {}, None, None, "empty", ""

    html_filename = get_html_cache_filename(refnr)

    # --------------------------------------------------------
    # 0) Загрузка HTML из кэша или BA
    # --------------------------------------------------------
    html = load_html_from_cache(refnr)
    if html is None:
        url = f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                return None, [], {}, None, None, "empty", html_filename
            html = r.text
            save_html_to_cache(refnr, html)
        except Exception:
            return None, [], {}, None, None, "empty", html_filename

    soup = BeautifulSoup(html, "html.parser")

    # --------------------------------------------------------
    # 1) Попытка извлечь секции по <h2>/<h3>
    # --------------------------------------------------------
    sections_dict = {}
    skills_found = set()
    full_blocks = []
    sections_text_blocks = []

    def is_section_title(text: str) -> bool:
        t = text.lower().strip().rstrip(":")
        return any(key in t for key in SECTION_TITLES)

    headers = soup.find_all(["h2", "h3"])
    for h in headers:
        title = h.get_text(strip=True)
        if not is_section_title(title):
            continue

        lines = []
        nxt = h.find_next_sibling()
        while nxt and nxt.name in {"p", "ul", "ol", "div"}:
            lines.append(nxt.get_text(separator="\n").strip())
            nxt = nxt.find_next_sibling()

        section_text = "\n".join(lines).strip()
        if section_text:
            sections_dict[title] = section_text
            full_blocks.append(f"{title}:\n{section_text}")
            sections_text_blocks.append(f"{title}:\n{section_text}")

    if sections_dict:
        html_full = "\n\n".join(full_blocks)
        sections_text = "\n\n".join(sections_text_blocks)
        hard, _, _ = extract_skills_and_level(html_full)
        skills_found.update(hard)
        html_raw_text = soup.get_text(separator="\n")
        return html_full, sorted(skills_found), sections_dict, sections_text, html_raw_text, "structured", html_filename

    # --------------------------------------------------------
    # 2) Попытка извлечь секции по <strong>
    # --------------------------------------------------------
    strongs = soup.find_all("strong")
    for s in strongs:
        title = s.get_text(strip=True)
        if not is_section_title(title):
            continue

        lines = []
        nxt = s.parent.find_next_sibling()
        while nxt and nxt.name in {"p", "ul", "ol", "div"}:
            lines.append(nxt.get_text(separator="\n").strip())
            nxt = nxt.find_next_sibling()

        section_text = "\n".join(lines).strip()
        if section_text:
            sections_dict[title] = section_text

    if sections_dict:
        full_blocks = [f"{k}:\n{v}" for k, v in sections_dict.items()]
        html_full = "\n\n".join(full_blocks)
        sections_text = html_full
        hard, _, _ = extract_skills_and_level(html_full)
        skills_found.update(hard)
        html_raw_text = soup.get_text(separator="\n")
        return html_full, sorted(skills_found), sections_dict, sections_text, html_raw_text, "structured", html_filename

    # --------------------------------------------------------
    # 3) Попытка извлечь секции по текстовым заголовкам с двоеточием
    # --------------------------------------------------------
    raw_text = soup.get_text(separator="\n")
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

    current_title = None
    buffer = []

    for line in lines:
        if line.endswith(":") and is_section_title(line):
            if current_title and buffer:
                sections_dict[current_title] = "\n".join(buffer).strip()
            current_title = line.rstrip(":")
            buffer = []
        else:
            if current_title:
                buffer.append(line)

    if current_title and buffer:
        sections_dict[current_title] = "\n".join(buffer).strip()

    if sections_dict:
        full_blocks = [f"{k}:\n{v}" for k, v in sections_dict.items()]
        html_full = "\n\n".join(full_blocks)
        sections_text = html_full
        hard, _, _ = extract_skills_and_level(html_full)
        skills_found.update(hard)
        return html_full, sorted(skills_found), sections_dict, sections_text, raw_text, "structured", html_filename

    # --------------------------------------------------------
    # 4) Попытка парсинга внешней страницы
    # --------------------------------------------------------
    external_url = extract_external_url(soup)
    if external_url:

        # # 4.0) Проверяем кэш внешних страниц
        # cached_ext = load_external_from_cache(external_url)
        # if cached_ext:
        #     soup_ext = BeautifulSoup(cached_ext, "html.parser")
        #     raw_text = soup_ext.get_text(separator="\n")
        #     hard, soft, level = extract_skills_and_level(raw_text)
        #     return None, sorted(hard), {}, None, raw_text, "external", html_filename
        #
        # # 4.1) Пробуем requests
        # try:
        #     r_ext = requests.get(external_url, timeout=10)
        #     if r_ext.status_code == 200 and len(r_ext.text) > 500:
        #         html_ext = r_ext.text
        #     else:
        #         html_ext = None
        # except Exception:
        #     html_ext = None
        #
        # # 4.2) Если requests не дал нормальный HTML → Selenium
        # if not html_ext:
        #     html_ext = fetch_external_html_selenium(external_url)
        #
        # # 4.3) Если получили HTML → сохраняем в кэш
        # if html_ext:
        #     save_external_to_cache(external_url, html_ext)
        #
        #     soup_ext = BeautifulSoup(html_ext, "html.parser")
        #     raw_text = soup_ext.get_text(separator="\n")
        #     hard, soft, level = extract_skills_and_level(raw_text)
        #
        #     return (
        #         None,
        #         sorted(hard),
        #         {},
        #         None,
        #         raw_text,
        #         "external",
        #         html_filename
        #     )
        external_url = extract_external_url(soup)
        if external_url:
            # 4.1) Проверяем кэш
            cached_ext = load_external_from_cache(external_url)
            if cached_ext:
                soup_ext = BeautifulSoup(cached_ext, "html.parser")
                raw_text = soup_ext.get_text(separator="\n")
                hard, soft, level = extract_skills_and_level(raw_text)
                return None, sorted(hard), {}, None, raw_text, "external", get_html_cache_filename(refnr)

            # 4.2) Пробуем requests
            try:
                r_ext = requests.get(external_url, timeout=10)
                if r_ext.status_code == 200 and len(r_ext.text) > 500:
                    html_ext = r_ext.text
                else:
                    html_ext = None
            except Exception:
                html_ext = None

            # 4.3) Если не сработало — пробуем Selenium
            if not html_ext:
                html_ext = fetch_external_html_selenium(external_url)

            # 4.4) Если получили HTML — сохраняем и возвращаем
            if html_ext:
                save_external_to_cache(external_url, html_ext)
                soup_ext = BeautifulSoup(html_ext, "html.parser")
                raw_text = soup_ext.get_text(separator="\n")
                hard, soft, level = extract_skills_and_level(raw_text)
                return None, sorted(hard), {}, None, raw_text, "external", get_html_cache_filename(refnr)

    # --------------------------------------------------------
    # 5) Fallback: весь текст страницы
    # --------------------------------------------------------
    html_raw_text = raw_text.strip()
    if html_raw_text:
        hard, _, _ = extract_skills_and_level(html_raw_text)
        skills_found.update(hard)
        return None, sorted(skills_found), {}, None, html_raw_text, "unstructured", html_filename

    # --------------------------------------------------------
    # 6) HTML пустой
    # --------------------------------------------------------
    return None, [], {}, None, None, "empty", html_filename





"""
aa_etl_patch.py

Рабочий патч: улучшенная многозадачность для ETL.
- Параллельный сбор (ThreadPoolExecutor) для aa_search (I/O-bound).
- Параллельная нормализация (ProcessPoolExecutor) для CPU-bound задач,
  с автоматическим откатом на ThreadPool, если normalize_aa_item не сериализуем.
- Улучшенный aa_search: requests.Session, retry с backoff и обработка 429/JSON.
- Безопасная запись Excel с fallback.

Интегрируй этот файл в проект и импортируй/замени run_etl и aa_search.
"""

from __future__ import annotations

import logging
import os
import time
import random
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import requests
import pandas as pd

# Импорты из проекта (предполагаются в других модулях)
# from aa_api_client import aa_search  # заменим локальной реализацией ниже
# from aa_normalize import normalize_aa_item
# from aa_utils import check_period, dedup
# from aa_config import SEARCH_TERMS, OUT_DIR, XLSX_PATH, SLEEP_BETWEEN, MAX_PAGES, PAGE_SIZE, DAYS_WINDOW, AA_BASE_URL, AA_HEADERS

# Для совместимости, если вы уже импортировали эти имена в глобальном пространстве,
# этот модуль будет их использовать. Иначе — определите их в aa_config.py как раньше.
try:
    from aa_config import (
        SEARCH_TERMS,
        OUT_DIR,
        XLSX_PATH,
        SLEEP_BETWEEN,
        MAX_PAGES,
        PAGE_SIZE,
        DAYS_WINDOW,
        AA_BASE_URL,
        AA_HEADERS,
    )
except Exception:
    # Безопасные дефолты, если aa_config не импортирован
    SEARCH_TERMS = ["Business Transformation Analyst"]
    OUT_DIR = Path("AA_output")
    XLSX_PATH = OUT_DIR / f"AA_jobs_de_{DAYS_WINDOW if 'DAYS_WINDOW' in globals() else 365}_days.xlsx"
    SLEEP_BETWEEN = 0.8
    MAX_PAGES = 100
    PAGE_SIZE = 100
    DAYS_WINDOW = 365
    AA_BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs"
    AA_HEADERS = {"X-API-Key": "jobboerse-jobsuche"}

# Функции проекта, ожидается, что они определены в других модулях.
# Если их нет — замените на реальные реализации.
try:
    from aa_normalize import normalize_aa_item
except Exception:
    def normalize_aa_item(item: Dict[str, Any], search_term: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # Заглушка: просто возвращает минимальную структуру
        return {
            "job_title": item.get("stellenbezeichnung") or item.get("title") or "",
            "company": item.get("arbeitgeber", {}).get("name") if isinstance(item.get("arbeitgeber"), dict) else item.get("arbeitgeber"),
            "posted_date": item.get("veroeffentlichungsdatum") or item.get("published"),
            "description_full": item.get("beschreibung") or item.get("description") or "",
            "source": "aa",
            "search_term": search_term,
            "raw": item
        }

try:
    from aa_utils import check_period, dedup
except Exception:
    def check_period(posted_date) -> bool:
        # Простая проверка: если есть дата — True
        return posted_date is not None

    def dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for r in rows:
            key = (r.get("job_title", ""), r.get("company", ""), str(r.get("posted_date", "")))
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out


# -------------------------
# Улучшенный aa_search
# -------------------------
def aa_search(query: str, max_pages: int = MAX_PAGES, size: int = PAGE_SIZE, published_since: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Выполняет поиск вакансий в API Arbeitsagentur по ключевому слову.
    Использует requests.Session, retry с экспоненциальным backoff и обработку 429.
    Возвращает список сырых элементов API.
    """
    rows: List[Dict[str, Any]] = []
    session = requests.Session()

    params = {
        "was": query,
        "wo": "Deutschland",
        "veroeffentlichtseit": int(published_since or DAYS_WINDOW),
        "size": int(size),
        "page": 1,
    }

    for page in range(1, int(max_pages) + 1):
        params["page"] = page
        delay = 1.0
        success = False
        r = None

        for attempt in range(1, 4):  # 3 attempts
            try:
                r = session.get(AA_BASE_URL, headers=AA_HEADERS, params=params, timeout=30)
                status = getattr(r, "status_code", None)
                if status == 200:
                    success = True
                    break
                if status == 429:
                    # Respect Retry-After if present
                    ra = r.headers.get("Retry-After")
                    try:
                        wait = int(ra)
                    except Exception:
                        wait = delay
                    jitter = random.uniform(0, 1)
                    logging.warning(f"[API 429] query='{query}' page={page} attempt={attempt} - waiting {wait + jitter:.1f}s")
                    time.sleep(wait + jitter)
                else:
                    logging.warning(f"[API] HTTP {status} for '{query}' page {page}, attempt {attempt}")
            except requests.RequestException as e:
                logging.warning(f"[API] RequestException for '{query}' page {page}, attempt {attempt}: {e}")
            except Exception as e:
                logging.error(f"[API] Unexpected error for '{query}' page {page}, attempt {attempt}: {e}")

            # exponential backoff with jitter
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= 2

        if not success:
            logging.error(f"[API] Пропуск страницы {page} для '{query}' после 3 попыток")
            break

        # безопасный парсинг JSON
        try:
            data = r.json()
        except ValueError:
            logging.error(f"[API] Некорректный JSON для '{query}' page {page}")
            break

        # поддержка разных ключей ответа
        items = data.get("stellenangebote") or data.get("jobs") or data.get("results") or []
        if not items:
            # если пустая страница — считаем, что дальше нет данных
            logging.info(f"[API] Пустая страница для '{query}' page {page}, прекращаем пагинацию")
            break

        rows.extend(items)
        logging.info(f"[API] query='{query}' page={page}: {len(items)} вакансий")

        # пауза между страницами чтобы не перегружать API
        time.sleep(SLEEP_BETWEEN)

    session.close()
    return rows


# -------------------------
# Вспомогательные обёртки для нормализации
# -------------------------
def _normalize_wrapper(item_search_pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Обёртка для normalize_aa_item, пригодная для ProcessPool/ThreadPool.
    item_search_pair: {"item": item, "search_term": term}
    """
    try:
        item = item_search_pair["item"]
        term = item_search_pair.get("search_term")
        return normalize_aa_item(item, search_term=term)
    except Exception as e:
        logging.exception("normalize_aa_item failed")
        return None


# -------------------------
# Основной run_etl (патч)
# -------------------------
def run_etl() -> None:
    """
    Основной процесс выгрузки вакансий с настоящей многозадачностью:
    1) Параллельно запускаем aa_search для всех SEARCH_TERMS (ThreadPoolExecutor).
    2) Параллельно нормализуем результаты (ProcessPoolExecutor, fallback на ThreadPool).
    3) Дедупликация, сортировка, запись в Excel.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Запуск ETL")

    # 1) Параллельный поиск по всем терминам (I/O-bound)
    search_entries: List[Dict[str, Any]] = []
    max_search_workers = min(16, max(2, len(SEARCH_TERMS)))
    with ThreadPoolExecutor(max_workers=max_search_workers) as tpool:
        future_to_term = {tpool.submit(aa_search, term): term for term in SEARCH_TERMS}
        for future in as_completed(future_to_term):
            term = future_to_term[future]
            try:
                items = future.result() or []
                logging.info(f"[SEARCH DONE] term='{term}' items={len(items)}")
            except Exception as e:
                logging.error(f"[SEARCH ERROR] term='{term}': {e}")
                items = []
            for it in items:
                search_entries.append({"item": it, "search_term": term})

    if not search_entries:
        logging.warning("Нет результатов поиска по всем терминам.")
        return

    # 2) Нормализация: пытаемся использовать ProcessPoolExecutor для CPU-bound задач.
    normalized_rows: List[Dict[str, Any]] = []

    # Проверяем, можно ли сериализовать normalize_aa_item (pickle)
    use_process_pool = True
    try:
        pickle.dumps(normalize_aa_item)
    except Exception:
        logging.warning("normalize_aa_item не сериализуем — используем ThreadPool для нормализации")
        use_process_pool = False

    if use_process_pool:
        try:
            max_workers = max(1, (os.cpu_count() or 2) - 1)
            with ProcessPoolExecutor(max_workers=max_workers) as ppool:
                futures = {ppool.submit(_normalize_wrapper, entry): entry for entry in search_entries}
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        row = future.result()
                    except Exception as e:
                        logging.error(f"[NORMALIZE ERROR] term='{entry.get('search_term')}' error: {e}")
                        row = None
                    if row and check_period(row.get("posted_date")):
                        normalized_rows.append(row)
        except Exception as e:
            logging.exception("ProcessPoolExecutor failed, falling back to ThreadPool for normalization")
            use_process_pool = False

    if not use_process_pool:
        # Fallback: ThreadPoolExecutor for normalization (safer but может быть медленнее для CPU-heavy)
        max_workers = min(16, max(2, (os.cpu_count() or 2)))
        with ThreadPoolExecutor(max_workers=max_workers) as tpool:
            futures = {tpool.submit(_normalize_wrapper, entry): entry for entry in search_entries}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    row = future.result()
                except Exception as e:
                    logging.error(f"[NORMALIZE ERROR] term='{entry.get('search_term')}' error: {e}")
                    row = None
                if row and check_period(row.get("posted_date")):
                    normalized_rows.append(row)

    # 3) Дедупликация, сортировка, запись
    all_rows = dedup(normalized_rows)
    logging.info(f"Итого уникальных вакансий: {len(all_rows)}")

    if not all_rows:
        logging.warning("Нет данных для сохранения.")
        return

    # Сортировка (reverse=True чтобы более свежие/приоритетные были первыми, при условии корректного posted_date)
    all_rows.sort(
        key=lambda r: (
            r.get("job_title") or "",
            r.get("company") or "",
            r.get("posted_date") or ""
        ),
        reverse=True
    )

    df = pd.DataFrame(all_rows)
    # Переименование поля description_full -> description если есть
    if "description_full" in df.columns and "description" not in df.columns:
        df = df.rename(columns={"description_full": "description"})

    # Сохранение с защитой
    try:
        df.to_excel(XLSX_PATH, index=False)
        logging.info(f"XLSX сохранён: {XLSX_PATH.resolve()}")
    except Exception as e:
        logging.exception(f"Ошибка при сохранении XLSX: {e}")
        fallback = OUT_DIR / f"AA_jobs_fallback_{int(time.time())}.csv"
        try:
            df.to_csv(fallback, index=False)
            logging.warning(f"Сохранено в CSV fallback: {fallback.resolve()}")
        except Exception as e2:
            logging.exception(f"Не удалось сохранить fallback CSV: {e2}")

    logging.info("ETL завершён.")


# -------------------------
# Если запускается как скрипт
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_etl()


def plot_top_skills_by_experience(df: pd.DataFrame,
                                  column: str,
                                  title_prefix: str,
                                  palette: str,
                                  top_n: int = 20):
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe.")
        return

    # Автоматически определяем уровни опыта
    exp_levels = sorted(df["experience_level"].dropna().unique())

    print(f"\nНайдены уровни опыта: {exp_levels}")

    for level in exp_levels:
        subset = df[df["experience_level"] == level]

        if subset.empty:
            continue

        print(f"\n=== Top {top_n} {title_prefix}-skills (experience: {level}) ===")

        counter = Counter()
        for skills in subset[column].dropna():
            for s in skills.split(","):
                s = s.strip()
                if s:
                    counter[s] += 1

        if not counter:
            print("Нет skills для этого уровня.")
            continue

        top = counter.most_common(top_n)

        for skill, count in top:
            print(f"{skill}: {count}")

        skills, counts = zip(*top)

        plt.figure(figsize=(10, 6))
        # palette может быть строкой (имя палитры) или уже списком цветов
        if isinstance(palette, str):
            pal = sns.color_palette(palette, n_colors=len(skills))
        else:
            # если передали список цветов, используем его (или приводим к нужной длине)
            pal = palette if len(palette) == len(skills) else sns.color_palette(palette, n_colors=len(skills))
        ax = sns.barplot(
            x=list(counts),
            y=list(skills),
            palette=pal
        )

        add_value_labels(ax)
        plt.title(f"Arbeitsagentur: Top {top_n} {title_prefix}-skills ({level})")
        plt.xlabel("Anzahl")
        plt.ylabel("Fähigkeit")

        if "hard" in title_prefix.lower():
            save_plot(f"04_AA_top_{title_prefix.lower()}_skills_{level}")
        else:
            save_plot(f"05_AA_top_{title_prefix.lower()}_skills_{level}")

"""
aa_main.py
-----------------
Главный модуль ETL‑пайплайна проекта.

Функции модуля:

1) Поиск вакансий (API Arbeitsagentur)
   - перебор SEARCH_TERMS
   - многопоточный сбор результатов через ThreadPoolExecutor

2) Нормализация данных
   - вызов normalize_aa_item()
   - извлечение описания, навыков, опыта (ML), локации, метаданных

3) Фильтрация по дате
   - оставляет только вакансии, опубликованные за последние DAYS_WINDOW дней

4) Дедупликация
   - удаление дублей по (job_title, company, location)

5) Сохранение результатов
   - сортировка
   - экспорт в XLSX (XLSX_PATH)

6) Аналитика
   - run_analytics(): сводные таблицы, графики, статистика
   - run_geo_analysis(): географический анализ вакансий

Основные точки входа:
- run_etl()              — полный процесс загрузки и нормализации
- run_with_analytics()   — запуск аналитики на готовом XLSX

"""


import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict

from aa_config import (
    SEARCH_TERMS,
    DAYS_WINDOW,
    OUT_DIR,
    XLSX_PATH,
    LOG_LEVEL,
    LOG_FORMAT,
)
from aa_api_client import aa_search
from aa_normalize import normalize_aa_item
from aa_analytic import run_analytics   # ← правильный импорт
from aa_location import run_geo_analysis


logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def check_period(date_iso: str) -> bool:
    """Проверяет, входит ли дата в последние DAYS_WINDOW дней."""
    if not date_iso:
        return False
    try:
        d = datetime.fromisoformat(date_iso).date()
        return d >= (datetime.today().date() - timedelta(days=DAYS_WINDOW))
    except Exception:
        return False


def dedup(rows: List[Dict]) -> List[Dict]:
    """Удаляет дубликаты по (job_title, company, location)."""
    seen = set()
    out = []
    for r in rows:
        key = (
            (r.get("job_title") or "").lower(),
            (r.get("company") or "").lower(),
            (r.get("location") or "").lower(),
        )
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def run_etl() -> None:
    """Основной процесс выгрузки вакансий."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = [] # Инициализация накопителя

    for term in SEARCH_TERMS: # Цикл по поисковым терминам
        logging.info(f"[TERM] {term}")

        with ThreadPoolExecutor(max_workers=8) as pool: # Параллельный вызов поиска

            # вызываем aa_search из aa_api_client.py и передаем запрос q из списка SEARCH_TERMS
            futures = {pool.submit(aa_search, q): q for q in [term]}

            for future in as_completed(futures): # Итерация по завершённым задачам с обработкой исключений
                q = futures[future]
                try:
                    items = future.result()
                except Exception as e:
                    logging.error(f"[THREAD ERROR] {q}: {e}")
                    continue

                for it in items:  # Нормализация и фильтрация по дате (преобразование сырого API‑объекта в унифицированную структуру)
                    row = normalize_aa_item(it, search_term=term)
                    if row and check_period(row["posted_date"]):
                        all_rows.append(row)

    all_rows = dedup(all_rows) # Дедупликация
    logging.info(f"Итого уникальных вакансий: {len(all_rows)}")

    if not all_rows: # Проверка наличия данных
        logging.warning("Нет данных для сохранения.")
        return

    # Сортировка
    all_rows.sort(
        key=lambda r: (
            r.get("job_title") or "",
            r.get("company") or "",
            r.get("posted_date") or ""
        ),
        reverse=True
    )

    # запись в Excel файл
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"description_full": "description"})
    df.to_excel(XLSX_PATH, index=False)

    logging.info("Готово.")
    logging.info(f"XLSX:  {XLSX_PATH.resolve()}")


def run_with_analytics():
    """Запуск ETL + аналитики."""
    run_etl()

    ANALYTICS_DIR = OUT_DIR / "analytics" # Папка для аналитики
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Запуск аналитики...")
    df = run_analytics(XLSX_PATH)
    run_geo_analysis(df)
    print(f"[INFO] Аналитика завершена. Результаты сохранены в {ANALYTICS_DIR.resolve()}")


if __name__ == "__main__":
    run_with_analytics()

"""
aa_api_client.py
-----------------

Клиент для обращения к API Arbeitsagentur (BA).
Выполняет поиск вакансий по ключевым словам с поддержкой:
- пагинации
- backoff при ошибках
- логирования
"""

import time
import logging
import requests
from typing import List, Dict, Any, Optional

from aa_config import (
    AA_BASE_URL,
    AA_HEADERS,
    PAGE_SIZE,
    MAX_PAGES,
    DAYS_WINDOW,
    SLEEP_BETWEEN,
)


def aa_search(query: str, max_pages: int = MAX_PAGES, size: int = PAGE_SIZE, published_since: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Выполняет поиск вакансий в API Arbeitsagentur по ключевому слову.

    Args:
        query (str): Поисковый запрос.
        max_pages (int): Максимальное количество страниц.
        size (int): Размер страницы.
        published_since (Optional[int]): Окно в днях для параметра veroeffentlichtseit.
            Если None — используется DAYS_WINDOW из конфигурации.

    Returns:
        List[Dict[str, Any]]: Список вакансий (сырые данные API).
    """
    rows: List[Dict[str, Any]] = []

    # используем значение из аргумента или DAYS_WINDOW из конфига
    window_days = int(published_since) if published_since is not None else int(DAYS_WINDOW)
    logging.info(f"[API] veroeffentlichtseit set to {window_days} days for query '{query}'")

    params = {
        "was": query,
        "wo": "Deutschland",
        "veroeffentlichtseit": window_days,
        "size": size,
        "page": 1,
    }

    for page in range(1, max_pages + 1):
        params["page"] = page

        delay = 1
        success = False

        for attempt in range(3):
            try:
                r = requests.get(AA_BASE_URL, headers=AA_HEADERS, params=params, timeout=30)
                if r.status_code == 200:
                    success = True
                    break
                else:
                    logging.warning(f"[API] HTTP {r.status_code} for '{query}' page {page}, retry {attempt+1}")
            except Exception as e:
                logging.error(f"[API] {e} for '{query}' page {page}, retry {attempt+1}")

            time.sleep(delay)
            delay *= 2

        if not success:
            logging.error(f"[API] Пропуск страницы {page} для '{query}' после 3 попыток")
            break

        try:
            data = r.json()
        except ValueError:
            logging.error(f"[API] Некорректный JSON для '{query}' page {page}")
            break

        items = data.get("stellenangebote") or data.get("jobs") or []

        if not items:
            break

        rows.extend(items)
        logging.info(f"[API] query='{query}' page={page}: {len(items)} вакансий")

        time.sleep(SLEEP_BETWEEN)

    return rows

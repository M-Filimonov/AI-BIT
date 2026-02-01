"""
aa_api_client.py
-----------------

Client for interacting with the Arbeitsagentur (BA) API.
Performs vacancy search with support for:
- pagination
- retry with exponential backoff
- logging
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
    Performs a vacancy search in the Arbeitsagentur API using a keyword.

    Args:
        query (str): Search query.
        max_pages (int): Maximum number of pages to fetch.
        size (int): Page size.
        published_since (Optional[int]): Number of days for the 'veroeffentlichtseit' filter.
            If None, the default DAYS_WINDOW from configuration is used.

    Returns:
        List[Dict[str, Any]]: List of vacancies (raw API data).
    """
    rows: List[Dict[str, Any]] = []

    # use the provided argument or fallback to DAYS_WINDOW from config
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

        # retry with exponential backoff
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
            logging.error(f"[API] Skipping page {page} for '{query}' after 3 failed attempts")
            break

        try:
            data = r.json()
        except ValueError:
            logging.error(f"[API] Invalid JSON for '{query}' page {page}")
            break

        # API sometimes returns 'stellenangebote' or 'jobs'
        items = data.get("stellenangebote") or data.get("jobs") or []

        if not items:
            break

        rows.extend(items)
        logging.info(f"[API] query='{query}' page={page}: {len(items)} vacancies")

        time.sleep(SLEEP_BETWEEN)

    return rows

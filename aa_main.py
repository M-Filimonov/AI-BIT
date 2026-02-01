"""
aa_main.py
-----------------
Main module of the project's ETL pipeline.

Module responsibilities:

1) Vacancy search (Arbeitsagentur API)
   - iterates over SEARCH_TERMS
   - collects results concurrently via ThreadPoolExecutor

2) Data normalization
   - calls normalize_aa_item()
   - extracts description, skills, experience (ML), location, metadata

3) Date filtering
   - keeps only vacancies published within the last DAYS_WINDOW days

4) Deduplication
   - removes duplicates by (job_title, company, location)

5) Saving results
   - sorting
   - exporting to XLSX (XLSX_PATH)

6) Analytics
   - run_analytics(): summary tables, charts, statistics
   - run_geo_analysis(): geographic vacancy analysis

Main entry points:
- run_etl()              — full loading and normalization process
- run_with_analytics()   — run analytics on the generated XLSX
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
from aa_analytic import run_analytics   # correct import
from aa_location import run_geo_analysis


logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def check_period(date_iso: str) -> bool:
    """Checks whether the date falls within the last DAYS_WINDOW days."""
    if not date_iso:
        return False
    try:
        d = datetime.fromisoformat(date_iso).date()
        return d >= (datetime.today().date() - timedelta(days=DAYS_WINDOW))
    except Exception:
        return False


def dedup(rows: List[Dict]) -> List[Dict]:
    """Removes duplicates by (job_title, company, location)."""
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
    """Main vacancy extraction and normalization process."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []  # accumulator

    for term in SEARCH_TERMS:  # iterate over search terms
        logging.info(f"[TERM] {term}")

        # parallel API calls
        with ThreadPoolExecutor(max_workers=8) as pool:

            # call aa_search from aa_api_client.py and pass q from SEARCH_TERMS
            futures = {pool.submit(aa_search, q): q for q in [term]}

            # iterate over completed tasks with exception handling
            for future in as_completed(futures):
                q = futures[future]
                try:
                    items = future.result()
                except Exception as e:
                    logging.error(f"[THREAD ERROR] {q}: {e}")
                    continue

                # normalization + date filtering (convert raw API object into unified structure)
                for it in items:
                    row = normalize_aa_item(it, search_term=term)
                    if row and check_period(row["posted_date"]):
                        all_rows.append(row)

    all_rows = dedup(all_rows)  # deduplication
    logging.info(f"Total unique vacancies: {len(all_rows)}")

    if not all_rows:  # no data check
        logging.warning("No data to save.")
        return

    # sorting
    all_rows.sort(
        key=lambda r: (
            r.get("job_title") or "",
            r.get("company") or "",
            r.get("posted_date") or ""
        ),
        reverse=True
    )

    # write to Excel
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"description_full": "description"})
    df.to_excel(XLSX_PATH, index=False)

    logging.info("Done.")
    logging.info(f"XLSX:  {XLSX_PATH.resolve()}")


def run_with_analytics():
    """Runs ETL and then analytics."""
    run_etl()

    ANALYTICS_DIR = OUT_DIR / "analytics"  # analytics output folder
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Running analytics...")
    df = run_analytics(XLSX_PATH)
    run_geo_analysis(df)
    print(f"[INFO] Analytics completed. Results saved to {ANALYTICS_DIR.resolve()}")


if __name__ == "__main__":
    run_with_analytics()


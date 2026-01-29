"""
aa_html_parser.py
---------------------
HTML‑парсер для страниц Arbeitsagentur.
- Извлечение секций по <h2>/<h3>
- Извлечение секций по <strong>
- Извлечение секций по текстовым заголовкам с двоеточием
- Игнорирование мусора до первой секции
- Fallback: весь текст страницы
- Возврат структурированных секций + fallback
"""

import re

import time
import requests
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup

from aa_config import HTML_CACHE_DIR
from aa_skills import extract_skills_and_level


SECTION_TITLES = [
    "stellenbeschreibung",
    "ihre aufgaben",
    "ihr profil",
    "wir bieten",
    "über uns",
    "das bringen sie mit",
    "das erwartet sie",
    "aufgaben",
    "profil",
]


# ------------------------------------------------------------
# Кэш
# ------------------------------------------------------------

def get_html_cache_filename(refnr: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9\-]", "_", refnr)
    return f"{safe}.html"


def _cache_path(refnr: str):
    return HTML_CACHE_DIR / get_html_cache_filename(refnr)


def load_html_from_cache(refnr: str) -> Optional[str]:
    path = _cache_path(refnr)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    return None


def save_html_to_cache(refnr: str, html: str) -> None:
    try:
        _cache_path(refnr).write_text(html, encoding="utf-8", errors="ignore")
    except Exception:
        pass

from aa_config import HTML_CACHE_DIR, EXTERNAL_CACHE_DIR

def get_external_cache_filename(url: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9\-]", "_", url)
    return f"{safe}.html"

def external_cache_path(url: str):
    return EXTERNAL_CACHE_DIR / get_external_cache_filename(url)

def load_external_from_cache(url: str) -> Optional[str]:
    path = external_cache_path(url)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    return None

def save_external_to_cache(url: str, html: str) -> None:
    try:
        external_cache_path(url).write_text(html, encoding="utf-8", errors="ignore")
    except Exception:
        pass


# ------------------------------------------------------------
# Основной парсер
# ------------------------------------------------------------

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
        html_filename: имя файла в кэше (BA или партнёр)
    """

    if not refnr:
        return None, [], {}, None, None, "empty", ""

    # имя BA-файла
    html_filename_ba = get_html_cache_filename(refnr)

    # --------------------------------------------------------
    # 0) Загрузка HTML BA из кэша или с сайта
    # --------------------------------------------------------
    html = load_html_from_cache(refnr)
    if html is None:
        url = f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                return None, [], {}, None, None, "empty", html_filename_ba
            html = r.text
            save_html_to_cache(refnr, html)
        except Exception:
            return None, [], {}, None, None, "empty", html_filename_ba

    soup = BeautifulSoup(html, "html.parser")

    # --------------------------------------------------------
    # 1) PRIORITY: если есть внешняя ссылка BA → ВСЕГДА парсим партнёра
    # --------------------------------------------------------
    external_btn = soup.find("a", id="detail-beschreibung-externe-url-btn")
    if external_btn and external_btn.has_attr("href"):
        external_url = external_btn["href"].strip()

        # имя файла партнёра
        external_filename = get_external_cache_filename(external_url)

        # 1.1) кэш
        cached_ext = load_external_from_cache(external_url)
        if cached_ext:
            soup_ext = BeautifulSoup(cached_ext, "html.parser")
            raw_text = soup_ext.get_text(separator="\n")
            hard, soft, level = extract_skills_and_level(raw_text)
            return None, sorted(hard), {}, None, raw_text, "external", external_filename

        # 1.2) requests
        try:
            r_ext = requests.get(external_url, timeout=10)
            if r_ext.status_code == 200 and len(r_ext.text) > 500:
                html_ext = r_ext.text
            else:
                html_ext = None
        except Exception:
            html_ext = None

        # 1.3) Selenium fallback
        if not html_ext:
            html_ext = fetch_external_html_selenium(external_url)

        # 1.4) если получили HTML → сохраняем и возвращаем
        if html_ext:
            save_external_to_cache(external_url, html_ext)
            soup_ext = BeautifulSoup(html_ext, "html.parser")
            raw_text = soup_ext.get_text(separator="\n")
            hard, soft, level = extract_skills_and_level(raw_text)
            return None, sorted(hard), {}, None, raw_text, "external", external_filename

    # --------------------------------------------------------
    # 2) Попытка извлечь секции по <h2>/<h3>
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
        return html_full, sorted(skills_found), sections_dict, sections_text, html_raw_text, "structured", html_filename_ba

    # --------------------------------------------------------
    # 3) Попытка извлечь секции по <strong>
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
        html_full = "\n\n".join(f"{k}:\n{v}" for k, v in sections_dict.items())
        sections_text = html_full
        hard, _, _ = extract_skills_and_level(html_full)
        skills_found.update(hard)
        html_raw_text = soup.get_text(separator="\n")
        return html_full, sorted(skills_found), sections_dict, sections_text, html_raw_text, "structured", html_filename_ba

    # --------------------------------------------------------
    # 4) Попытка извлечь секции по заголовкам с двоеточием
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
        html_full = "\n\n".join(f"{k}:\n{v}" for k, v in sections_dict.items())
        sections_text = html_full
        hard, _, _ = extract_skills_and_level(html_full)
        skills_found.update(hard)
        return html_full, sorted(skills_found), sections_dict, sections_text, raw_text, "structured", html_filename_ba

    # --------------------------------------------------------
    # 5) Fallback: весь текст страницы
    # --------------------------------------------------------
    html_raw_text = raw_text.strip()
    if html_raw_text:
        hard, _, _ = extract_skills_and_level(html_raw_text)
        skills_found.update(hard)
        return None, sorted(skills_found), {}, None, html_raw_text, "unstructured", html_filename_ba

    # --------------------------------------------------------
    # 6) HTML пустой
    # --------------------------------------------------------
    return None, [], {}, None, None, "empty", html_filename_ba



    # --------------------------------------------------------
    # fetch_external_html_selenium
    # --------------------------------------------------------
def fetch_external_html_selenium(url: str, wait_time: int = 5) -> Optional[str]:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager

    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        chrome_options.add_argument("--disable-features=IsolateOrigins,site-per-process")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-background-networking")
        chrome_options.add_argument("--disable-default-apps")
        chrome_options.add_argument("--disable-sync")
        chrome_options.add_argument("--disable-translate")
        chrome_options.add_argument("--metrics-recording-only")
        chrome_options.add_argument("--mute-audio")

        chrome_prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2,
            "profile.managed_default_content_settings.stylesheets": 1,
        }
        chrome_options.add_experimental_option("prefs", chrome_prefs)

        driver = webdriver.Chrome(
            ChromeDriverManager(cache_valid_range=30).install(),
            options=chrome_options
        )

        driver.get(url)

        # Быстрый клик по cookies
        cookie_selectors = [
            "button#onetrust-accept-btn-handler",
            "button[aria-label='Accept cookies']",
            "button[title='Akzeptieren']",
            "button.cookie-accept",
        ]
        for sel in cookie_selectors:
            try:
                driver.find_element(By.CSS_SELECTOR, sel).click()
                break
            except:
                pass

        # Ждём загрузки DOM
        WebDriverWait(driver, 3).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        html = driver.page_source
        driver.quit()
        return html

    except Exception:
        return None


"""
aa_normalize.py
-----------------
Normalization module for Arbeitsagentur (BA) vacancy data.

This module merges API fields with HTML‑parsed content, cleans and structures
text fields, extracts skills, classifies experience level using multiple
heuristics, and parses location metadata.

Core responsibilities:

1. Data merging
   - combines API vacancy fields with HTML description blocks
   - unifies raw text, structured sections, and fallback content

2. Text processing
   - cleans and normalizes description text
   - extracts structured sections (JSON + plain text)
   - determines description source (API / Website / Combined)

3. Skill extraction
   - identifies hard and soft skills
   - extracts experience signals from text

4. Experience classification
   - title‑based heuristics
   - regex‑based detection
   - semantic scoring model
   - unified mapping to entry / advanced classes

5. Location parsing
   - extracts raw location fields
   - parses index, city, state, country, address

6. Output structure
   Produces a fully normalized vacancy record with fields:
   - job_title, beruf, company, job_url, posted_date, search_term
   - description_full, description_sections_json, description_sections_text
   - hard_skills, soft_skills, work_experience, experience_level
   - location, index, city, state, country, address
   - salary, contract_type
   - html_filename, html_quality, source, normalized_at

Main entry point:
- normalize_aa_item() — full normalization pipeline for a single vacancy
"""

import json
import re
from typing import Dict, Any, Optional
from datetime import datetime
from dateutil import parser as dateparser

from aa_skills import extract_skills_and_level
from aa_html_parser import fetch_description_from_site
from aa_utils import parse_location_field
from aa_config import (
    BAD_LOG_PATH,
    EXPERIENCE_LEVEL_PATTERNS,
    EXPERIENCE_LEVEL_ORDER,
    LEVEL_TO_CLASS,
    TITLE_ENTRY_KEYWORDS,
    TITLE_ADVANCED_KEYWORDS,
)


# ------------------------------------------------------------
# NORMALIZATION HELPERS
# ------------------------------------------------------------

def extract_location(loc: Any) -> str:
    """Extracts a unified location string from various API formats."""
    if loc is None:
        return ""
    if isinstance(loc, str):
        return loc.strip()
    if isinstance(loc, dict):
        parts = []
        for key in ("plz", "ort", "ortsteil", "region", "land", "strasse"):
            val = loc.get(key)
            if isinstance(val, str):
                parts.append(val.strip())
        return ", ".join(parts)
    if isinstance(loc, list):
        if loc:
            return extract_location(loc[0])
        return ""
    return str(loc).strip()


# ------------------------------------------------------------
# EXPERIENCE LEVEL NORMALIZATION
# ------------------------------------------------------------

def normalize_level_name(level: str) -> str:
    """Maps various level names to unified categories."""
    if not isinstance(level, str):
        return ""
    t = level.strip().lower()
    if not t:
        return ""
    if t in {"entry", "junior"}:
        return "Junior"
    if t in {"mid", "middle", "professional"}:
        return "Mid"
    if t in {"senior", "expert", "experte"}:
        return "Senior"
    if t in {"lead", "principal", "manager"}:
        return "Lead"
    return ""


def detect_experience_level_from_title(title: str) -> Optional[str]:
    """Detects experience level based on job_title keywords."""
    if not isinstance(title, str) or not title.strip():
        return None

    t = title.lower()

    # Entry
    for kw in TITLE_ENTRY_KEYWORDS:
        if kw in t:
            return "Junior"

    # Advanced
    for kw in TITLE_ADVANCED_KEYWORDS:
        if kw in t:
            if any(x in t for x in ["lead", "teamleiter", "head of", "principal", "manager", "managing", "director"]):
                return "Lead"
            return "Senior"

    return None


def detect_experience_level_from_text(text: str) -> Optional[str]:
    """Detects experience level using regex patterns in text."""
    if not isinstance(text, str) or not text.strip():
        return None

    t = text.lower()
    matched_levels = set()

    for pattern, level in EXPERIENCE_LEVEL_PATTERNS.items():
        if re.search(pattern, t, flags=re.IGNORECASE):
            normalized = normalize_level_name(level)
            if normalized:
                matched_levels.add(normalized)

    if not matched_levels:
        return None

    for lvl in reversed(EXPERIENCE_LEVEL_ORDER):
        if lvl in matched_levels:
            return lvl

    return None


def classify_experience(text: str) -> str:
    """
    Semantic classification of experience level.
    Returns: junior / mid / senior / lead
    """
    if not isinstance(text, str) or not text.strip():
        return "mid"

    text_lower = text.lower()

    # 1. Regex
    for pattern, level in EXPERIENCE_LEVEL_PATTERNS.items():
        if re.search(pattern, text_lower):
            return level.lower()

    # 2. Semantic scoring
    semantic_signals = {
        "lead": {
            "qualifizierung": 2,
            "schulung": 2,
            "mitarbeiter": 3,
            "teamfähigkeit": 1,
            "einsatzplanung": 3,
            "kpi": 3,
            "qualitätssicherung": 2,
            "qualitätsmanagement": 3,
            "sonderentscheidungen": 3,
            "verantwortung": 3,
            "koordination": 2,
            "führung": 3,
            "leitung": 3,
            "teamleitung": 4,
            "steuerung": 2,
        },
        "senior": {
            "vertiefte kenntnisse": 3,
            "sehr gute kenntnisse": 3,
            "prozesse": 1,
            "anwendungen": 1,
            "fachanweisungen": 2,
            "mehrjährige erfahrung": 3,
            "expertise": 3,
            "selbstständig": 2,
        },
        "mid": {
            "kenntnisse": 1,
            "erfahrung": 1,
            "prozesse": 1,
            "aufgaben": 1,
            "verantwortlich": 1,
        },
        "junior": {
            "grundkenntnisse": 2,
            "erste erfahrungen": 2,
            "einsteiger": 3,
            "ausbildung": 1,
            "unterstützung": 1,
        }
    }

    score = {"lead": 0, "senior": 0, "mid": 0, "junior": 0}

    for level, keywords in semantic_signals.items():
        for kw, weight in keywords.items():
            if kw in text_lower:
                score[level] += weight

    # Context boosts
    if any(x in text_lower for x in ["team führen", "führung", "leitung", "mitarbeiter führen"]):
        score["lead"] += 5

    if "verantwortung für" in text_lower:
        score["lead"] += 3
        score["senior"] += 2

    if "mehrjährige erfahrung" in text_lower:
        score["senior"] += 4

    if "ausbildung" in text_lower and score["lead"] == 0 and score["senior"] == 0:
        score["junior"] += 2

    if all(v == 0 for v in score.values()):
        return "mid"

    return max(score, key=score.get)


# ------------------------------------------------------------
# MAIN NORMALIZATION FUNCTION
# ------------------------------------------------------------

def normalize_aa_item(item: Dict[str, Any], search_term: str) -> Optional[Dict[str, Any]]:
    """Full normalization pipeline for a single BA vacancy."""
    try:
        job_title = (item.get("titel") or "").strip()
        beruf = (item.get("beruf") or item.get("berufsbezeichnung") or "").strip()
        company = (item.get("arbeitgeber") or "").strip()

        # --- RAW LOCATION ---
        location_raw = extract_location(item.get("arbeitsort"))

        parsed_loc = parse_location_field(location_raw)
        loc_index = parsed_loc["index"]
        loc_city = parsed_loc["city"]
        loc_state = parsed_loc["state"]
        loc_country = parsed_loc["country"]
        loc_address = parsed_loc["address"]

        raw_date = (
            item.get("aktuelleVeroeffentlichungsdatum")
            or item.get("veroeffentlichtAm")
            or item.get("modifikationsTimestamp")
        )
        posted = None
        if raw_date:
            try:
                posted = dateparser.parse(raw_date).date()
            except Exception:
                posted = None

        refnr = item.get("refnr")
        job_url = f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}" if refnr else ""

        (
            html_full,
            html_skills,
            sections_dict,
            sections_text,
            html_raw_text,
            html_quality,
            html_filename
        ) = fetch_description_from_site(refnr)

        if html_full:
            description_full = html_full
        elif html_raw_text:
            description_full = html_raw_text
        else:
            description_full = ""

        if beruf and (html_full or html_raw_text):
            description_source = "API+Website"
        elif beruf:
            description_source = "API"
        elif html_full or html_raw_text:
            description_source = "Website"
        else:
            description_source = "None"

        if sections_dict:
            try:
                description_sections_json = json.dumps(sections_dict, ensure_ascii=False)
            except Exception:
                description_sections_json = ""
        else:
            description_sections_json = ""

        description_sections_text = sections_text or ""

        # ------------------------------------------------------------
        # SKILLS
        # ------------------------------------------------------------
        if html_quality == "external":
            hard_html = html_skills
            _, soft_html, lvl_html = extract_skills_and_level(description_full)
        else:
            hard_html, soft_html, lvl_html = extract_skills_and_level(description_full)

        hard_skills = ", ".join(sorted(set(hard_html)))
        soft_skills = ", ".join(sorted(set(soft_html)))

        experience_level_html = normalize_level_name(lvl_html)

        # ------------------------------------------------------------
        # EXPERIENCE HEURISTICS
        # ------------------------------------------------------------
        text_for_exp = description_sections_text or description_full

        level_from_title = detect_experience_level_from_title(job_title)

        level_from_semantic = None
        if not level_from_title and text_for_exp:
            level_from_semantic = classify_experience(text_for_exp)

        level_from_text = None
        if not level_from_title and not level_from_semantic and text_for_exp:
            level_from_text = detect_experience_level_from_text(text_for_exp)

        if level_from_title:
            experience_level_raw = normalize_level_name(level_from_title)
        elif level_from_semantic:
            experience_level_raw = normalize_level_name(level_from_semantic)
        elif level_from_text:
            experience_level_raw = normalize_level_name(level_from_text)
        elif experience_level_html:
            experience_level_raw = experience_level_html
        else:
            experience_level_raw = "Mid"

        experience_level = LEVEL_TO_CLASS.get(experience_level_raw, "advanced")

        # ------------------------------------------------------------
        # RETURN RESULT
        # ------------------------------------------------------------
        return {
            "search_term": search_term,
            "job_title": job_title,
            "beruf": beruf,
            "job_url": job_url,
            "company": company,
            "posted_date": posted.isoformat() if posted else None,

            "location": location_raw,
            "index": loc_index,
            "city": loc_city,
            "state": loc_state,
            "country": loc_country,
            "address": loc_address,

            "description_full": description_full,
            "description_sections_json": description_sections_json,
            "description_sections_text": description_sections_text,
            "description_source": description_source,

            "html_quality": html_quality,
            "html_filename": html_filename,

            "salary": item.get("entgelt") or item.get("salary"),
            "contract_type": item.get("befristung") or item.get("angebotsart"),

            "hard_skills": hard_skills,
            "soft_skills": soft_skills,

            "work_experience": level_from_title,
            "experience_level": experience_level,

            "source": "Arbeitsagentur",
            "normalized_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        with open(BAD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"error": str(e), "item": item}, ensure_ascii=False) + "\n")
        return None

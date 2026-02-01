"""
aa_skills.py
-----------------
Functions for extracting hard/soft skills and experience level from text.
Uses skill dictionaries defined in aa_config.py.
Also includes a helper for cleaning HTML tags from text.
"""

import re
from typing import List, Tuple, Optional
from bs4 import BeautifulSoup

from aa_config import HARD_SKILLS_DICT, SOFT_SKILLS_DICT, EXPERIENCE_LEVEL_PATTERNS


def extract_skills_and_level(text: str) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Extracts hard skills, soft skills, and experience level from text.
    GUARANTEES returning a tuple: (hard_list, soft_list, level_str_or_None).

    Args:
        text (str): Input text from vacancy description.

    Returns:
        Tuple[List[str], List[str], Optional[str]]:
            - hard skills (sorted list)
            - soft skills (sorted list)
            - experience level (string or None)
    """
    # Full protection against None, numbers, lists, etc.
    if not isinstance(text, str) or not text.strip():
        return [], [], None

    txt = text.lower()
    hard, soft = set(), set()
    level = None

    # -----------------------------
    # Hard skills
    # -----------------------------
    for pattern, name in HARD_SKILLS_DICT.items():
        try:
            if re.search(pattern, txt):
                hard.add(name)
        except re.error:
            # skip invalid regex patterns
            continue

    # -----------------------------
    # Soft skills
    # -----------------------------
    for pattern, name in SOFT_SKILLS_DICT.items():
        try:
            if re.search(pattern, txt):
                soft.add(name)
        except re.error:
            continue

    # -----------------------------
    # Experience level (regex)
    # -----------------------------
    for pattern, name in EXPERIENCE_LEVEL_PATTERNS.items():
        try:
            if re.search(pattern, txt):
                level = name
        except re.error:
            continue

    # -----------------------------
    # Final tuple normalization
    # -----------------------------
    hard_list = sorted(hard)
    soft_list = sorted(soft)

    # level is always either a string or None
    if level is not None:
        level = str(level).lower().strip()

    return hard_list, soft_list, level

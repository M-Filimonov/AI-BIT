"""
aa_skills.py
-----------------
Функции для извлечения навыков (hard/soft skills) и уровня опыта из текста.
Использует словари из aa_config.py.
Также содержит функцию очистки HTML-тегов из текста.
"""

import re
from typing import List, Tuple, Optional
from bs4 import BeautifulSoup

from aa_config import HARD_SKILLS_DICT, SOFT_SKILLS_DICT, EXPERIENCE_LEVEL_PATTERNS



def extract_skills_and_level(text: str) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Извлекает hard skills, soft skills и уровень опыта из текста.
    ГАРАНТИРУЕТ возврат tuple (hard_list, soft_list, level_str_or_None).
    """
    # Полная защита от None, чисел, списков и т.п.
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
            # если паттерн некорректный — пропускаем
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
    # Гарантируем корректный tuple
    # -----------------------------
    hard_list = sorted(hard)
    soft_list = sorted(soft)

    # level всегда либо строка, либо None
    if level is not None:
        level = str(level).lower().strip()

    return hard_list, soft_list, level

"""
aa_config.py
-----------------
Глобальная конфигурация проекта:
- Константы (пути, параметры API, настройки выгрузки)
- Настройки логирования
- Словари навыков (hard skills, soft skills, experience patterns)
"""

from pathlib import Path
from typing import Dict, List


# ============================================================
# Основные параметры выгрузки
# ============================================================

# Поисковые титлы
SEARCH_TERMS: List[str] = [
    "Business Transformation Analyst",
    "Digital Process Analyst",
    "AI Governance Analyst",
    "AI Automation Specialist",
    "Prompt Engineer",
    "Junior Automation Specialist",
    "AI Project Manager",
    "AI Product Manager",
    "Prozessmanager/in",
    "KI-Manager/in",
    "Prozessmanager/in – RPA",
    "KI-Prompter",
    "Prozessmanager/in – RPA (Junior)",
]

# SEARCH_TERMS = [
#     # ТОП‑5 для AI Business Automation Specialist
#     "AI Engineer",
#     "Data & AI Engineer",
#     "AI Automation Engineer",
#     "Hyperautomation Engineer",
#     "Business Process Automation Specialist",
#
#     # Дополнительные 3
#     "AI Solutions Architect",
#     "Process Mining Engineer",
#     "AI Integration Engineer"
# ]



DAYS_WINDOW: int = 365 # окно вакансий по времени в днях от даты скрепинга
PAGE_SIZE: int = 100
MAX_PAGES: int = 100
SLEEP_BETWEEN: float = 0.8


# ============================================================
# Пути
# ============================================================

OUT_DIR: Path = Path("AA_output")
HTML_CACHE_DIR: Path = Path("html_cache")

OUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

XLSX_PATH: Path = OUT_DIR / f"AA_jobs_de_{DAYS_WINDOW}_days.xlsx"
BAD_LOG_PATH: Path = OUT_DIR / "bad_items.log"


# ============================================================
# API параметры
# ============================================================

AA_BASE_URL: str = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs"
AA_HEADERS: Dict[str, str] = {
    "X-API-Key": "jobboerse-jobsuche"
}


# ============================================================
# Настройки логирования
# ============================================================

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"


# ============================================================
# Навыки (регулярные выражения → нормализованные названия)
# ============================================================

HARD_SKILLS_DICT: Dict[str, str] = {
    r"\bpython\b": "Python",
    r"\btensorflow\b": "TensorFlow",
    r"\b(pytorch|torch)\b": "PyTorch",
    r"\bscikit-?learn\b": "scikit-learn",
    r"\bkeras\b": "Keras",
    r"\bmlops\b": "MLOps",
    r"\bazure\b": "Azure",
    r"\baws\b": "AWS",
    r"\bgcp\b": "GCP",
    r"\b(kubernetes|k8s)\b": "Kubernetes",
    r"\bdocker\b": "Docker",
    r"\bsql\b": "SQL",
    r"\b(no ?sql|nosql)\b": "NoSQL",
    r"\bspark\b": "Spark",
    r"\bnlp\b": "NLP",
    r"\b(llm|large language model)\b": "LLM",
    r"\buipath\b": "UiPath",
    r"\bautomation anywhere\b": "Automation Anywhere",
    r"\bblue prism\b": "Blue Prism",
    r"\brpa\b": "RPA",
    r"\bprocess mining\b": "Process Mining",
    r"\bcelonis\b": "Celonis",
    r"\bbpmn\b": "BPMN",
    r"\bsix sigma\b": "Six Sigma",
    r"\blean\b": "Lean",
    r"\bjira\b": "Jira",
    r"\bconfluence\b": "Confluence",
    r"\bscrum\b": "Scrum",
    r"\bkanban\b": "Kanban",
    r"\bexcel\b": "Excel",
    r"\bpower ?bi\b": "Power BI",
    r"\bsap\b": "SAP",
    r"\bagile\b": "Agile",
    r"\bchange management\b": "Change Management",
    r"\bdigitalisierung\b": "Digitalisierung",
    r"\bstrategie\b": "Strategie",
    r"\bconsulting\b": "Consulting",
    r"\blow-?code\b": "Low-Code",
    r"\bno-?code\b": "No-Code",
    r"\bmake\.com\b": "Make.com",
    r"\bn8n\b": "n8n",
}


SOFT_SKILLS_DICT: Dict[str, str] = {
    r"\bcommunication\b|\bkommunikation\b": "Communication",
    r"\bteamwork\b|\bteamarbeit\b": "Teamwork",
    r"\bleadership\b|\bführung\b|\bfuehrung\b": "Leadership",
    r"\bproblem-?solving\b|\bproblemlösung\b": "Problem Solving",
    r"\bownership\b|\bverantwortung\b": "Ownership",
    r"\bstakeholder\b": "Stakeholder Management",
    r"\bcollaboration\b|\bzusammenarbeit\b": "Collaboration",
    r"\banalytical\b|\banalytisch\b": "Analytical Thinking",
    r"\bself-?organized\b|\bselbstständig\b": "Self-Organization",
}


EXPERIENCE_LEVEL_PATTERNS: Dict[str, str] = {
    r"\bjunior\b|\beinsteiger\b": "Junior",
    r"\bmid(dle)?\b|\bprofessional\b": "Mid",
    r"\bsenior\b|\bexperte\b|\bexpert\b": "Senior",
    r"\blead\b|\bprincipal\b|\bmanager\b": "Lead",
}

# ------------------------------------------------------------
# География: извлечение города и земли
# ------------------------------------------------------------

GERMAN_STATES = {
    "baden-württemberg": "Baden-Württemberg",
    "bayern": "Bayern",
    "berlin": "Berlin",
    "brandenburg": "Brandenburg",
    "bremen": "Bremen",
    "hamburg": "Hamburg",
    "hessen": "Hessen",
    "mecklenburg-vorpommern": "Mecklenburg-Vorpommern",
    "niedersachsen": "Niedersachsen",
    "nordrhein-westfalen": "Nordrhein-Westfalen",
    "rheinland-pfalz": "Rheinland-Pfalz",
    "saarland": "Saarland",
    "sachsen": "Sachsen",
    "sachsen-anhalt": "Sachsen-Anhalt",
    "schleswig-holstein": "Schleswig-Holstein",
    "thüringen": "Thüringen",
}


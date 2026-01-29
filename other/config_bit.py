from pathlib import Path
from datetime import datetime, timedelta

print("config_bit.py loaded")

##########################################################
# 0. Конфигурация 
##########################################################
# Константа: дата скрапинга
SCRAPING_DATE = datetime(2025, 12, 15)

# Источник данных: "LinkedIn" или "Stepstone"
#SCRAPING_SYST_NAME = "LinkedIn"
SCRAPING_SYST_NAME = "Stepstone"

# папки для разных типов файлов
DATA_FOLDER = Path("data")       # Папка с  исходными данными из скрапера в формате Excel 
REPORT_FOLDER = Path("report")   # Папка отчетов
RESULT_FOLDER = Path("result")   # Промежуточные результаты объединения, очистки, обогащения и дедубликации

# удаления дубликатов, с разной логикой
DEDUPLICATION_MODE = "canonical"  # допустимые значения: "simple", "canonical"
'''
simple - удаление дубликатов по отчёту по безопасным колонкам без приоритета выбора
canonical - удаление дубликатов с выбором лучшей строки из группы по строковой сигнатуре. 
Приоритет выбора: наличие job_url, salary, posted_at и большая длина description 
'''

KEEP = "first" # допустимые значения: "first","last" при удалении дубликатов оставляет превую или последнюю строку

##########################################################
# 1. Указание констант: Словари и Списки  
##########################################################
# список вакансий
JOB_TITLE = [
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
    "Prozessmanager/in – RPA (Junior)"    
]

# термины для классификации вакансий AI Business Automation Specialist по направлениям
AI_SPETIALIST_TERMS = {
    # 1. Consulting & Business Strategy
    "Consulting & Business Strategy": [
        # English terms
        "business transformation", "process thinking", "bpmn", "sipoc",
        "value chain", "theory of constraints", "bottleneck analysis",
        "impact-effort matrix", "risk-regret matrix", "consulting frameworks",
        "ai readiness", "strategic analysis", "stakeholder management",
        # German terms
        "geschäftstransformation", "prozessdenken", "wertschöpfungskette",
        "engpasstheorie", "engpassanalyse", "impact-effort-matrix",
        "risk-regret-matrix", "beratungsframeworks", "ai-reifegrad",
        "strategische analyse", "stakeholder-management"
    ],

    # 2. Data Management & Compliance (EU Focus)
    "Data Management & Compliance (EU Focus)": [
        # English terms
        "gdpr", "data protection", "ai act", "risk classification",
        "high-risk systems", "dpia", "technical documentation",
        "data governance", "data lake", "data warehouse", "data mart",
        "etl", "elt", "data quality", "anonymization", "pseudonymization",
        "hashing", "synthetic data", "responsible ai", "fairness",
        "accountability", "transparency", "bias management",
        # German terms
        "datenschutz", "risikoklassifizierung", "hochrisiko-systeme",
        "technische dokumentation", "daten-governance", "datenqualität",
        "anonymisierung", "pseudonymisierung", "synthetische daten",
        "transparenz"
    ],

    # 3. AI Automation & Implementation (Low/No-Code)
    "AI Automation & Implementation (Low/No-Code)": [
        # English terms
        "ai agents", "orchestrator", "agent memory", "low-code", "no-code",
        "n8n", "make", "workflow automation", "process optimization",
        "prompt engineering", "zero-shot", "few-shot", "chain-of-thought",
        "tree-of-thought", "self-criticism prompting", "AUTOMAT framework",
        "RTF framework", "TAG framework", "BAB framework", "RISE framework",
        "multimodal ai", "asr", "tts", "ocr", "computer vision",
        # German terms
        "ki-agenten", "orchestrator", "agentenspeicher", "workflow-automatisierung",
        "prozessoptimierung", "prompt engineering", "multimodale ki",
        "automatische spracherkennung", "sprachsynthese", "texterkennung",
        "computer vision"
    ]
}

##########################################################
# Приоритет направлений
##########################################################
DIRECTION_PRIORITY = [
    "Consulting & Business Strategy",
    "Data Management & Compliance (EU Focus)",
    "AI Automation & Implementation (Low/No-Code)"
]

##########################################################
# Типы занятости
##########################################################
EMPLOYMENT_TYPES = ["full-time", "part-time", "contract", "internship", "temporary"]


##########################################################
# Виды работы
##########################################################
WORK_TYPES = ["remote", "on-site", "hybrid"]

##########################################################
# Ключевые слова для грейдов
##########################################################
GRADE_KEYWORDS = {
    "intern": ["intern", "internship", "praktikum"],                         # стажёр
    "junior": ["junior", r"\bjr\b", "trainee", "praktikant", "entry level"], # начальный уровень
    "middle": ["middle", "mid", "specialist", "associate"],                  # средний уровень
    "senior": ["senior", "lead", "principal", "head", "director"]            # старший уровень    
}


##########################################################
# Hard Skills (технические навыки, инструменты, платформы)
##########################################################
HARD_SKILLS = [
    # --- Языки программирования и базы данных ---
    "Python",          # язык для анализа данных, автоматизации и AI‑разработки
    "SQL",             # язык запросов к базам данных
    "PostgreSQL",      # популярная реляционная база данных
    "JSON Schemas",    # формат для структурирования и валидации данных
    "Function Calling",# метод интеграции LLM через API‑вызовы

    # --- Аналитика и бизнес-анализ ---
    "BPMN",            # стандарт моделирования бизнес-процессов
    "SIPOC",           # инструмент описания процессов (Suppliers, Inputs, Process, Outputs, Customers)
    "Value Chain Analysis", # анализ цепочки создания ценности
    "Theory of Constraints",# метод выявления узких мест
    "Ishikawa Diagram",# диаграмма причинно-следственных связей
    "5 Why Analysis",  # метод поиска корневых причин
    "Impact-Effort Matrix", # приоритизация инициатив по усилиям и эффекту
    "Risk-Regret Matrix",   # оценка рисков и последствий
    "AI Readiness Assessment", # оценка зрелости компании в AI
    "Consulting Frameworks",   # фреймворки для консалтинга

    # --- Управление данными и комплаенс ---
    "GDPR Principles", # принципы защиты персональных данных в ЕС
    "AI Act",          # классификация AI-систем по уровню риска
    "DPIA",            # оценка воздействия на защиту данных
    "AI Technical Documentation", # документация для соответствия AI Act
    "Data Lake",       # хранилище сырых данных
    "Data Warehouse",  # централизованное хранилище данных
    "Data Mart",       # специализированное хранилище данных
    "ETL",             # процесс извлечения, трансформации и загрузки данных
    "ELT",             # альтернативный процесс загрузки данных
    "Data Quality Audit", # проверка качества данных
    "Anonymization",   # удаление идентификаторов из данных
    "Pseudonymization",# замена идентификаторов псевдонимами
    "Hashing",         # защита данных через хэширование
    "Synthetic Data",  # генерация искусственных данных
    "Responsible AI",  # этические принципы разработки AI
    "Fairness",        # справедливость в алгоритмах
    "Accountability",  # ответственность за решения AI
    "Transparency",    # прозрачность алгоритмов
    "Bias Management", # управление предвзятостью

    # --- Автоматизация и AI-внедрение ---
    "AI Agents",       # системы, объединяющие модель, память и инструменты
    "Orchestrator",    # компонент управления агентами
    "Agent Memory",    # память агента (краткосрочная/долгосрочная)
    "Low-Code Automation", # автоматизация без программирования
    "No-Code Automation",  # автоматизация без кода
    "n8n",             # платформа для автоматизации процессов
    "Make",            # платформа для Low/No-Code автоматизации
    "Zero-Shot Prompting", # техника промптинга без примеров
    "Few-Shot Prompting",  # техника промптинга с примерами
    "Chain-of-Thought",    # пошаговое рассуждение в промптах
    "Tree-of-Thought",     # ветвление рассуждений в промптах
    "Self-Criticism Prompting", # техника самопроверки модели
    "AUTOMAT Framework",   # фреймворк структурирования запросов
    "RTF Framework",       # фреймворк управления тоном и форматом
    "TAG Framework",       # фреймворк для промптинга
    "BAB Framework",       # фреймворк для бизнес-анализа
    "RISE Framework",      # фреймворк для оценки ответов
    "Multimodal AI",       # интеграция текста, речи и изображений
    "ASR",                 # автоматическое распознавание речи
    "TTS",                 # синтез речи
    "OCR",                 # распознавание текста в изображениях
    "Computer Vision Models", # модели компьютерного зрения
]


##########################################################
# Soft Skil
##########################################################
SOFT_SKILLS = [
    # --- Коммуникация и командная работа ---
    "Communication",       # умение ясно выражать мысли
    "Teamwork",            # работа в команде
    "Collaboration",       # совместное достижение целей
    "Stakeholder Management", # управление ожиданиями заинтересованных сторон
    "Negotiation",         # ведение переговоров
    "Presentation Skills", # навыки презентации
    "Interpersonal Skills",# межличностное взаимодействие
    "Cross-Functional Collaboration", # работа между отделами
    "Conflict Resolution", # решение конфликтов
    "Active Listening",    # внимательное слушание

    # --- Аналитика и мышление ---
    "Analytical Thinking", # структурирование и анализ данных
    "Problem Solving",     # решение сложных задач
    "Strategic Thinking",  # видение долгосрочных целей
    "Root Cause Analysis", # выявление первопричин
    "Decision-Making Under Constraints", # принятие решений в условиях ограничений
    "Consulting Mindset",  # ориентация на клиента и структурный подход
    "Business Acumen",     # понимание бизнес-логики

    # --- Этические и комплаенс-навыки ---
    "Ethical Awareness",   # понимание этических норм
    "Compliance Mindset",  # ориентация на соблюдение правил
    "Risk Awareness",      # внимание к рискам
    "Transparency Orientation", # стремление к прозрачности
    "Attention to Data Quality", # аккуратность в работе с данными

    # --- Личностные качества ---
    "Adaptability",        # гибкость и умение адаптироваться
    "Proactive Attitude",  # инициативность
    "Willingness to Learn",# готовность учиться
    "Self-Motivated",      # самостоятельность
    "Self-Starter",        # умение начинать задачи без внешнего давления
    "Innovation Mindset",  # ориентация на новые идеи
]


##########################################################
# языковые паттерны
##########################################################
LANG_PATTERNS = {
    "german": [r"\bgerman\b", r"\bdeutsch\b"],
    "english": [r"\benglish\b", r"\benglisch\b"]
}

# уровни CEFR
LANG_LEVELS = {
    "A1": [r"\ba1\b", r"beginner"],
    "A2": [r"\ba2\b", r"elementary"],
    "B1": [r"\bb1\b", r"intermediate"],
    "B2": [r"\bb2\b", r"upper intermediate"],
    "C1": [r"\bc1\b", r"advanced"],
    "C2": [r"\bc2\b", r"proficient", r"fluent"]
}

LANG_LEVEL_DESCRIPTIONS = {
    "A1": "Beginner",
    "A2": "Elementary",
    "B1": "Intermediate",
    "B2": "Upper Intermediate",
    "C1": "Advanced",
    "C2": "Proficient / Fluent"
}

##########################################################
# Примерные курсы валют (можно обновлять через API)
##########################################################
CURRENCY_RATES = {
    "USD": 0.92,   # 1 USD = 0.92 EUR
    "EUR": 1.0,    # базовая валюта
    "GBP": 1.15    # 1 GBP = 1.15 EUR
}

##########################################################
# множители для перевода ЗП за час, месяц в год
##########################################################
PERIOD_MULTIPLIERS = {
    "hour": 40 * 52,   # 40 часов в неделю * 52 недели = ~2080 часов
    "month": 12,
    "year": 1
}

##########################################################
# Stepstone: search_titles, synonyms, keywords
##########################################################

STEPSTONE_SEARCH_TITLES = [
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
    "Prozessmanager/in – RPA (Junior)"
]

STEPSTONE_SYNONYMS = {
    "AI Product Manager": [
        "product owner ai", "ai produktmanager", "ml product manager",
        "product manager ai", "ai po", "ai product owner",
        "product manager digital", "product manager data", "product manager tech"
    ],
    "Prompt Engineer": [
        "llm engineer", "nlp engineer", "ai content engineer",
        "generative ai engineer", "prompt specialist",
        "prompting", "llm", "genai", "chatgpt", "ai text"
    ],
    "Prozessmanager/in": [
        "process manager", "business process manager", "process owner",
        "prozess spezialist", "prozessmanager", "process analyst",
        "business analyst process", "process consultant",
        "portfolio process analyst", "process governance analyst"
    ],
    "AI Automation Specialist": [
        "rpa developer", "automation engineer", "intelligent automation",
        "rpa specialist", "automation specialist",
        "low-code automation", "n8n", "make.com", "workflow automation",
        "facility automation", "building automation"
    ],
    "Junior Automation Specialist": [
        "junior rpa", "junior automation", "junior process automation",
        "junior workflow automation"
    ],
    "KI-Manager/in": [
        "ai manager", "manager ki", "leiter ki", "head of ai",
        "ai lead", "ai team lead"
    ],
    "AI Project Manager": [
        "project manager ai", "ai projektmanager", "ml project manager",
        "project manager digital", "project manager data"
    ],
    "Digital Process Analyst": [
        "digital analyst", "web analyst", "data analyst",
        "digital data analyst", "digital forensic analyst",
        "strategic analyst digital", "junior analyst digital",
        "digital retail analyst", "digital feature analyst"
    ],
    "Business Transformation Analyst": [
        "business analyst", "business analyst digital",
        "business analyst insurance", "business analyst finance",
        "business analyst crm", "business analyst growth",
        "business analyst camunda", "business analyst supply chain",
        "business analyst servicecontrolling"
    ],
    "AI Governance Analyst": [
        "ai compliance", "ai governance", "responsible ai",
        "digital compliance", "regulatory compliance",
        "governance analyst", "process governance"
    ],
    "KI-Prompter": [
        "prompting specialist", "ki prompt", "ai prompt",
        "prompt writer", "prompt designer"
    ],
    "Prozessmanager/in – RPA": [
        "rpa process manager", "rpa prozessmanager",
        "process automation rpa", "rpa analyst"
    ],
    "Prozessmanager/in – RPA (Junior)": [
        "junior rpa process manager", "junior rpa prozessmanager",
        "junior rpa analyst"
    ]
}

STEPSTONE_KEYWORDS = {
    "AI Product Manager": ["product", "manager", "ai", "ml", "digital product", "product owner"],
    "Prompt Engineer": ["prompt", "llm", "nlp", "genai", "chatgpt", "ai text", "prompting"],
    "Prozessmanager/in": ["process", "prozess", "workflow", "governance", "operations",
                          "process optimization", "process improvement"],
    "AI Automation Specialist": ["automation", "rpa", "workflow", "low-code", "n8n", "make.com",
                                 "intelligent automation", "robotic"],
    "Junior Automation Specialist": ["junior", "automation", "rpa", "workflow"],
    "KI-Manager/in": ["ai", "ki", "manager", "lead", "head"],
    "AI Project Manager": ["project", "manager", "ai", "digital project", "it project"],
    "Digital Process Analyst": ["digital", "analyst", "web", "data", "forensic",
                                "dashboard", "tracking", "kpi", "reporting"],
    "Business Transformation Analyst": ["business", "analyst", "consultant", "requirements",
                                        "stakeholder", "process", "insurance", "finance"],
    "AI Governance Analyst": ["governance", "compliance", "regulatory", "responsible ai",
                              "audit", "policy"],
    "KI-Prompter": ["prompt", "ki", "ai", "genai"],
    "Prozessmanager/in – RPA": ["rpa", "automation", "prozess", "robotic"],
    "Prozessmanager/in – RPA (Junior)": ["junior", "rpa", "prozess", "automation"]
}

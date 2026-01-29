from pathlib import Path

print("config_ec.py loaded")

##########################################################
# 0. Конфигурация 
##########################################################

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

# Словарь терминов e-commerce
E_COMMERCE_TERMS = {
    "marketplace": [
        "marketplace",        # общий термин для торговой онлайн‑площадки
        "amazon",             # крупнейший международный маркетплейс
        "ebay",               # глобальная платформа для онлайн‑аукционов и торговли
        "fba",                # Fulfillment by Amazon — услуга хранения и доставки
        "seller central",     # интерфейс для продавцов на Amazon
        "market place",       # альтернативное написание marketplace
        "marketplace manager",# менеджер по работе с маркетплейсами
        "marketplace specialist", # специалист по управлению продажами на маркетплейсах
        "marketplace betreuer",   # (нем.) куратор маркетплейса
        "marktplace",         # вариант написания marketplace
        "marktplatz"          # (нем.) торговая площадка
    ],
    "on-line sales": [
        "e-commerce",         # электронная коммерция, онлайн‑торговля
        "ecommerce",          # то же самое, другое написание
        "online sales",       # продажи через интернет
        "online retail",      # интернет‑ритейл, розничная торговля онлайн
        "webshop",            # интернет‑магазин
        "onlineshop",         # (нем.) интернет‑магазин
        "shopify",            # платформа для создания интернет‑магазинов
        "e-tail",             # electronic retail — онлайн‑ритейл
        "order management",   # управление заказами
        "fulfillment",        # процесс обработки заказов: хранение, упаковка, доставка
        "merchandiser",       # специалист по ассортименту и выкладке
        "category manager",   # менеджер по товарной категории
        "vertrieb"            # (нем.) сбыт, продажи
    ],
    "on-line marketing": [
        "marketing",          # маркетинг, продвижение товаров и услуг
        "performance",        # performance‑маркетинг (ориентирован на измеримые результаты)
        "paid",               # платная реклама (Paid Ads)
        "ppc",                # Pay‑Per‑Click реклама
        "seo",                # Search Engine Optimization — оптимизация для поисковиков
        "sem",                # Search Engine Marketing — маркетинг в поисковых системах
        "crm",                # Customer Relationship Management — управление клиентами
        "retention",          # удержание клиентов
        "email",              # email‑маркетинг
        "influencer",         # маркетинг через лидеров мнений
        "social commerce",    # продажи через социальные сети
        "conversion",         # конверсия (посетитель → покупатель)
        "cvr",                # Conversion Rate — коэффициент конверсии
        "cro",                # Conversion Rate Optimization — оптимизация конверсии
        "ux",                 # User Experience — пользовательский опыт
        "growth",             # рост бизнеса, growth‑маркетинг
        "ga4",                # Google Analytics 4 — инструмент веб‑аналитики
        "ga "                 # Google Analytics (старые версии)
    ]
}

##########################################################
# Приоритет направлений
##########################################################
DIRECTION_PRIORITY = ["marketplace", "on-line sales", "on-line marketing"]


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
    "junior": ["junior", r"\bjr\b", "trainee", "praktikant", "entry level"], # начальный уровень
    "middle": ["middle", "mid", "specialist", "associate"],                  # средний уровень
    "senior": ["senior", "lead", "principal", "head", "director"],           # старший уровень
    "intern": ["intern", "praktikum"]                                        # стажёр
}


##########################################################
# Hard Skills (технические навыки, инструменты, платформы)
##########################################################
HARD_SKILLS = [
    # Платформы и CMS
    "shopify", "magento", "woocommerce", "shopware", "wordpress",   # системы для e-commerce

    # Маркетплейсы
    "amazon", "seller central",   # торговая площадка и интерфейс для продавцов

    # Аналитика и BI
    "google analytics", "ga4", "tableau", "power bi", "sap", 
    "looker studio", "mixpanel", "web analytics",   # инструменты аналитики и визуализации

    # Языки программирования и базы данных
    "sql", "python", "html", "css", "javascript", "api", "rest", "postgresql",   # разработка и базы данных

    # CRM и сервисные системы
    "salesforce", "zendesk", "crm", "hubspot", "pipedrive", "sap crm",   # управление клиентами

    # Облако и инфраструктура
    "aws", "docker", "kubernetes", "linux", "terraform",   # облачные технологии и инфраструктура

    # Реклама и маркетинг
    "adwords", "google ads", "linkedin ads", "meta ads", "seo", 
    "social media marketing", "marketing automation",   # рекламные и маркетинговые инструменты

    # Офисные инструменты
    "excel", "microsoft office", "google workspace", "powerpoint",   # офисные пакеты и презентации

    # Дизайн и мультимедиа
    "adobe photoshop", "graphic design", "video editing",   # дизайн и обработка мультимедиа

    # Прочее
    "edi", "generative ai tools"   # электронный обмен данными, инструменты ИИ
]

##########################################################
# Soft Skills (личные качества, управленческие навыки)
##########################################################
SOFT_SKILLS = [
    # Коммуникация и взаимодействие
    "communication", "teamwork", "collaboration", "stakeholder management", "negotiation",   # общение и работа с людьми

    # Аналитика и мышление
    "analytical thinking", "problem solving", "strategic thinking",   # мышление и решение задач

    # Организация и управление
    "organisation", "project management", "time management",   # планирование и управление процессами

    # Личностные качества
    "adaptability", "proactive attitude", "self-motivated", "self-starter", "willingness to learn",   # личная эффективность

    # Лидерство и влияние
    "leadership", "presentation skills", "storytelling",   # управление и влияние

    # Креативность и внимание к деталям
    "detail-oriented", "attention to detail", "creativity", "writing skills"   # креативность и аккуратность
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
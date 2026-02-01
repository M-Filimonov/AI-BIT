import os
import re
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Вспомогательная функция для сохранения графиков
# ------------------------------------------------------------

def save_plot(name: str):
    os.makedirs("AA_output/analytics", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"AA_output/analytics/{name}.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# Подписи значений на графиках
# ------------------------------------------------------------

def add_value_labels(ax, spacing=5, fontsize=8):
    for p in ax.patches:
        if p.get_height() > p.get_width():  # вертикальный barplot
            value = p.get_height()
            if value >= 1:
                ax.annotate(
                    f"{int(value)}",
                    (p.get_x() + p.get_width() / 2, value),
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    xytext=(0, spacing),
                    textcoords="offset points"
                )
        else:  # горизонтальный barplot
            value = p.get_width()
            if value >= 1:
                ax.annotate(
                    f"{int(value)}",
                    (value, p.get_y() + p.get_height() / 2),
                    ha="left",
                    va="center",
                    fontsize=fontsize,
                    xytext=(spacing, 0),
                    textcoords="offset points"
                )


# ------------------------------------------------------------
# Парсинг строки из поля 'location'
# ------------------------------------------------------------

GERMAN_STATES = {
    "baden-württemberg", "bayern", "berlin", "brandenburg", "bremen",
    "hamburg", "hessen", "mecklenburg-vorpommern", "niedersachsen",
    "nordrhein-westfalen", "rheinland-pfalz", "saarland", "sachsen",
    "sachsen-anhalt", "schleswig-holstein", "thüringen"
}

CITY_STATES = {"berlin", "hamburg", "bremen"}


def parse_location_field(location: str) -> dict:
    """
    Нормализует строку из поля 'location' (Arbeitsagentur API) и извлекает:
    - index   — почтовый индекс (5 цифр)
    - city    — город
    - state   — федеральная земля Германии
    - country — страна
    - address — улица/дом (если есть)

    Логика:
    1. Удаляются элементы 'null'.
    2. Каждый элемент классифицируется:
       - 5 цифр подряд → index
       - одно из 16 земель → state
       - 'Deutschland' → country
       - строка с цифрами+буквами или типичным окончанием улицы → address
    3. city = первый элемент, который не попал в index/state/country/address.
    4. Для городов‑земель (Berlin, Hamburg, Bremen):
       city = state.
    5. Если исходная строка заканчивалась на 'null':
       address = 'unknown'.
    6. Все отсутствующие значения → 'unknown'.

    Функция устойчива к строкам длиной от 1 до 6 элементов и корректно
    обрабатывает реальные кейсы Arbeitsagentur.
    """

    # Базовый результат
    result = {k: None for k in ["index", "city", "state", "country", "address"]}

    # Пустой ввод
    if not isinstance(location, str) or not location.strip():
        return {k: "unknown" for k in result}

    # Разбивка и очистка
    raw_parts = [p.strip() for p in location.split(",") if p.strip()]
    had_null_at_end = raw_parts and raw_parts[-1].lower() == "null"
    parts = [p for p in raw_parts if p.lower() != "null"]

    if not parts:
        return {k: "unknown" for k in result}

    # --- Классификаторы ---
    def is_index(s): return bool(re.fullmatch(r"\d{5}", s))
    def is_country(s): return s.lower() == "deutschland"
    def is_state(s): return s.lower() in GERMAN_STATES
    def is_address(s):
        if re.search(r"[0-9]", s) and re.search(r"[A-Za-zÄÖÜäöüß]", s):
            return True
        if re.search(r"(straße|str\.|weg|allee|platz|ring|gasse)$", s.lower()):
            return True
        return False

    # --- Этап 1: классификация index/state/country/address ---
    for p in parts:
        low = p.lower()
        if result["index"] is None and is_index(p):
            result["index"] = p
        elif result["state"] is None and is_state(p):
            result["state"] = p
        elif result["country"] is None and is_country(p):
            result["country"] = p
        elif result["address"] is None and is_address(p):
            result["address"] = p

    # --- Этап 2: city = первый неклассифицированный элемент ---
    classified_values = {v for v in result.values() if v is not None}
    for p in parts:
        if p not in classified_values:
            result["city"] = p
            break

    # --- Город‑земля ---
    if result["state"] and result["state"].lower() in CITY_STATES:
        result["city"] = result["state"]

    # --- Если null был в конце → адрес отсутствует ---
    if had_null_at_end:
        result["address"] = "Unknown"

    # --- None → unknown ---
    for k in result:
        if result[k] is None:
            result[k] = "Unknown"

    return result

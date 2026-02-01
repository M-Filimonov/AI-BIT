"""
aa_utils.py
-----------------
Utility functions used across analytics and normalization modules.

Includes:
- save_plot(): unified plot saving helper
- add_value_labels(): bar chart value annotations
- parse_location_field(): robust parsing of the 'location' field from the
  Arbeitsagentur API into structured components (index, city, state, country, address)
"""

import os
import re
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Plot saving helper
# ------------------------------------------------------------

def save_plot(name: str):
    """Saves a plot into AA_output/analytics with consistent formatting."""
    os.makedirs("AA_output/analytics", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"AA_output/analytics/{name}.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# Value labels for bar charts
# ------------------------------------------------------------

def add_value_labels(ax, spacing=5, fontsize=8):
    """
    Adds numeric labels to bars in a barplot.

    Automatically detects orientation:
    - vertical bars → label above the bar
    - horizontal bars → label to the right of the bar
    """
    for p in ax.patches:
        if p.get_height() > p.get_width():  # vertical barplot
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
        else:  # horizontal barplot
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
# Parsing the 'location' field
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
    Normalizes the 'location' string from the Arbeitsagentur API and extracts:

    - index   — postal code (5 digits)
    - city    — city name
    - state   — German federal state
    - country — country
    - address — street/house (if present)

    Logic:
    1. Removes 'null' elements.
    2. Each element is classified:
       - 5 digits → index
       - one of the 16 German states → state
       - 'Deutschland' → country
       - strings containing digits+letters or typical street suffixes → address
    3. city = first element not classified as index/state/country/address.
    4. For city-states (Berlin, Hamburg, Bremen):
       city = state.
    5. If original string ended with 'null':
       address = 'Unknown'.
    6. Missing values → 'Unknown'.

    The function is robust to strings of length 1–6 and handles real-world
    Arbeitsagentur cases reliably.
    """

    # Base result
    result = {k: None for k in ["index", "city", "state", "country", "address"]}

    # Empty input
    if not isinstance(location, str) or not location.strip():
        return {k: "Unknown" for k in result}

    # Split and clean
    raw_parts = [p.strip() for p in location.split(",") if p.strip()]
    had_null_at_end = raw_parts and raw_parts[-1].lower() == "null"
    parts = [p for p in raw_parts if p.lower() != "null"]

    if not parts:
        return {k: "Unknown" for k in result}

    # --- Classifiers ---
    def is_index(s): return bool(re.fullmatch(r"\d{5}", s))
    def is_country(s): return s.lower() == "deutschland"
    def is_state(s): return s.lower() in GERMAN_STATES
    def is_address(s):
        if re.search(r"[0-9]", s) and re.search(r"[A-Za-zÄÖÜäöüß]", s):
            return True
        if re.search(r"(straße|str\.|weg|allee|platz|ring|gasse)$", s.lower()):
            return True
        return False

    # --- Step 1: classify index/state/country/address ---
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

    # --- Step 2: city = first unclassified element ---
    classified_values = {v for v in result.values() if v is not None}
    for p in parts:
        if p not in classified_values:
            result["city"] = p
            break

    # --- City-states ---
    if result["state"] and result["state"].lower() in CITY_STATES:
        result["city"] = result["state"]

    # --- If null was at the end → address missing ---
    if had_null_at_end:
        result["address"] = "Unknown"

    # --- None → Unknown ---
    for k in result:
        if result[k] is None:
            result[k] = "Unknown"

    return result

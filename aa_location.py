"""
aa_location.py
---------------------
Geolocation and geographic analytics utilities for the project.

Functionality includes:
1. City name normalization
2. Cached geocoding using Nominatim
3. Coordinate enrichment for vacancy datasets
4. City‑level and state‑level vacancy statistics
5. Bar charts for cities and federal states
6. Interactive geographic maps (Plotly)
7. Experience‑level–specific geographic analysis
8. Color palette resolution for visualizations

This module is used by:
- run_geo_analysis() in aa_main.py
- geo_by_experience() in aa_analytic.py
- city/state distribution charts
- coordinate enrichment during analytics
"""



import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from time import sleep

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from geopy.geocoders import Nominatim

from aa_utils import add_value_labels, save_plot
from aa_config import GERMAN_STATES, HTML_CACHE_DIR, OUT_DIR


# ============================================================
# Geocoding with caching
# ============================================================

geolocator = Nominatim(user_agent="aa_job_market_analysis")
CACHE_FILE = HTML_CACHE_DIR / "city_geocache.csv"


def normalize_city(name: str) -> str:
    """Unified normalization of city names."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()

    # remove parentheses (e.g., Berlin (DE))
    name = re.sub(r"\(.*?\)", "", name)

    # remove commas and extra spaces
    name = name.replace(",", " ")
    name = re.sub(r"\s+", " ", name)

    return name.title()


def load_geocache():
    """Loads geocache and normalizes city names."""
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE)
        df["City"] = df["City"].astype(str).str.strip().str.title()
        return df.dropna(subset=["Latitude", "Longitude"])
    return pd.DataFrame(columns=["City", "Latitude", "Longitude"])


def save_geocache(df: pd.DataFrame):
    """Saves geocache without NaN values and without duplicates."""
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df.drop_duplicates(subset=["City"])
    df.to_csv(CACHE_FILE, index=False)


def geocode_city(city: str, retries=2):
    """Geocodes a single city name."""
    if not city:
        return None, None

    for _ in range(retries):
        try:
            loc = geolocator.geocode(f"{city}, Germany")
            sleep(1)
            if loc:
                return loc.latitude, loc.longitude
        except Exception:
            sleep(2)

    return None, None


# ============================================================
# enrich_with_coordinates — used in prepare_city_stats
# ============================================================

def enrich_with_coordinates(city_stats: pd.DataFrame):
    """
    Robust coordinate enrichment:
    - normalizes all city names
    - loads and normalizes geocache
    - identifies only new cities
    - geocodes only new entries
    - updates geocache
    - returns city_stats with coordinates
    """

    # --- 1. Normalize input data ---
    city_stats = city_stats.copy()
    city_stats["City"] = city_stats["City"].astype(str).apply(normalize_city)

    # --- 2. Load and normalize cache ---
    cache = load_geocache().copy()
    cache["City"] = cache["City"].astype(str).apply(normalize_city)

    # remove duplicates after normalization
    cache = cache.drop_duplicates(subset=["City"])

    cached_cities = set(cache["City"])
    all_cities = set(city_stats["City"])

    # --- 3. Determine new cities ---
    new_cities = sorted(all_cities - cached_cities)

    print(f"[INFO] Total cities: {len(all_cities)}")
    print(f"[INFO] Cached cities: {len(cached_cities)}")
    print(f"[INFO] New cities to geocode: {len(new_cities)}")

    # --- 4. Geocode only new cities ---
    new_rows = []
    for city in tqdm(new_cities, desc="Geocoding new cities", unit="city"):
        lat, lon = geocode_city(city)
        if lat and lon:
            new_rows.append({"City": city, "Latitude": lat, "Longitude": lon})

    # --- 5. Update cache ---
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, new_df], ignore_index=True)

        # final cleanup
        cache = cache.dropna(subset=["Latitude", "Longitude"])
        cache = cache.drop_duplicates(subset=["City"])

        save_geocache(cache)

    # --- 6. Merge coordinates into city_stats ---
    merged = city_stats.merge(cache, on="City", how="left")

    # --- 7. Return only cities with coordinates ---
    return merged.dropna(subset=["Latitude", "Longitude"])


# ============================================================
# City statistics preparation — used in run_geo_analysis
# ============================================================

def prepare_city_stats(df: pd.DataFrame):
    """Aggregates vacancy counts by city and enriches them with coordinates."""
    city_stats = (
        df.dropna(subset=["city"])
          .groupby("city", as_index=False)
          .size()
          .rename(columns={"city": "City", "size": "Vacancies"})
    )
    return enrich_with_coordinates(city_stats)


def prepare_city_stats_by_level(df: pd.DataFrame, level: str):
    """Filters by experience level and prepares city statistics."""
    if level is None:
        return prepare_city_stats(df)
    lvl = str(level).strip().lower()
    return prepare_city_stats(df[df["experience_level"].astype(str).str.lower() == lvl])


# ============================================================
# Bar chart: Top‑25 cities — used in run_geo_analysis, aa_analytic.run_analytics
# ============================================================

def city_distribution(df: pd.DataFrame, top_n=25, suffix="All", palette="Greys_r", color=None):
    """Plots a bar chart of the top N cities by vacancy count."""
    counts = (
        df["city"]
        .dropna()
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
        .iloc[:top_n]
    )
    if counts.empty:
        return

    plt.figure(figsize=(10, 9))
    n = len(counts)

    # resolve palette or single color
    resolved = resolve_colors(palette if palette is not None else color, n)

    if isinstance(resolved, str):
        ax = sns.barplot(x=counts.values, y=counts.index, color=resolved, hue=counts.index, dodge=False)
    else:
        ax = sns.barplot(x=counts.values, y=counts.index, palette=resolved, hue=counts.index, dodge=False)

    ax.tick_params(axis="y", labelsize=10)
    add_value_labels(ax)

    plt.title(
        f"Top {top_n} Städte nach Anzahl der Stellenangebote (Erfahrungsstufe: {suffix.upper()})",
        fontsize=14
    )
    plt.xlabel("Anzahl der Stellenangebote")
    plt.ylabel("Stadt")

    save_plot(f"09_AA_city_distribution_{suffix.upper()}")


# ============================================================
# Bar chart: Top‑16 states — used in run_geo_analysis, aa_analytic.run_analytics
# ============================================================

def state_distribution(df: pd.DataFrame, top_n=16, suffix="All", palette="Purples_r", color=None):
    """Plots a bar chart of the top N German states by vacancy count."""
    counts = (
        df["state"]
        .dropna()
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
        .iloc[:top_n]
    )
    if counts.empty:
        return

    plt.figure(figsize=(10, 7))
    n = len(counts)
    resolved = resolve_colors(palette if palette is not None else color, n)

    if isinstance(resolved, str):
        ax = sns.barplot(x=counts.values, y=counts.index, color=resolved, hue=counts.index, dodge=False)
    else:
        ax = sns.barplot(x=counts.values, y=counts.index, palette=resolved, hue=counts.index, dodge=False)

    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)

    add_value_labels(ax)
    plt.title(
        f"Verteilung der Stellenangebote nach Bundesländern (Erfahrungsstufe: {suffix.upper()})",
        fontsize=14
    )
    plt.xlabel("Anzahl der Stellenangebote", fontsize=12)
    plt.ylabel("Bundesland", fontsize=12)

    save_plot(f"10_AA_state_distribution_{suffix.upper()}")


# ============================================================
# Interactive Germany map — used in run_geo_analysis
# ============================================================

def vacancy_map(city_stats: pd.DataFrame, title: str, filename: str):
    """Creates an interactive map of vacancies across Germany."""
    fig = px.scatter_map(
        city_stats,
        lat="Latitude",
        lon="Longitude",
        size="Vacancies",
        color="Vacancies",
        hover_name="City",
        hover_data={"Latitude": False, "Longitude": False},
        color_continuous_scale="RdYlGn",
        size_max=35,
        zoom=5,
        center={"lat": 51.1657, "lon": 10.4515},
        title=title
    )

    fig.update_layout(map_style="carto-positron")
    out_path = OUT_DIR / "analytics"
    fig.write_html(out_path / filename)


# ============================================================
# Full geo‑analytics pipeline — used in aa_main.run_with_analytics
# ============================================================

def run_geo_analysis(df: pd.DataFrame):
    """Runs full geographic analytics: all, entry, advanced."""

    # all vacancies
    city_stats_all = prepare_city_stats(df)
    vacancy_map(
        city_stats_all,
        "Geografische Verteilung der Stellenangebote in Deutschland",
        "06_AA_geo_vacancy_map_ALL.html"
    )

    # entry level
    city_stats_Entry = prepare_city_stats_by_level(df, "entry")
    vacancy_map(
        city_stats_Entry,
        "Stellenangebote – Entry Level",
        "07_AA_geo_vacancy_map_ENTRY.html"
    )

    # advanced level
    city_stats_Advanced = prepare_city_stats_by_level(df, "advanced")
    vacancy_map(
        city_stats_Advanced,
        "Stellenangebote – Advanced Level",
        "08_AA_geo_vacancy_map_ADVANCED.html"
    )


# ============================================================
# geo_by_experience — used in aa_analytic.run_analytics
# ============================================================

def geo_by_experience(df: pd.DataFrame, level: str):
    """Runs city/state distribution charts for a specific experience level."""
    if not isinstance(level, str):
        print("Level must be a string")
        return

    df_lvl = df[df["experience_level"].astype(str).str.lower() == level.lower()]

    if df_lvl.empty:
        print(f"No data for level {level}")
        return

    suffix = level.lower()

    # palettes
    if suffix == "entry":
        city_palette = "Oranges_r"
        state_palette = "Oranges_d"
    else:
        city_palette = "Greens_r"
        state_palette = "GnBu_r"

    city_distribution(df_lvl, top_n=25, suffix=suffix, palette=city_palette)
    state_distribution(df_lvl, top_n=16, suffix=suffix, palette=state_palette)


# ============================================================
# resolve_colors — used in city_distribution, state_distribution
# ============================================================

import matplotlib.colors as mcolors

def resolve_colors(palette_or_color, n):
    """
    Returns:
      - a list of n colors (if palette_or_color is a palette name or list),
      - or a single valid color (hex or named) if palette_or_color is a color string.
    """
    if palette_or_color is None:
        return sns.color_palette("Oranges_r", n_colors=max(1, n))

    # list of colors
    if isinstance(palette_or_color, (list, tuple)):
        pal = list(palette_or_color)
        if len(pal) >= n:
            return pal[:n]
        return sns.color_palette(pal, n_colors=max(1, n))

    # string: either a color or a palette name
    if isinstance(palette_or_color, str):
        if mcolors.is_color_like(palette_or_color):
            return palette_or_color
        try:
            return sns.color_palette(palette_or_color, n_colors=max(1, n))
        except Exception:
            return sns.color_palette("Greens_r", n_colors=max(1, n))

    # fallback
    return sns.color_palette("bone_r", n_colors=max(1, n))

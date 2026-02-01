"""
aa_analytic.py
-----------------
Analytics module for Arbeitsagentur vacancy data.

Functions:
1. Vacancy distribution by experience level (with chart)
2. Vacancy distribution by search terms (with charts)
3. Monthly vacancy dynamics (with chart)
4. Top HARD skills (with chart)
5. Top SOFT skills (with chart)
6. Clustering of job titles (per search_term)
7. Repeated skill words in descriptions by experience level
8. Semantic clustering of skills (HARD / SOFT)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import re
import numpy as np

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from aa_utils import add_value_labels, save_plot
from aa_location import (
    city_distribution,
    state_distribution,
    geo_by_experience
)
from aa_config import SEARCH_TERMS, OUT_DIR

sns.set(style="whitegrid")  # grid style for charts


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Loads XLSX data with parsed posted_date."""
    df = pd.read_excel(path, parse_dates=["posted_date"])
    return df


# ------------------------------------------------------------
# Palette helper
# ------------------------------------------------------------

def palette_for_n(palette_name, n):
    """
    Returns a list of n colors.
    palette_name may be a palette name or a list of colors.
    Safe fallback — tab10.
    """
    try:
        if isinstance(palette_name, str):
            return sns.color_palette(palette_name, n_colors=max(1, n))
        pal = list(palette_name)
        if len(pal) >= n:
            return pal[:n]
        return sns.color_palette(pal, n_colors=max(1, n))
    except Exception:
        return sns.color_palette("tab10", n_colors=max(1, n))


# ------------------------------------------------------------
# 1. Vacancy distribution by experience level
# ------------------------------------------------------------

def experience_distribution(df: pd.DataFrame):
    print("\n=== Arbeitsagentur: Distribution of vacancies by experience level ===")

    df["experience_level"] = (
        df.get("experience_level", pd.Series(["unknown"] * len(df)))
          .fillna("unknown")
          .astype(str)
          .str.strip()
          .str.lower()
    )

    counts = df["experience_level"].value_counts()
    total = counts.sum()
    labels = [
        f"{cat}\n{cnt} ({cnt / total * 100:.1f}%)"
        for cat, cnt in zip(counts.index, counts.values)
    ]

    color_map = {
        "entry": "#d65f36",     # orange-red
        "advanced": "#2ca02c",  # green
    }
    colors = [color_map.get(cat, "#b0b0b0") for cat in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        textprops={"fontsize": 14}
    )

    ax.axis("equal")
    plt.title("Distribution by experience level – count and percentage", fontsize=16)
    save_plot("01_AA_experience_distribution")
    plt.show()


# ------------------------------------------------------------
# 2. Vacancy distribution by search terms
# ------------------------------------------------------------

def search_term_distribution(df: pd.DataFrame):
    """
    Builds vacancy distribution by search terms:
    1) overall chart for all vacancies
    2) separate charts by experience level (entry, advanced)
    """

    print("\n=== Arbeitsagentur: Distribution of vacancies by search terms ===")

    # 1. Overall distribution
    counts_all = df["search_term"].value_counts()

    print("\n--- Overall (all experience levels) ---")
    print(counts_all)

    plt.figure(figsize=(10, 5))
    pal = palette_for_n("Blues_r", len(counts_all))
    df_plot_all = pd.DataFrame({
        "term": counts_all.index.astype(str),
        "value": counts_all.values
    })
    ax = sns.barplot(
        data=df_plot_all,
        y="value",
        x="term",
        hue="term",
        palette=pal,
        dodge=False
    )
    add_value_labels(ax)

    plt.title("Arbeitsagentur: Vacancy distribution by search terms (All)")
    plt.xlabel("Search term")
    plt.ylabel("Vacancy count")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    save_plot("02_AA_search_term_distribution_all")

    # 2. Charts by experience level
    exp_levels = df["experience_level"].fillna("unknown").astype(str).str.strip().unique()

    for level in exp_levels:
        subset = df[df["experience_level"].astype(str).str.strip() == level]
        if subset.empty:
            continue

        counts = subset["search_term"].value_counts()

        print(f"\n--- Experience level: {level} ---")
        print(counts)

        plt.figure(figsize=(10, 5))
        lvl = str(level).strip().lower()
        if "entry" in lvl or "junior" in lvl:
            pal = palette_for_n("Oranges_r", len(counts))
        else:
            pal = palette_for_n("Greens_r", len(counts))

        df_plot_lvl = pd.DataFrame({
            "term": counts.index.astype(str),
            "value": counts.values
        })
        ax = sns.barplot(
            data=df_plot_lvl,
            y="value",
            x="term",
            hue="term",
            palette=pal,
            dodge=False
        )
        add_value_labels(ax)

        plt.title(
            f"Arbeitsagentur: Vacancy distribution by search terms "
            f"for experience level: ({level.upper()})"
        )
        plt.xlabel("Search term")
        plt.ylabel("Vacancy count")
        plt.xticks(rotation=45, ha="right", fontsize=8)

        save_plot(f"02_AA_search_term_distribution_{level}")


# ------------------------------------------------------------
# 3. Monthly vacancy dynamics
# ------------------------------------------------------------

def monthly_dynamics(df: pd.DataFrame):
    """
    Builds monthly vacancy dynamics:
    - overall line for all vacancies
    - separate lines for experience levels (entry, advanced)
    - all on one chart with legend
    """
    import matplotlib.dates as mdates

    print("\n=== Arbeitsagentur: Monthly vacancy dynamics ===")

    if "posted_date" not in df.columns:
        print("posted_date column not found")
        return

    df_local = df.copy()

    if "experience_level" in df_local.columns:
        df_local["experience_level"] = (
            df_local["experience_level"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        df_local["experience_level"] = "unknown"

    df_local["posted_date"] = pd.to_datetime(df_local["posted_date"], errors="coerce")
    df_local = df_local.dropna(subset=["posted_date"])
    if df_local.empty:
        print("No valid posted_date values found")
        return

    df_local["month"] = df_local["posted_date"].dt.to_period("M")

    counts_all = df_local.groupby("month").size().sort_index()
    if counts_all.empty:
        print("No aggregated monthly data")
        return
    months_all = counts_all.index.to_timestamp()

    exp_levels = sorted(df_local["experience_level"].dropna().unique())

    colors = {
        "all": "#1f77b4",       # blue
        "entry": "#ff8c00",     # darkorange
        "advanced": "#2e8b57",  # seagreen
        "unknown": "#7f7f7f"    # grey
    }

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        x=months_all,
        y=counts_all.values,
        marker="o",
        color=colors["all"],
        label="All"
    )

    text_color = "#222222"
    for x, y in zip(months_all, counts_all.values):
        plt.text(x, y, str(int(y)), fontsize=8, ha="center", va="bottom", color=text_color)

    for level in exp_levels:
        subset = df_local[df_local["experience_level"] == level]
        if subset.empty:
            continue

        counts = subset.groupby("month").size().sort_index()
        if counts.empty:
            continue
        months = counts.index.to_timestamp()

        lvl_key = "unknown"
        if "entry" in level or "junior" in level:
            lvl_key = "entry"
        elif "senior" in level or "advanced" in level or "expert" in level:
            lvl_key = "advanced"

        sns.lineplot(
            x=months,
            y=counts.values,
            marker="o",
            color=colors.get(lvl_key, colors["unknown"]),
            label=level.capitalize()
        )

        for x, y in zip(months, counts.values):
            plt.text(x, y, str(int(y)), fontsize=8, ha="center", va="bottom", color=text_color)

    plt.title("Arbeitsagentur: Monthly vacancy dynamics (All + experience levels)")
    plt.xlabel("Month")
    plt.ylabel("Vacancy count")

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.xticks(rotation=45, ha="right", fontsize=8)

    plt.legend(title="Experience level")
    plt.tight_layout()

    save_plot("03_AA_monthly_dynamics_combined")


# ------------------------------------------------------------
# 4–5. Top HARD and SOFT skills by experience level
# ------------------------------------------------------------

def plot_top_skills_by_experience(df: pd.DataFrame,
                                  column: str,
                                  title_prefix: str,
                                  palette: str = None,
                                  top_n: int = 20):
    """
    Plots top-N skills (hard or soft) for each experience level.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe.")
        return

    exp_levels = sorted(df["experience_level"].fillna("unknown").astype(str).unique())

    print(f"\nDetected experience levels: {exp_levels}")

    for level in exp_levels:
        subset = df[df["experience_level"].fillna("unknown").astype(str) == level]
        if subset.empty:
            continue

        print(f"\n=== Top {top_n} {title_prefix}-skills (experience: {level}) ===")

        counter = Counter()
        for skills in subset[column].dropna():
            for s in skills.split(","):
                s = s.strip()
                if s:
                    counter[s] += 1

        if not counter:
            print("No skills found for this level.")
            continue

        top = counter.most_common(top_n)

        for skill, count in top:
            print(f"{skill}: {count}")

        skills, counts = zip(*top)
        n = len(skills)

        if palette:
            pal = palette_for_n(palette, n)
        else:
            lvl = str(level).strip().lower()
            if "entry" in lvl or "junior" in lvl:
                pal = palette_for_n("Oranges_r", n)
            else:
                pal = palette_for_n("Greens_r", n)

        plt.figure(figsize=(10, 6))
        df_plot_skills = pd.DataFrame({
            "skill": list(skills),
            "count": list(counts)
        })
        ax = sns.barplot(
            data=df_plot_skills,
            x="count",
            y="skill",
            hue="skill",
            palette=pal,
            dodge=False
        )

        if ax.get_legend() is not None:
            ax.get_legend().remove()
        add_value_labels(ax)

        plt.title(f"Arbeitsagentur: Top {top_n} {title_prefix}-skills ({level})")
        plt.xlabel("Count")
        plt.ylabel("Skill")

        fname = f"04_AA_top_{title_prefix.lower()}_skills_{level}"
        if "hard" not in title_prefix.lower():
            fname = f"05_AA_top_{title_prefix.lower()}_skills_{level}"
        save_plot(fname)


# ------------------------------------------------------------
# 6. Clustering job titles by search_term
# ------------------------------------------------------------

GERMAN_STOPWORDS = [
    "und", "oder", "aber", "nicht", "mit", "für", "von", "im", "in", "am",
    "an", "auf", "aus", "dem", "der", "die", "das", "ein", "eine", "einer",
    "ist", "sind", "war", "werden", "wird", "zu", "zum", "zur"
]

TITLE_NOISE_PATTERNS = [
    r"\b(m/w/d|mwd|m/f/d)\b",
    r"\b(remote|hybrid|onsite|home\s*office)\b",
    r"\b(vollzeit|teilzeit|werkstudent|praktikum)\b",
    r"\b(junior|jr\.?|mid|senior|sr\.?|lead|principal|expert)\b"
]


def clean_job_title(title: str) -> str:
    """Cleans job titles from noise, patterns, and punctuation."""
    if not isinstance(title, str):
        return ""
    t = title.lower()
    for p in TITLE_NOISE_PATTERNS:
        t = re.sub(p, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def top_terms(vectorizer, centroid, n=8):
    """Returns top-N terms for a cluster centroid."""
    terms = vectorizer.get_feature_names_out()
    weights = centroid.ravel()
    idx = np.argsort(weights)[::-1][:n]
    return [terms[i] for i in idx if weights[i] > 0]


def title_clusters_by_experience(
    df: pd.DataFrame,
    clusters_per_group=5,
    out_dir=OUT_DIR / "analytics"
):
    """
    Clusters job titles (job_title) grouped by search_term,
    separately for Entry and Advanced experience levels.

    Uses TF-IDF + KMeans, cleans titles, prints representative examples
    and cluster terms. Saves results to text files in analytics folder.
    For each cluster it logs:
      - search_term (as beruf label)
      - cluster key terms
      - sample titles.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for level in ["entry", "advanced"]:
        df_lvl = df[df["experience_level"].fillna("").astype(str) == level]

        out_file = out_dir / f"11_title_clusters_{level.upper()}.txt"
        lines = []

        def log(s=""):
            print(s)
            lines.append(s)

        log(f"\n=== Title clusters for experience level: {level.upper()} ===")

        for term, group in df_lvl.groupby("search_term"):
            beruf = term
            raw_titles = group["job_title"].dropna().astype(str).tolist()
            cleaned = [(t, clean_job_title(t)) for t in raw_titles]
            cleaned = [(r, c) for r, c in cleaned if c]

            if len(cleaned) < clusters_per_group:
                log(f"\n[ {beruf} ] — insufficient data")
                continue

            titles_raw = [x[0] for x in cleaned]
            titles_clean = [x[1] for x in cleaned]

            log(f"\n[ {beruf} ] — Clustering (n={len(titles_clean)})")

            vectorizer = TfidfVectorizer(
                stop_words=GERMAN_STOPWORDS,
                ngram_range=(1, 2),
                min_df=2
            )

            X = vectorizer.fit_transform(titles_clean)
            if X.shape[1] == 0:
                log("  ⚠ No features left after cleaning")
                continue

            k = min(clusters_per_group, X.shape[0])
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X)

            df_term = pd.DataFrame({
                "title": titles_raw,
                "cluster": labels
            })

            centers = kmeans.cluster_centers_

            for c in range(k):
                cluster_titles = df_term[df_term["cluster"] == c]["title"].tolist()
                if not cluster_titles:
                    continue

                terms = top_terms(vectorizer, centers[c])
                terms_str = ", ".join(terms) if terms else "(no terms)"

                log(f"\n Cluster {c} (n={len(cluster_titles)})")
                log(f"   Search term: {beruf}")
                log(f"   Cluster terms: {terms_str}")
                log("    Examples:")

                for t in cluster_titles[:5]:
                    log(f"    - {t}")

        out_file.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# 7.1 Repeated HARD-skill words by experience level
# ------------------------------------------------------------

def repeated_hard_skill_words_by_level(
    df: pd.DataFrame,
    top_n: int = 30,
    out_dir = OUT_DIR / "analytics"
):
    """
    Compares repeated words in hard_skills for entry vs advanced levels.
    Returns a long-format DataFrame: word, level, count, total.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot = out_dir / "12_repeated_HARD_skill_words_entry_vs_advanced.png"
    out_file = out_dir / "12_repeated_HARD_skill_words_entry_vs_advanced.txt"

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("\n=== Repeated HARD skill words by experience level ===")

    if "hard_skills" not in df.columns:
        log("Column 'hard_skills' not found.")
        Path(out_file).write_text("\n".join(lines), encoding="utf-8")
        return pd.DataFrame()

    df_local = df.copy()
    df_local["experience_level"] = (
        df_local.get("experience_level", pd.Series(["unknown"] * len(df_local)))
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    token_re = re.compile(r"\b\w{3,}\b", flags=re.UNICODE)

    def extract_words(series: pd.Series) -> pd.Series:
        return series.dropna().astype(str).apply(lambda s: token_re.findall(s.lower()))

    def counter_from_series_of_lists(series_of_lists: pd.Series) -> Counter:
        if series_of_lists.empty:
            return Counter()
        exploded = series_of_lists.explode().dropna().astype(str)
        return Counter(exploded.values)

    entry_lists = extract_words(df_local.loc[df_local["experience_level"] == "entry", "hard_skills"])
    adv_lists = extract_words(df_local.loc[df_local["experience_level"] == "advanced", "hard_skills"])

    counter_entry = counter_from_series_of_lists(entry_lists)
    counter_adv = counter_from_series_of_lists(adv_lists)

    top_entry = [w for w, _ in counter_entry.most_common(top_n)]
    top_adv = [w for w, _ in counter_adv.most_common(top_n)]
    words_union = list(dict.fromkeys(top_entry + top_adv))

    if not words_union:
        log("No hard skill words found for Entry or Advanced.")
        Path(out_file).write_text("\n".join(lines), encoding="utf-8")
        return pd.DataFrame()

    rows = []
    for w in words_union:
        e_cnt = counter_entry.get(w, 0)
        a_cnt = counter_adv.get(w, 0)
        total = e_cnt + a_cnt
        rows.append({"word": w, "level": "entry", "count": e_cnt, "total": total})
        rows.append({"word": w, "level": "advanced", "count": a_cnt, "total": total})

    df_plot = pd.DataFrame(rows)

    log("\nWord\tEntry\tAdvanced\tTotal")
    for w in words_union:
        log(
            f"{w}\t{counter_entry.get(w,0)}\t"
            f"{counter_adv.get(w,0)}\t"
            f"{counter_entry.get(w,0)+counter_adv.get(w,0)}"
        )
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")

    plt.figure(figsize=(10, max(4, 0.35 * len(words_union))))
    palette_map = {
        "entry": sns.color_palette("Oranges", 3)[2],
        "advanced": sns.color_palette("Greens", 3)[2]
    }

    ax = sns.barplot(
        data=df_plot,
        x="count",
        y="word",
        hue="level",
        palette=palette_map,
        dodge=True
    )

    leg = ax.legend(title="Experience level", loc="upper right")
    for text in leg.get_texts():
        text.set_fontsize(8)
    leg.get_frame().set_alpha(0.9)

    totals = df_plot.groupby("word")["count"].sum().to_dict()

    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    ytick_positions = [t.get_position()[1] for t in ax.get_yticklabels()]

    for p in ax.patches:
        width = p.get_width()
        y_center = p.get_y() + p.get_height() / 2.0

        if ytick_positions:
            idx_nearest = int(np.argmin([abs(y_center - yp) for yp in ytick_positions]))
            word_label = ytick_labels[idx_nearest]
        else:
            word_label = ""

        total_for_word = totals.get(word_label, 0)
        pct = (width / total_for_word * 100) if total_for_word > 0 else 0.0
        label_text = f"{int(width)}  ({pct:.0f}%)"
        ax.annotate(
            label_text,
            (width, y_center),
            ha="left",
            va="center",
            fontsize=8,
            xytext=(6, 0),
            textcoords="offset points"
        )

    plt.title("Top repeated HARD-skill words: Entry vs Advanced (count and % of word total)")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.tight_layout()

    fig = ax.get_figure()
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df_plot


# ------------------------------------------------------------
# 7.2 Repeated SOFT-skill words by experience level
# ------------------------------------------------------------

def repeated_soft_skill_words_by_level(
    df: pd.DataFrame,
    top_n: int = 30,
    out_dir = OUT_DIR / "analytics"
):
    """
    Compares repeated words in soft_skills for entry vs advanced levels.
    Returns a long-format DataFrame: word, level, count, total.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot = out_dir / "12_repeated_SOFT_skill_words_entry_vs_advanced.png"
    out_file = out_dir / "12_repeated_SOFT_skill_words_entry_vs_advanced.txt"

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("\n=== Repeated SOFT skill words by experience level ===")

    if "soft_skills" not in df.columns:
        log("Column 'soft_skills' not found.")
        Path(out_file).write_text("\n".join(lines), encoding="utf-8")
        return pd.DataFrame()

    df_local = df.copy()
    df_local["experience_level"] = (
        df_local.get("experience_level", pd.Series(["unknown"] * len(df_local)))
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    token_re = re.compile(r"\b\w{3,}\b", flags=re.UNICODE)

    def extract_words(series: pd.Series) -> pd.Series:
        return series.dropna().astype(str).apply(lambda s: token_re.findall(s.lower()))

    def counter_from_series_of_lists(series_of_lists: pd.Series) -> Counter:
        if series_of_lists.empty:
            return Counter()
        exploded = series_of_lists.explode().dropna().astype(str)
        return Counter(exploded.values)

    entry_lists = extract_words(df_local.loc[df_local["experience_level"] == "entry", "soft_skills"])
    adv_lists = extract_words(df_local.loc[df_local["experience_level"] == "advanced", "soft_skills"])

    counter_entry = counter_from_series_of_lists(entry_lists)
    counter_adv = counter_from_series_of_lists(adv_lists)

    top_entry = [w for w, _ in counter_entry.most_common(top_n)]
    top_adv = [w for w, _ in counter_adv.most_common(top_n)]
    words_union = list(dict.fromkeys(top_entry + top_adv))

    if not words_union:
        log("No soft skill words found for Entry or Advanced.")
        Path(out_file).write_text("\n".join(lines), encoding="utf-8")
        return pd.DataFrame()

    rows = []
    for w in words_union:
        e_cnt = counter_entry.get(w, 0)
        a_cnt = counter_adv.get(w, 0)
        total = e_cnt + a_cnt
        rows.append({"word": w, "level": "entry", "count": e_cnt, "total": total})
        rows.append({"word": w, "level": "advanced", "count": a_cnt, "total": total})

    df_plot = pd.DataFrame(rows)

    log("\nWord\tEntry\tAdvanced\tTotal")
    for w in words_union:
        log(
            f"{w}\t{counter_entry.get(w,0)}\t"
            f"{counter_adv.get(w,0)}\t"
            f"{counter_entry.get(w,0)+counter_adv.get(w,0)}"
        )
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")

    plt.figure(figsize=(10, max(4, 0.35 * len(words_union))))
    palette_map = {
        "entry": sns.color_palette("Oranges", 3)[1],
        "advanced": sns.color_palette("Greens", 3)[1]
    }

    ax = sns.barplot(
        data=df_plot,
        x="count",
        y="word",
        hue="level",
        palette=palette_map,
        dodge=True
    )

    leg = ax.legend(title="Experience level", loc="upper right")
    for text in leg.get_texts():
        text.set_fontsize(8)
    leg.get_frame().set_alpha(0.9)

    totals = df_plot.groupby("word")["count"].sum().to_dict()

    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    ytick_positions = [t.get_position()[1] for t in ax.get_yticklabels()]

    for p in ax.patches:
        width = p.get_width()
        y_center = p.get_y() + p.get_height() / 2.0

        if ytick_positions:
            idx_nearest = int(np.argmin([abs(y_center - yp) for yp in ytick_positions]))
            word_label = ytick_labels[idx_nearest]
        else:
            word_label = ""

        total_for_word = totals.get(word_label, 0)
        pct = (width / total_for_word * 100) if total_for_word > 0 else 0.0
        label_text = f"{int(width)}  ({pct:.0f}%)"
        ax.annotate(
            label_text,
            (width, y_center),
            ha="left",
            va="center",
            fontsize=8,
            xytext=(6, 0),
            textcoords="offset points"
        )

    plt.title("Top repeated SOFT-skill words: Entry vs Advanced (count and % of word total)")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.tight_layout()

    fig = ax.get_figure()
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df_plot


# ------------------------------------------------------------
# 8. Semantic clustering of skills (HARD / SOFT)
# ------------------------------------------------------------

def skill_semantic_clusters(
    df: pd.DataFrame,
    skill_type: str = "hard_skills",
    clusters: int = 5,
    out_dir = OUT_DIR / "analytics"
):
    """
    Performs semantic clustering of skills (HARD or SOFT)
    using TF-IDF and KMeans.

    Prints clusters to console and saves them
    to a text file in the analytics folder.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"13_skill_semantic_CLUSTERS_{skill_type}.txt"

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log(f"\n=== Semantic grouping of skills ({skill_type}) ===")

    skills_list = []
    for skills in df[skill_type].dropna():
        for s in skills.split(","):
            s = s.strip()
            if len(s) > 1:
                skills_list.append(s)

    skills_list = list(set(skills_list))  # unique skills

    if len(skills_list) < clusters:
        log("Not enough skills for clustering")
        out_file.write_text("\n".join(lines), encoding="utf-8")
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(skills_list)

    kmeans = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init="auto"
    )
    labels = kmeans.fit_predict(X)

    df_clusters = pd.DataFrame({
        "skill": skills_list,
        "cluster": labels
    })

    for c in range(clusters):
        log(f"\n--- Cluster {c} ---")
        cluster_items = df_clusters[df_clusters["cluster"] == c]["skill"].tolist()
        for item in cluster_items[:20]:
            log(f" - {item}")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    return df_clusters


# ------------------------------------------------------------
# Main analytics entry point
# ------------------------------------------------------------

def run_analytics(path: str):
    """
    Full analytics pipeline on the exported XLSX:
    - loads data
    - normalizes experience_level
    - experience distribution
    - search term distribution
    - monthly dynamics
    - top HARD and SOFT skills
    - geo distributions (overall + by experience)
    - title clusters
    - repeated HARD/SOFT skill words
    - semantic clustering of HARD and SOFT skills
    """
    df = load_data(path)

    df["experience_level"] = (
        df.get("experience_level", pd.Series(["unknown"] * len(df)))
          .fillna("unknown")
          .astype(str)
          .str.strip()
          .str.lower()
    )

    experience_distribution(df)
    search_term_distribution(df)
    monthly_dynamics(df)

    # HARD skills
    plot_top_skills_by_experience(
        df=df,
        column="hard_skills",
        title_prefix="HARD",
        palette=None,
        top_n=20
    )

    # SOFT skills
    plot_top_skills_by_experience(
        df=df,
        column="soft_skills",
        title_prefix="SOFT",
        palette=None,
        top_n=20
    )

    # Geography – overall market
    city_distribution(df)
    state_distribution(df)

    # Geography – Entry and Advanced
    geo_by_experience(df, "entry")
    geo_by_experience(df, "advanced")

    # Title clusters
    title_clusters_by_experience(df, clusters_per_group=5, out_dir=OUT_DIR / "analytics")

    # Repeated skill words
    repeated_hard_df = repeated_hard_skill_words_by_level(df, top_n=30)
    repeated_soft_df = repeated_soft_skill_words_by_level(df, top_n=30)

    # Semantic skill clusters
    skill_semantic_clusters(df, "hard_skills", clusters=6)
    skill_semantic_clusters(df, "soft_skills", clusters=6)

    return df

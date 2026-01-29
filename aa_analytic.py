"""
aa_analytic.py
-----------------
Аналитика вакансий Arbeitsagentur.
Функции:
1. Распределение вакансий по уровням опыта (с графиком)
2. Распределение вакансий по поисковым тайтлам (с графиком)
3. Динамика вакансий по месяцам (с графиком)
4. Top HARD-skills (с графиком)
5. Top SOFT-skills (с графиком)
6. Кластеры в названиях вакансий (по search_term)
7. Повторяющиеся навыки в описаниях по уровням опыта
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

sns.set(style="whitegrid")  # стиль сетки на графиках


# ------------------------------------------------------------
# Загрузка данных
# ------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, parse_dates=["posted_date"])
    return df


# ------------------------------------------------------------
# Определение палитр из n цветов
# ------------------------------------------------------------
def palette_for_n(palette_name, n):
    """
    Возвращает список цветов длиной n.
    palette_name может быть строкой (имя палитры) или уже списком цветов.
    Безопасный fallback — tab10.
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
# 1. Распределение вакансий по уровням опыта
# ------------------------------------------------------------

def experience_distribution(df: pd.DataFrame):
    print("\n=== Arbeitsagentur: Verteilung der offenen Stellen nach Erfahrungsstufen ===")

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

    # Цвета по категориям
    color_map = {
        "entry": "#d65f36", # насыщенный оранжево‑красный
        "advanced": "#2ca02c", # зелёный
    }
    colors = [color_map.get(cat, "#b0b0b0") for cat in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        textprops={"fontsize": 14}
    )

    ax.axis("equal")
    plt.title("Verteilung nach Erfahrungsniveau – Anzahl und Prozentsatz", fontsize=16)
    save_plot("01_AA_experience_distribution")
    plt.show()

# ------------------------------------------------------------
# 2. Распределение вакансий по поисковым тайтлам
# ------------------------------------------------------------

def search_term_distribution(df: pd.DataFrame):
    """
    Строит распределение вакансий по поисковым тайтлам:
    1) общий график по всем вакансиям
    2) отдельные графики по уровням опыта (entry, advanced)
    """

    print("\n=== Arbeitsagentur: Verteilung der offenen Stellen nach Suchbegriffen ===")

    # -----------------------------
    # 1. Общий график по всем вакансиям
    # -----------------------------
    counts_all = df["search_term"].value_counts()

    print("\n--- Gesamt (alle Erfahrungsstufen) ---")
    print(counts_all)

    plt.figure(figsize=(10, 5))
    pal = palette_for_n("Blues_r", len(counts_all))
    df_plot_all = pd.DataFrame({
           "term": counts_all.index.astype(str),
           "value": counts_all.values
    })
    ax = sns.barplot(data=df_plot_all, y="value", x="term", hue="term", palette=pal, dodge=False)
    add_value_labels(ax)

    plt.title("Arbeitsagentur: Verteilung der offenen Stellen nach Suchbegriffen (All)")
    plt.xlabel("Suchbegriff")
    plt.ylabel("Stellenanzahl")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    save_plot("02_AA_search_term_distribution_all")

    # -----------------------------
    # 2. Графики по уровням опыта
    # -----------------------------
    # Нормализуем уровни для корректного сравнения
    exp_levels = df["experience_level"].fillna("unknown").astype(str).str.strip().unique()

    for level in exp_levels:
        subset = df[df["experience_level"].astype(str).str.strip() == level]

        if subset.empty:
            continue

        counts = subset["search_term"].value_counts()

        print(f"\n--- Erfahrungsstufe: {level} ---")
        print(counts)

        plt.figure(figsize=(10, 5))
        lvl = str(level).strip().lower()
        if "entry" in lvl or "junior" in lvl:
            pal = palette_for_n("Oranges_r", len(counts))  # тёплая градиентная палитра
        else:
            pal = palette_for_n("Greens_r", len(counts))  # холодная градиентная палитра
        df_plot_lvl = pd.DataFrame({
            "term": counts.index.astype(str),
            "value": counts.values
        })
        ax = sns.barplot(data=df_plot_lvl, y="value", x="term", hue="term", palette=pal, dodge=False)
        add_value_labels(ax)

        plt.title(f"Arbeitsagentur: Verteilung der offenen Stellen nach Suchbegriffen für Erfahrungsniveau: ({level.upper()})")
        plt.xlabel("Suchbegriff")
        plt.ylabel("Stellenanzahl")
        plt.xticks(rotation=45, ha="right", fontsize=8)

        save_plot(f"02_AA_search_term_distribution_{level}")


# ------------------------------------------------------------
# 3. Динамика вакансий по месяцам
# ------------------------------------------------------------

def monthly_dynamics(df: pd.DataFrame):
    """
    Строит динамику количества вакансий по месяцам:
    - общая линия по всем вакансиям
    - отдельные линии по уровням опыта (entry, advanced)
    - всё на одном графике с легендой
    """
    import matplotlib.dates as mdates

    print("\n=== Arbeitsagentur: Dynamik der offenen Stellen im Monatsverlauf ===")

    # Защита от отсутствующих колонок
    if "posted_date" not in df.columns:
        print("posted_date column not found")
        return

    # Копируем фрейм, чтобы не менять оригинал
    df_local = df.copy()

    # Нормализация experience_level (строчные значения)
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

    # Приводим posted_date к datetime, безопасно
    df_local["posted_date"] = pd.to_datetime(df_local["posted_date"], errors="coerce")

    # Убираем строки без даты
    df_local = df_local.dropna(subset=["posted_date"])
    if df_local.empty:
        print("Нет данных с корректными датами posted_date")
        return

    # Месяц как период
    df_local["month"] = df_local["posted_date"].dt.to_period("M")

    # Общая динамика
    counts_all = df_local.groupby("month").size().sort_index()
    if counts_all.empty:
        print("Нет агрегированных данных по месяцам")
        return
    months_all = counts_all.index.to_timestamp()

    # Уровни опыта (в нижнем регистре)
    exp_levels = sorted(df_local["experience_level"].dropna().unique())

    # Цвета для линий (конкретные цветные значения, не имена палитр)
    colors = {
        "all": "#1f77b4",       # blue
        "entry": "#ff8c00",     # darkorange
        "advanced": "#2e8b57",  # seagreen
        "unknown": "#7f7f7f"    # grey
    }

    plt.figure(figsize=(12, 6))

    # --- Линия всех вакансий ---
    sns.lineplot(
        x=months_all,
        y=counts_all.values,
        marker="o",
        color=colors["all"],
        label="Alle"
    )

    # Подписи значений для общей линии (контрастный цвет текста)
    text_color = "#222222"
    for x, y in zip(months_all, counts_all.values):
        plt.text(x, y, str(int(y)), fontsize=8, ha="center", va="bottom", color=text_color)

    # --- Линии по уровням опыта ---
    for level in exp_levels:
        subset = df_local[df_local["experience_level"] == level]
        if subset.empty:
            continue

        counts = subset.groupby("month").size().sort_index()
        if counts.empty:
            continue
        months = counts.index.to_timestamp()

        # Выбираем цвет: если уровень содержит "entry" или "junior" — entry, иначе advanced/unknown
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

    # Оформление графика
    plt.title("Arbeitsagentur: Dynamik der offenen Stellen im Monatsverlauf (All + Erfahrungsstufen)")
    plt.xlabel("Monat")
    plt.ylabel("Stellenanzahl")

    # Форматирование оси X для дат
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    plt.xticks(rotation=45, ha="right", fontsize=8)

    plt.legend(title="Erfahrungsniveau")
    plt.tight_layout()

    save_plot("03_AA_monthly_dynamics_combined")


# ------------------------------------------------------------
# 4-5. Top HARD и SOFT-skills по уровням опыта
# ------------------------------------------------------------

def plot_top_skills_by_experience(df: pd.DataFrame,
                                  column: str,
                                  title_prefix: str,
                                  palette: str = None,
                                  top_n: int = 20):
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe.")
        return

    # Автоматически определяем уровни опыта (сохраняем оригинальный регистр для заголовков)
    exp_levels = sorted(df["experience_level"].fillna("unknown").astype(str).unique())

    print(f"\nНайдены уровни опыта: {exp_levels}")

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
            print("Нет skills для этого уровня.")
            continue

        top = counter.most_common(top_n)

        for skill, count in top:
            print(f"{skill}: {count}")

        skills, counts = zip(*top)
        n = len(skills)

        # Жёсткое правило: если palette явно не задана, выбираем по уровню
        if palette:
            pal = palette_for_n(palette, n)
        else:
            lvl = str(level).strip().lower()
            if "entry" in lvl or "junior" in lvl:
                pal = palette_for_n("Oranges_r", n)   # тёплая градиентная палитра
            else:
                pal = palette_for_n("Greens_r", n)     # холодная градиентная палитра

        # plt.figure(figsize=(10, 6))
        # ax = sns.barplot(
        #     x=list(counts),
        #     y=list(skills),
        #     palette=pal,
        #     hue=counts.index,
        #     dodge=False
        # )
        #
        # add_value_labels(ax)
        plt.figure(figsize=(10, 6))
        df_plot_skills = pd.DataFrame({
            "skill": list(skills),
            "count": list(counts)
        })
        ax = sns.barplot(data=df_plot_skills, x="count", y="skill", hue="skill", palette=pal, dodge=False)

        if ax.get_legend() is not None:
            ax.get_legend().remove()
            add_value_labels(ax)

        plt.title(f"Arbeitsagentur: Top {top_n} {title_prefix}-skills ({level})")
        plt.xlabel("Anzahl")
        plt.ylabel("Fähigkeit")

        fname = f"04_AA_top_{title_prefix.lower()}_skills_{level}"
        if "hard" not in title_prefix.lower():
            fname = f"05_AA_top_{title_prefix.lower()}_skills_{level}"
        save_plot(fname)


# ------------------------------------------------------------
# 6. Кластеры в названиях вакансий (по search_term)
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
    terms = vectorizer.get_feature_names_out()
    weights = centroid.ravel()
    idx = np.argsort(weights)[::-1][:n]
    return [terms[i] for i in idx if weights[i] > 0]


# def title_clusters_by_experience(
#     df: pd.DataFrame,
#     clusters_per_group=5,
#     out_dir=OUT_DIR / "analytics"
# ):
#     """
#     Кластеризует названия вакансий (job_title) по search_term
#     отдельно для уровней опыта Entry и Advanced.
#
#     Использует TF-IDF + KMeans, выполняет очистку названий,
#     выводит репрезентативные примеры и ключевые термы кластеров.
#     Результаты сохраняются в текстовые файлы в папке analytics
#     и дублируются в консоль.
#     """
#
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     for level in ["entry", "advanced"]:
#         df_lvl = df[df["experience_level"].fillna("").astype(str) == level]
#
#         out_file = out_dir / f"11_title_clusters_{level.upper()}.txt"
#         lines = []
#
#         def log(s=""):
#             print(s)
#             lines.append(s)
#
#         log(f"\n=== Title clusters for experience level: {level.upper()} ===")
#
#         for term, group in df_lvl.groupby("search_term"):
#             raw_titles = group["job_title"].dropna().astype(str).tolist()
#             cleaned = [(t, clean_job_title(t)) for t in raw_titles]
#             cleaned = [(r, c) for r, c in cleaned if c]
#
#             if len(cleaned) < clusters_per_group:
#                 log(f"\n[ {term} ] — unzureichende Daten")
#                 continue
#
#             titles_raw = [x[0] for x in cleaned]
#             titles_clean = [x[1] for x in cleaned]
#
#             log(f"\n[ {term} ] — Clustering (n={len(titles_clean)})")
#
#             vectorizer = TfidfVectorizer(
#                 stop_words=GERMAN_STOPWORDS,
#                 ngram_range=(1, 2),
#                 min_df=2
#             )
#
#             X = vectorizer.fit_transform(titles_clean)
#             if X.shape[1] == 0:
#                 log("  ⚠ Nach der Reinigung keine Spuren mehr sichtbar")
#                 continue
#
#             k = min(clusters_per_group, X.shape[0])
#             kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
#             labels = kmeans.fit_predict(X)
#
#             df_term = pd.DataFrame({
#                 "title": titles_raw,
#                 "cluster": labels
#             })
#
#             centers = kmeans.cluster_centers_
#
#             for c in range(k):
#                 cluster_titles = df_term[df_term["cluster"] == c]["title"].tolist()
#                 if not cluster_titles:
#                     continue
#
#                 terms = top_terms(vectorizer, centers[c])
#
#                 log(f"\n Cluster {c} (n={len(cluster_titles)})")
#                 log("    Beispiele:")
#                 for t in cluster_titles[:5]:
#                     log(f"    - {t}")
#                 log(f"   Erläuterung: {', '.join(terms)}")
#
#         out_file.write_text("\n".join(lines), encoding="utf-8")

def title_clusters_by_experience(
    df: pd.DataFrame,
    clusters_per_group=5,
    out_dir=OUT_DIR / "analytics"
):
    """
    Кластеризует названия вакансий (job_title) по search_term
    отдельно для уровней опыта Entry и Advanced.

    Использует TF-IDF + KMeans, выполняет очистку названий,
    выводит репрезентативные примеры и ключевые термы кластеров.
    Результаты сохраняются в текстовые файлы в папке analytics
    и дублируются в консоль.

    Для каждого репрезентативного примера теперь выводятся:
      - ключевые термы кластера (cluster terms)
      - beruf (search_term), т.е. профессиональная метка группы
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
            beruf = term  # beruf — профессиональная метка (search_term)
            raw_titles = group["job_title"].dropna().astype(str).tolist()
            cleaned = [(t, clean_job_title(t)) for t in raw_titles]
            cleaned = [(r, c) for r, c in cleaned if c]

            if len(cleaned) < clusters_per_group:
                log(f"\n[ {beruf} ] — unzureichende Daten")
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
                log("  ⚠ Nach der Reinigung keine Spuren mehr sichtbar")
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

                # ключевые термы для данного центра
                terms = top_terms(vectorizer, centers[c])
                terms_str = ", ".join(terms) if terms else "(keine Terme)"

                log(f"\n Cluster {c} (n={len(cluster_titles)})")
                log(f"   Beruf: {beruf}")
                log(f"   Schlüsselterme: {terms_str}")
                log("    Beispiele:")

                # Для каждого репрезентативного примера выводим сам заголовок, ключевые термы и beruf
                for t in cluster_titles[:5]:
                    log(f"    - {t}")
                    #log(f"       -> Cluster-Terme: {terms_str}")
                    #log(f"       -> Beruf: {beruf}")

        out_file.write_text("\n".join(lines), encoding="utf-8")



# ------------------------------------------------------------
#  7.1 Повторяющиеся слова в HARD‑skills
# ------------------------------------------------------------

def repeated_hard_skill_words_by_level(
    df: pd.DataFrame,
    top_n: int = 30,
    out_dir = OUT_DIR / "analytics"
):
    """
    Сравнение повторяющихся слов в hard_skills для уровней entry и advanced.
    Возвращает DataFrame в длинном формате: word, level, count, total.
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

    # Нормализация experience_level
    df_local = df.copy()
    df_local["experience_level"] = (
        df_local.get("experience_level", pd.Series(["unknown"] * len(df_local)))
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Вспомогательная функция: извлечь слова длиной >=3 из строки
    token_re = re.compile(r"\b\w{3,}\b", flags=re.UNICODE)

    def extract_words(series: pd.Series) -> pd.Series:
        """Возвращает Series списков слов для каждой строки (split по non-word)."""
        return series.dropna().astype(str).apply(lambda s: token_re.findall(s.lower()))

    # Собираем слова для каждого уровня и считаем частоты через value_counts
    def counter_from_series_of_lists(series_of_lists: pd.Series) -> Counter:
        # series_of_lists: Series[list[str]]
        if series_of_lists.empty:
            return Counter()
        exploded = series_of_lists.explode().dropna().astype(str)
        return Counter(exploded.values)

    entry_lists = extract_words(df_local.loc[df_local["experience_level"] == "entry", "hard_skills"])
    adv_lists = extract_words(df_local.loc[df_local["experience_level"] == "advanced", "hard_skills"])

    counter_entry = counter_from_series_of_lists(entry_lists)
    counter_adv = counter_from_series_of_lists(adv_lists)

    # Топы и объединение слов (с сохранением порядка)
    top_entry = [w for w, _ in counter_entry.most_common(top_n)]
    top_adv = [w for w, _ in counter_adv.most_common(top_n)]
    words_union = list(dict.fromkeys(top_entry + top_adv))

    if not words_union:
        log("No hard skill words found for Entry or Advanced.")
        Path(out_file).write_text("\n".join(lines), encoding="utf-8")
        return pd.DataFrame()

    # Сбор данных в длинном формате
    rows = []
    for w in words_union:
        e_cnt = counter_entry.get(w, 0)
        a_cnt = counter_adv.get(w, 0)
        total = e_cnt + a_cnt
        rows.append({"word": w, "level": "entry", "count": e_cnt, "total": total})
        rows.append({"word": w, "level": "advanced", "count": a_cnt, "total": total})

    df_plot = pd.DataFrame(rows)

    # Логирование таблицы
    log("\nWord\tEntry\tAdvanced\tTotal")
    for w in words_union:
        log(f"{w}\t{counter_entry.get(w,0)}\t{counter_adv.get(w,0)}\t{counter_entry.get(w,0)+counter_adv.get(w,0)}")
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")

    # --- Построение графика ---
    plt.figure(figsize=(10, max(4, 0.35 * len(words_union))))
    palette_map = {"entry": sns.color_palette("Oranges", 3)[2], "advanced": sns.color_palette("Greens", 3)[2]}

    ax = sns.barplot(
        data=df_plot,
        x="count",
        y="word",
        hue="level",
        palette=palette_map,
        dodge=True
    )

    # Легенда в правом нижнем углу внутри осей
    leg = ax.legend(title="Experience level", loc="upper right")
    for text in leg.get_texts():
        text.set_fontsize(8)
    leg.get_frame().set_alpha(0.9)

    # Подписи: абсолютное значение и процент от суммы по слову
    totals = df_plot.groupby("word")["count"].sum().to_dict()

    # Кэшируем метки и их позиции для быстрого поиска ближайшей метки по y
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

    # Сохранение PNG с dpi=300
    fig = ax.get_figure()
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df_plot


# ------------------------------------------------------------
#  7.2 Повторяющиеся слова в SOFT‑skills
# ------------------------------------------------------------

def repeated_soft_skill_words_by_level(
    df: pd.DataFrame,
    top_n: int = 30,
    out_dir = OUT_DIR / "analytics"
):
    """
    Сравнение повторяющихся слов в soft_skills для уровней entry и advanced.
    Возвращает DataFrame с колонками: word, level, count, total.
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

    # Нормализация experience_level
    df_local = df.copy()
    df_local["experience_level"] = df_local.get(
        "experience_level",
        pd.Series(["unknown"] * len(df_local))
    ).fillna("unknown").astype(str).str.strip().str.lower()

    def count_words_for_level(level):
        counter = Counter()
        subset = df_local[df_local["experience_level"] == level]
        for skills in subset["soft_skills"].dropna():
            for skill in skills.split(","):
                for w in skill.strip().lower().split():
                    if len(w) > 2:
                        counter[w] += 1
        return counter

    counter_entry = count_words_for_level("entry")
    counter_adv = count_words_for_level("advanced")

    # Топы и объединение слов
    top_entry = [w for w, _ in counter_entry.most_common(top_n)]
    top_adv = [w for w, _ in counter_adv.most_common(top_n)]
    words_union = list(dict.fromkeys(top_entry + top_adv))

    if not words_union:
        log("No soft skill words found for Entry or Advanced.")
        Path(out_file).write_text("\n".join(lines), encoding="utf-8")
        return pd.DataFrame()

    # Сбор данных
    data = []
    for w in words_union:
        e_cnt = counter_entry.get(w, 0)
        a_cnt = counter_adv.get(w, 0)
        total = e_cnt + a_cnt
        data.append({"word": w, "level": "entry", "count": e_cnt, "total": total})
        data.append({"word": w, "level": "advanced", "count": a_cnt, "total": total})

    df_plot = pd.DataFrame(data)

    # Логирование таблицы
    log("\nWord\tEntry\tAdvanced\tTotal")
    for w in words_union:
        log(f"{w}\t{counter_entry.get(w,0)}\t{counter_adv.get(w,0)}\t{counter_entry.get(w,0)+counter_adv.get(w,0)}")
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")

    # --- Построение графика ---
    plt.figure(figsize=(10, max(4, 0.35 * len(words_union))))
    palette_map = {"entry": sns.color_palette("Oranges", 3)[2], "advanced": sns.color_palette("Greens", 3)[2]}

    ax = sns.barplot(
        data=df_plot,
        x="count",
        y="word",
        hue="level",
        palette=palette_map,
        dodge=True
    )

    # Разместить легенду в правом нижнем углу внутри осей
    leg = ax.legend(title="Experience level", loc="lower right")
    # Настроим шрифт и прозрачность рамки для читаемости
    for text in leg.get_texts():
        text.set_fontsize(8)
    leg.get_frame().set_alpha(0.9)

    # Подписи: абсолютное значение и процент от суммы по слову
    totals = df_plot.groupby("word")["count"].sum().to_dict()

    # подпишем каждый бар (значение и процент)
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

    # Сохранение PNG с dpi=300
    fig = ax.get_figure()
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df_plot




# Кластеризация навыков по смыслу (semantic clustering)

def skill_semantic_clusters(
    df: pd.DataFrame,
    skill_type: str = "hard_skills",
    clusters: int = 5,
    out_dir = OUT_DIR / "analytics"
):
    """
    Выполняет семантическую кластеризацию навыков
    (HARD или SOFT) с использованием TF‑IDF и KMeans.

    Выводит кластеры в консоль и сохраняет результат
    в текстовый файл в папке analytics.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"13_skill_semantic_CLUSTERS_{skill_type}.txt"

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log(f"\n=== Semantische Gruppierung von Fähigkeiten ({skill_type}) ===")

    skills_list = []

    for skills in df[skill_type].dropna():
        for s in skills.split(","):
            s = s.strip()
            if len(s) > 1:
                skills_list.append(s)

    skills_list = list(set(skills_list))  # уникальные навыки

    if len(skills_list) < clusters:
        log("Недостаточно навыков для кластеризации")
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
# Основной запуск
# ------------------------------------------------------------

def run_analytics(path: str):
    df = load_data(path)

    # Нормализуем experience_level для всего пайплайна аналитики
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

    # Вызов для HARD‑skills (palette не передаём — функция сама выберет по уровню)
    plot_top_skills_by_experience(
        df=df,
        column="hard_skills",
        title_prefix="HARD",
        palette=None,
        top_n=20
    )
    # Вызов для SOFT‑skills (palette не передаём — функция сама выберет по уровню)
    plot_top_skills_by_experience(
        df=df,
        column="soft_skills",
        title_prefix="SOFT",
        palette=None,
        top_n=20
    )

    # География — общий рынок
    city_distribution(df)
    state_distribution(df)

    # География — Entry и Advanced
    geo_by_experience(df, "Entry")
    geo_by_experience(df, "Advanced")

    title_clusters_by_experience(df, clusters_per_group=5, out_dir=OUT_DIR / "analytics")

    # repeated_hard_skill_words(df)
    # repeated_soft_skill_words(df)

    # Сравнительный график повторяющихся слов в HARD skills для Entry vs Advanced
    repeated_hard_df = repeated_hard_skill_words_by_level(df, top_n=30)
    # Сравнительный график повторяющихся слов в SOFT skills для Entry vs Advanced
    repeated_soft_df = repeated_soft_skill_words_by_level(df, top_n=30)

    skill_semantic_clusters(df, "hard_skills", clusters=6)
    skill_semantic_clusters(df, "soft_skills", clusters=6)

    return df

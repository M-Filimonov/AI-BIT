"""
aa_build_experience_model.py
--------------------
Периодическое обучение ML‑модели классификации опыта (entry / advanced)
на основе вакансий Arbeitsagentur.

ML‑модель классификации опыта (entry / advanced), обученная на вакансиях BA.
Пайплайн включает:
- загрузку исходного XLSX с вакансиями;
- очистку текста и нормализацию;
- объединение уровней junior/mid → entry, senior/lead → advanced;
- балансировку классов;
- обучение модели (TF‑IDF + LogisticRegression);
- оценку качества (accuracy ≈ 0.99, macro F1 ≈ 0.99);
- сохранение модели в AA_experience_model.joblib.

Модуль предназначен для периодического запуска и обновления модели,
используемой в aa_normalize.py для автоматической классификации опыта.
"""

import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
OUT_DIR: Path = Path("../AA_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH  = OUT_DIR / "AA_jobs_de_365_days.xlsx"     # исходный XLSX
MODEL_PATH = OUT_DIR / "AA_experience_model.joblib"      # куда сохранять модель
MIN_TEXT_LEN = 20                                     # минимальная длина описания
MAX_SAMPLES_PER_CLASS = 2000                          # балансировка


# ------------------------------------------------------------
# TEXT CLEANING
# ------------------------------------------------------------

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"[^a-zäöüß0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ------------------------------------------------------------
# LOAD & PREPARE DATASET
# ------------------------------------------------------------

def load_and_prepare_dataset() -> pd.DataFrame:
    print("Loading XLSX:", DATA_PATH)
    df = pd.read_excel(DATA_PATH)

    df_ml = df[["description_sections_text", "experience_level"]].copy()

    # Clean text
    df_ml["text"] = df_ml["description_sections_text"].fillna("").apply(clean_text)
    df_ml = df_ml[df_ml["text"].str.len() > MIN_TEXT_LEN]

    # Merge classes
    df_ml["level2"] = df_ml["experience_level"].str.lower().replace({
        "junior": "entry",
        "mid": "entry",
        "senior": "advanced",
        "lead": "advanced"
    })

    print("\nClass distribution BEFORE balancing:")
    print(df_ml["level2"].value_counts())

    return df_ml


# ------------------------------------------------------------
# BALANCE DATASET
# ------------------------------------------------------------

def balance_dataset(df_ml: pd.DataFrame) -> pd.DataFrame:
    entry = df_ml[df_ml["level2"] == "entry"]
    advanced = df_ml[df_ml["level2"] == "advanced"]

    target_entry = min(len(entry), MAX_SAMPLES_PER_CLASS)
    target_advanced = min(len(advanced), MAX_SAMPLES_PER_CLASS)

    entry_bal = resample(entry, replace=True, n_samples=target_entry, random_state=42)
    advanced_bal = resample(advanced, replace=True, n_samples=target_advanced, random_state=42)

    df_balanced = pd.concat([entry_bal, advanced_bal], ignore_index=True)

    print("\nClass distribution AFTER balancing:")
    print(df_balanced["level2"].value_counts())

    return df_balanced


# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

def train_model(df_balanced: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df_balanced["text"], df_balanced["level2"], test_size=0.2, random_state=42
    )

    print("\nTrain/Test sizes:")
    print("Train:", len(X_train), "Test:", len(X_test))

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(y_test, pred))

    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_test, pred))

    return model


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved as:", MODEL_PATH)


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------

def main():
    df_ml = load_and_prepare_dataset()
    df_balanced = balance_dataset(df_ml)
    model = train_model(df_balanced)
    save_model(model)


if __name__ == "__main__":
    main()

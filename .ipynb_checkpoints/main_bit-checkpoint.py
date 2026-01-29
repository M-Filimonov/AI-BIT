import os
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from typing import Optional
from openai import OpenAI
from difflib import get_close_matches
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import func_ec as mvf  # Project-specific functions
import importlib
importlib.reload(mvf)


# Загрузка переменных окружения из .env
load_dotenv()

# Получение API-ключа
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

from config_ec import DATA_FOLDER, REPORT_FOLDER, RESULT_FOLDER





def main() -> None:
    start_time = datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    #######################  XLSX loading and merged
    list_files = mvf.list_file_names()

    output_file = os.path.join(RESULT_FOLDER, "merged_all_jobs.xlsx")

    df_merged = mvf.load_and_merge_excel_files(list_files, DATA_FOLDER, output_file)
    print(df_merged)

    # === Task 01:  ===
    


    
    end_time = datetime.now()
    elapsed = end_time - start_time

    print(f"Script finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {elapsed}")

if __name__ == "__main__":
    main()

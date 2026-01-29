# Arbeitsagentur Job Analytics (BA)

A complete ETL and analytics pipeline for processing job vacancies from **Arbeitsagentur (BA)**.  
The project fetches, parses, normalizes, enriches, and analyzes job postings, producing structured datasets, visual reports, semantic clusters, and interactive geographic maps.

---

## Key Features

### Data Extraction & Parsing
- Fetches job descriptions from BA API and HTML pages  
- Extracts structured sections from `<h2>/<h3>`, `<strong>`, and colon‑based headers  
- Handles external partner job pages  
- Caches HTML responses for performance  

### Skill Extraction & Experience Classification
- Regex‑based hard/soft skill detection  
- Semantic scoring model for experience level  
- Title‑based heuristics  
- Unified mapping to `entry` / `advanced`  

### Geo‑Analytics
- Normalizes city names  
- Cached geocoding via Nominatim  
- Enriches vacancies with coordinates  
- Generates:
  - city‑level vacancy distribution  
  - state‑level vacancy distribution  
  - interactive geographic maps (Plotly)  

### Analytical Reports
- Vacancy distribution by experience level  
- Vacancy distribution by search terms  
- Monthly vacancy dynamics  
- Top HARD and SOFT skills  
- Repeated skill words (Entry vs Advanced)  
- Job title clustering (TF‑IDF + KMeans)  
- Semantic clustering of skills (HARD/SOFT)  

---

## Project Structure

```text
AA-BIT/
├── aa_main.py             # Main script
├── aa_config.py           # Global configuration, skill dictionaries, experience rules
├── aa_api_client.py       # Client for interacting with the Arbeitsagentur (BA) API
├── aa_html_parser.py      # HTML parsing, section extraction, external link handling
├── aa_skills.py           # Hard/soft skill extraction and experience detection
├── aa_normalize.py        # Full vacancy normalization pipeline
├── aa_location.py         # Geocoding, city/state analytics, interactive maps
├── aa_analytic.py         # All analytical charts, clustering, semantic grouping
└── aa_utils.py            # Plot saving, value labels, location parsing
```
---

## Installation
### 1. Clone the repository
- git clone https://github.com/M-Filimonov/AI-BIT.git
- cd <your-repo>

### Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. (Optional) Install ChromeDriver for Selenium fallback
pip install webdriver-manager

## Usage
1. Run the full project: **python aa_main.py**
2. BA dataset will automaticly prepared:
   all normalized BA vacancies exporting into - **AA_output/AA_jobs_de_365_days.xlsx**
3. View results
All charts, reports, and cluster files are saved to: **AA_output/analytics/**

##### This includes:
- experience distribution
- search term distribution
- monthly dynamics
- top skills
- job title clusters
- semantic skill clusters
- repeated skill word comparisons
- interactive geographic maps

## Example Output
######  Experience Distribution
    - Pie chart showing share of Entry vs Advanced roles.

###### Search Term Distribution
    Bar charts for:
    - all vacancies
    - per experience level

###### Monthly Dynamics
Trend lines for:
    - All vacancies
    - Entry
    - Advanced

###### Skill Analytics
    - Top 20 hard skills
    - Top 20 soft skills
    - Semantic clusters of skills
    - Repeated skill words (Entry vs Advanced)

###### Geo‑Analytics
- Top 25 cities
- Top 16 federal states
- Interactive map of vacancies across Germany (ALL /Advanced / Entry )

## Technologies Used
    - Python 3.10+
    - Pandas / NumPy — data processing
    - BeautifulSoup4 — HTML parsing
    - Requests — HTTP fetching
    - Scikit‑Learn — TF‑IDF, KMeans clustering
    - Matplotlib / Seaborn / Plotly — visualization
    - Geopy — geocoding
    - Selenium (fallback) — external job pages

## License
This project is licensed under the **MIT License**.
MIT License

Copyright (c) 2026 Mykhailo Filimonov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
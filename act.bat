@echo off
\.venv\Scripts\activate.bat
jupyter lab --ServerApp.iopub_data_rate_limit=10000000 --ServerApp.rate_limit_window=3
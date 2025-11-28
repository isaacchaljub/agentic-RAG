@echo off
REM Set UTF-8 encoding
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Activate venv
call .venv\Scripts\activate.bat

REM Start Streamlit
streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501

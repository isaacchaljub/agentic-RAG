@echo off
REM Set UTF-8 encoding
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Activate venv
call .venv\Scripts\activate.bat

REM Start FastAPI
fastapi run serving_api/main.py --host 0.0.0.0 --port 8000

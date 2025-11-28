@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Activate venv (call so control returns to this script)
call .venv\Scripts\activate.bat

REM Start FastAPI in background (no window, logs to file)
start "" /B cmd /c fastapi run serving_api/main.py --host 0.0.0.0 --port 8000 > fastapi.log 2>&1

REM Wait a moment for FastAPI to start
timeout /t 30 /nobreak >nul

REM Start Streamlit in background (no window, logs to file)
start "" /B cmd /c streamlit run app/main.py > streamlit.log 2>&1

REM Wait a moment for Streamlit to start
timeout /t 5 /nobreak >nul

REM Run ngrok in visible window (so you can copy the URL)
start "Ngrok - Copy HTTPS URL" cmd /k "ngrok http 8501"

echo FastAPI and Streamlit are running in the background.
echo Check fastapi.log and streamlit.log for output.
echo Ngrok window is open - copy the HTTPS URL from there.
pause
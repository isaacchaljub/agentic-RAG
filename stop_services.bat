@echo off
echo Stopping FastAPI, Streamlit, and ngrok processes...

REM Kill Python processes (FastAPI and Streamlit)
taskkill /F /IM python.exe /T 2>nul

REM Kill ngrok
taskkill /F /IM ngrok.exe /T 2>nul

echo Services stopped.
pause


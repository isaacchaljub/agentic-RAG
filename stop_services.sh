#!/bin/bash
echo "Stopping FastAPI, Streamlit, and ngrok processes..."

# Kill processes by name
pkill -f "fastapi.*serving_api/main.py" 2>/dev/null
pkill -f "streamlit.*app/main.py" 2>/dev/null
pkill -f "ngrok.*http.*8501" 2>/dev/null

# Alternative: kill all Python processes (more aggressive)
# pkill -f python 2>/dev/null

echo "Services stopped."


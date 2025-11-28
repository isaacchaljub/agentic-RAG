#!/bin/bash
# Set UTF-8 encoding
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

# Activate venv
source .venv/bin/activate

# Start FastAPI in background (logs to file)
nohup .venv/bin/fastapi run serving_api/main.py --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
sleep 30

# Start Streamlit in background (logs to file)
nohup .venv/bin/streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Wait a moment for Streamlit to start
sleep 3

# Run ngrok in background (logs to file)
nohup ngrok http 8501 > ngrok.log 2>&1 &
NGROK_PID=$!

echo "FastAPI (PID: $FASTAPI_PID) and Streamlit (PID: $STREAMLIT_PID) are running in the background."
echo "Check fastapi.log, streamlit.log, and ngrok.log for output."
echo "Ngrok is running (PID: $NGROK_PID) - check ngrok.log for the HTTPS URL."
echo ""
echo "To stop all services, run: kill $FASTAPI_PID $STREAMLIT_PID $NGROK_PID"
echo "Or use: pkill -f 'fastapi|streamlit|ngrok'"


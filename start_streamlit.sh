#!/bin/bash
# Set UTF-8 encoding
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

# Activate venv
source .venv/bin/activate

# Start Streamlit
streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501


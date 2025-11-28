#!/bin/bash
# Set UTF-8 encoding
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

# Activate venv
source .venv/bin/activate

# Start FastAPI
fastapi run serving_api/main.py --host 0.0.0.0 --port 8000


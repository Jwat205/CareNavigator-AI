#!/bin/bash
# Start FastAPI backend
WORKERS="${UVICORN_WORKERS:-$(python3 -c 'import os; print(min(os.cpu_count() or 1, 4))')}"
uvicorn api:app --host 0.0.0.0 --port 8000 --workers "$WORKERS" &
# Start Streamlit frontend
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

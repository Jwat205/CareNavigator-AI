#!/bin/bash
# Start FastAPI backend
uvicorn api:app --host 0.0.0.0 --port 8000 &
# Start Streamlit frontend
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

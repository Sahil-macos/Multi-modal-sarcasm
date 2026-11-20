@echo off
cd /d "%~dp0"
echo Starting Multimodal Sarcasm Detector on http://localhost:8000  (models load in the background, ~1-2 min first time)
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

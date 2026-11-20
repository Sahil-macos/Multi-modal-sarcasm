"""FastAPI backend for the multimodal sarcasm detection system.

Run:  python -m uvicorn backend.app:app --port 8000   (from the project root)
"""
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, pipeline
from .preprocess.input_handler import build_input, classify_file

app = FastAPI(title="Multimodal Sarcasm Detection API", version="1.0")
STARTED = time.time()


@app.on_event("startup")
def _warm():
    if os.environ.get("MSS_WARMUP", "1") == "1":
        threading.Thread(target=pipeline.registry.warmup, daemon=True).start()


@app.get("/api/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - STARTED),
            "models": pipeline.registry.status(),
            "fusion_weights": config.MODEL_WEIGHTS,
            "fused_threshold": config.FUSED_THRESHOLD}


def _analyze_sync(text, tmp_path, filename, transcribe, ocr):
    mm = build_input(text=text, file_path=tmp_path, filename=filename,
                     transcribe=transcribe, ocr=ocr)
    return pipeline.run(mm)


@app.post("/api/analyze")
async def analyze(text: Optional[str] = Form(None),
                  file: Optional[UploadFile] = File(None),
                  transcribe: bool = Form(True),
                  ocr: bool = Form(True)):
    has_file = file is not None and file.filename
    if not (text and text.strip()) and not has_file:
        raise HTTPException(400, "Provide text and/or a file.")

    tmp_dir = tmp_path = None
    try:
        if has_file:
            try:
                classify_file(file.filename)
            except ValueError as e:
                raise HTTPException(415, str(e))
            tmp_dir = tempfile.mkdtemp(prefix="mss_")
            # never trust the client filename for the path; keep only the extension
            tmp_path = Path(tmp_dir) / ("upload" + Path(file.filename).suffix.lower())
            size, limit = 0, config.MAX_UPLOAD_MB * 1024 * 1024
            with open(tmp_path, "wb") as f:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB")
                    f.write(chunk)
        try:
            return await run_in_threadpool(
                _analyze_sync, text, tmp_path, file.filename if has_file else None, transcribe, ocr)
        except ValueError as e:
            raise HTTPException(422, str(e))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import io
import os
import uuid
import threading
import zipfile
from pydantic import BaseModel
from pipeline import run_refine_job, replace_glyph, gpu_available, MODIFIED_FONTS_DIR

app = FastAPI()

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")


@app.get("/api/fonts")
def list_fonts():
    fonts = sorted([f for f in os.listdir(FONTS_DIR) if f.lower().endswith(".ttf")])
    versions = {f: int(os.path.getmtime(os.path.join(FONTS_DIR, f))) for f in fonts}
    return {"fonts": fonts, "versions": versions}


@app.get("/fonts/{filename}")
def serve_font(filename: str):
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400)
    path = os.path.join(FONTS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="font/ttf")


jobs: dict = {}


class RefineRequest(BaseModel):
    target_font: str
    target_cp: int
    example_font: str
    example_cp: int


@app.post("/api/refine")
def start_refine(req: RefineRequest):
    if not gpu_available():
        raise HTTPException(
            status_code=503,
            detail="GPU not available on this server. 'Refine by example' requires CUDA.",
        )
    for f in (req.target_font, req.example_font):
        if ".." in f or "/" in f:
            raise HTTPException(status_code=400)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "progress": 0, "result_font": None, "error": None}
    threading.Thread(
        target=run_refine_job,
        args=(job_id, jobs, req.target_font, req.target_cp, req.example_font, req.example_cp),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/api/refine/{job_id}")
def poll_refine(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404)
    return jobs[job_id]


@app.get("/fonts_modified/{filename}")
def serve_modified_font(filename: str):
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400)
    path = os.path.join(MODIFIED_FONTS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="font/ttf")


class ReplaceRequest(BaseModel):
    target_font: str
    target_cp: int
    example_font: str
    example_cp: int


@app.post("/api/replace")
def do_replace(req: ReplaceRequest):
    for f in (req.target_font, req.example_font):
        if ".." in f or "/" in f:
            raise HTTPException(status_code=400)
    result_font = replace_glyph(req.target_font, req.target_cp, req.example_font, req.example_cp)
    return {"result_font": result_font, "target_font": req.target_font}


class DownloadRequest(BaseModel):
    fonts: list[str]


@app.post("/api/download")
def download_fonts(req: DownloadRequest):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in req.fonts:
            if ".." in filename or "/" in filename:
                continue
            path = os.path.join(FONTS_DIR, filename)
            if os.path.isfile(path):
                zf.write(path, filename)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="hebrew_fonts.zip"'},
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")

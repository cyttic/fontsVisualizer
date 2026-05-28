from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import io
import os
import zipfile
from pydantic import BaseModel

app = FastAPI()

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")


@app.get("/api/fonts")
def list_fonts():
    fonts = sorted([f for f in os.listdir(FONTS_DIR) if f.lower().endswith(".ttf")])
    return {"fonts": fonts}


@app.get("/fonts/{filename}")
def serve_font(filename: str):
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400)
    path = os.path.join(FONTS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="font/ttf")


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

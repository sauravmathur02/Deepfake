"""
GenD face-forensics worker (runs inside venv_gend, started automatically by app.py).

POST /score  (multipart file: image or video)  ->  {"gend": float|None, "faces": int}
gend = mean P(fake) over face crops (8 frames for video, 1 for image).
"""
import io
import os
import sys
import tempfile

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "detectors", "GenD"))
from src.hf.modeling_gend import GenD  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = GenD.from_pretrained("yermandy/GenD_CLIP_L_14").to(DEVICE).eval()
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
app = FastAPI()


def _video_frames(data: bytes, suffix: str, n: int = 8):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(data)
        path = t.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out = []
        for i in np.linspace(0, max(total - 1, 0), n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                out.append(f)
        cap.release()
        return out
    finally:
        os.remove(path)


def _face_crop(bgr, margin=0.3):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(g, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    m = int(margin * max(w, h))
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, bgr.shape[1]), min(y + h + m, bgr.shape[0])
    return Image.fromarray(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}


@app.post("/score")
async def score(file: UploadFile = File(...)):
    data = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(VIDEO_EXT) or (file.content_type or "").startswith("video/"):
        frames = _video_frames(data, os.path.splitext(name)[1] or ".mp4")
    else:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        frames = [arr] if arr is not None else []
    crops = [c for c in (_face_crop(f) for f in frames) if c is not None]
    if not crops:
        return {"gend": None, "faces": 0}
    with torch.no_grad():
        x = torch.stack([model.feature_extractor.preprocess(c) for c in crops]).to(DEVICE)
        p = model(x).softmax(dim=-1)[:, 1].cpu().numpy()
    return {"gend": float(p.mean()), "faces": len(crops)}

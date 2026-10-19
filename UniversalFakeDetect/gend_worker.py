"""
GenD face-forensics worker (runs inside venv_gend, started automatically by app.py).

POST /score  (multipart file: image or video)  ->  {"gend": float|None, "faces": int, "box": list|None}
gend = P(fake) over detected face crops with robust multi-angle multi-face localization.
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "detectors", "GenD"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multi-angle face cascades (frontal + profile) for high recall across video motion
cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
app = FastAPI()

# Try loading GenD; fallback to active pixel detectors
gend_model = None
face_detector = None
try:
    from src.hf.modeling_gend import GenD
    ckpt_dir = os.path.join(HERE, "pretrained_weights", "GenD")
    if os.path.isdir(ckpt_dir):
        gend_model = GenD.from_pretrained(ckpt_dir).to(DEVICE).eval()
except Exception:
    gend_model = None

if gend_model is None:
    try:
        from detectors.pixel import CommForDetector, BFreeDetector
        face_detector_commfor = CommForDetector()
        face_detector_npr = BFreeDetector()
    except Exception as e:
        print(f"GenD worker detector initialization warning: {e}")
        face_detector_commfor = None
        face_detector_npr = None
else:
    face_detector_commfor = None
    face_detector_npr = None


def _video_frames(data: bytes, suffix: str, n: int = 12):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(data)
        path = t.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out = []
        if total > 0:
            indices = np.linspace(0, max(total - 1, 0), n).astype(int)
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if ok:
                    out.append(f)
        cap.release()
        return out
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _detect_faces_robust(gray):
    faces = cascade_frontal.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30))
    if len(faces) > 0:
        return faces

    if not cascade_profile.empty():
        faces = cascade_profile.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            return faces
        flipped = cv2.flip(gray, 1)
        faces_flipped = cascade_profile.detectMultiScale(flipped, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30))
        if len(faces_flipped) > 0:
            W = gray.shape[1]
            return [(W - x - w, y, w, h) for (x, y, w, h) in faces_flipped]

    return []


def _face_crops_all(bgr, margin=0.25):
    if bgr is None or bgr.size == 0:
        return []
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces_robust(g)
    if len(faces) == 0:
        return []
    H, W = bgr.shape[:2]
    out = []
    # Score up to top 2 faces per frame (handles dialogue scenes and interviews)
    sorted_faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[:2]
    filtered_faces = [sorted_faces[0]]
    if len(sorted_faces) > 1:
        primary_area = sorted_faces[0][2] * sorted_faces[0][3]
        sec_area = sorted_faces[1][2] * sorted_faces[1][3]
        if sec_area >= 0.12 * primary_area or (sorted_faces[1][2] >= 60 and sorted_faces[1][3] >= 60):
            filtered_faces.append(sorted_faces[1])

    for (x, y, w, h) in filtered_faces:
        m = int(margin * max(w, h))
        x0, y0 = max(x - m, 0), max(y - m, 0)
        x1, y1 = min(x + w + m, W), min(y + h + m, H)
        crop = Image.fromarray(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        box = [x / W, y / H, w / W, h / H]
        out.append((crop, box))
    return out


def _score_crops(crops):
    if not crops:
        return []
    if gend_model is not None:
        with torch.no_grad():
            x = torch.stack([gend_model.feature_extractor.preprocess(c) for c in crops]).to(DEVICE)
            probs = gend_model(x).softmax(dim=-1)[:, 1].cpu().numpy()
            return [float(p) for p in probs]

    c_scores = face_detector_commfor.predict(crops) if face_detector_commfor is not None else [0.0] * len(crops)
    n_scores = face_detector_npr.predict(crops) if face_detector_npr is not None else [None] * len(crops)

    combined = []
    for c, n in zip(c_scores, n_scores):
        if n is not None:
            if c > 0.45 or n > 0.45:
                combined.append(max(c, n))
            else:
                combined.append(0.50 * c + 0.50 * n)
        else:
            combined.append(float(c))
    return combined


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "gend": gend_model is not None,
        "commfor": face_detector_commfor is not None,
        "npr": face_detector_npr is not None
    }


@app.post("/score")
async def score(file: UploadFile = File(...)):
    data = await file.read()
    name = (file.filename or "").lower()
    is_video = name.endswith(VIDEO_EXT) or (file.content_type or "").startswith("video/")

    if is_video:
        frames = _video_frames(data, os.path.splitext(name)[1] or ".mp4", n=12)
    else:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        frames = [arr] if arr is not None else []

    crops = []
    boxes = []
    for f in frames:
        for crop, box in _face_crops_all(f):
            if crop.size[0] >= 36 and crop.size[1] >= 36:
                crops.append(crop)
                boxes.append(box)

    if not crops:
        return {"gend": None, "faces": 0, "box": None}

    probs = _score_crops(crops)
    if not probs:
        return {"gend": None, "faces": 0, "box": None}

    # Video multi-frame persistence: require sustained facial manipulation across multiple frames
    if is_video and len(probs) >= 4:
        sorted_probs = sorted(probs, reverse=True)
        if sorted_probs[1] > 0.40:
            top_k = max(2, int(len(sorted_probs) * 0.35))
            final_score = float(0.70 * np.mean(sorted_probs[:top_k]) + 0.30 * np.median(sorted_probs))
        else:
            final_score = float(0.25 * sorted_probs[0] + 0.75 * np.median(sorted_probs))
    elif is_video and len(probs) >= 2:
        sorted_probs = sorted(probs, reverse=True)
        final_score = float(0.50 * sorted_probs[0] + 0.50 * sorted_probs[1])
    else:
        final_score = float(np.mean(probs))

    best_idx = int(np.argmax(probs))
    return {"gend": round(final_score, 6), "faces": len(crops), "box": boxes[best_idx]}

import os
import io
import torch
import numpy as np
import warnings
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from PIL import Image

from detectors.provenance import check_provenance
from detectors.pixel import CommForDetector, BFreeDetector
from detectors.video import sample_video_bytes
from detectors.temporal import temporal_features_bytes
from detectors import calibration
from pydantic import BaseModel
from typing import List

warnings.filterwarnings("ignore")

from detectors import gend_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting GenD worker...")
    gend_client.start(timeout=60.0)
    yield
    print("Stopping GenD worker...")
    gend_client.stop()

app = FastAPI(title="Universal Synthetic Media Detector API", lifespan=lifespan)

print("Loading Ensemble Detectors...")
commfor_model = CommForDetector()
bfree_model = BFreeDetector()
print("Models ready.")

# FastAPI Routes
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return open("static/index.html", encoding="utf-8").read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        is_video = (
            file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
            or file.content_type.startswith('video/')
        )

        # 1. Provenance check (Metadata / C2PA)
        prov_result = check_provenance(contents)
        is_prov_fake = (prov_result.get("verdict") == "ai_declared")

        gend_score, faces_found = gend_client.score(file.filename, contents, file.content_type)


        temporal = None
        if is_video:
            try:
                temporal = temporal_features_bytes(contents)
            except Exception:
                temporal = None

        # 2. Pixel Models
        if is_video:
            frames = sample_video_bytes(contents, n_frames=8)
            if not frames:
                return {"success": False, "error": "Could not extract frames. Upload a valid MP4/AVI/MOV."}
            
            c_scores = commfor_model.predict(frames)
            b_scores = bfree_model.predict(frames)
            
            avg_c = float(np.mean(c_scores))
            avg_b = float(np.mean(b_scores))
            frames_analyzed = len(frames)
        else:
            try:
                img = Image.open(io.BytesIO(contents)).convert("RGB")
            except Exception as e:
                return {"success": False, "error": f"Cannot open image: {e}"}

            c_scores = commfor_model.predict([img])
            b_scores = bfree_model.predict([img])
            
            avg_c = float(c_scores[0])
            avg_b = float(b_scores[0])
            frames_analyzed = 1

        # Fixed ensemble weights by default, or loaded from user calibration
        cal = calibration.load()
        cal_p = calibration.apply(cal, avg_c, avg_b)
        if cal_p is not None:
            pixel_final, threshold = cal_p, 0.5
        else:
            if gend_score is not None:
                pixel_final, threshold = float(0.33 * avg_c + 0.33 * avg_b + 0.34 * gend_score), 0.30
            else:
                pixel_final, threshold = float(0.50 * avg_c + 0.50 * avg_b), 0.30

        if is_prov_fake:
            final_score = max(pixel_final, 0.99)
            prov_score = 1.0
        else:
            final_score = pixel_final
            prov_score = 0.0

        is_fake = final_score > threshold
        confidence = final_score if is_fake else (1.0 - final_score)

        return {
            "success": True,
            "prediction": "Fake" if is_fake else "Real",
            "score": round(final_score, 6),
            "confidence": f"{confidence * 100:.2f}%",
            "commfor_score": round(avg_c, 6),
            "bfree_score": round(avg_b, 6),
            "prov_score": round(prov_score, 6),
            "frames_analyzed": int(frames_analyzed),
            "temporal": temporal,
            "gend_score": gend_score,
            "faces_found": faces_found,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e) or type(e).__name__}


class Sample(BaseModel):
    commfor: float
    bfree: float
    is_fake: bool  # ground truth


@app.post("/feedback")
async def feedback(samples: List[Sample]):
    """Store labeled samples in a review queue (does NOT change live predictions)."""
    calibration.add_samples([s.dict() for s in samples])
    return {"ok": True, "stored": len(samples)}

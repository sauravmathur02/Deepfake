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

import base64
from PIL import ImageChops, ImageEnhance, ExifTags


def extract_exif(img):
    out = {"Format": str(img.format), "Dimensions": f"{img.width} x {img.height}", "Mode": img.mode}
    try:
        for k, v in (img.getexif() or {}).items():
            name = ExifTags.TAGS.get(k, str(k))
            if isinstance(v, bytes):
                continue
            out[name] = str(v)[:80]
    except Exception:
        pass
    for k, v in (getattr(img, "info", {}) or {}).items():
        if isinstance(v, str) and k.lower() in ("software", "parameters", "prompt", "comment", "description"):
            out[f"PNG:{k}"] = v[:120]
    return out


def make_ela(img, quality=90):
    """Error Level Analysis heatmap (base64 PNG) highlighting recompression inconsistencies."""
    im = img.copy()
    im.thumbnail((512, 512))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    diff = ImageChops.difference(im, Image.open(buf).convert("RGB"))
    ext = max(e[1] for e in diff.getextrema()) or 1
    diff = ImageEnhance.Brightness(diff).enhance(255.0 / ext)
    out = io.BytesIO()
    diff.save(out, "PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def face_region_ela(img, box, quality=90):
    """Ratio of mean ELA error inside the face box to outside it (None if unavailable)."""
    try:
        im = img.copy()
        im.thumbnail((512, 512))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        diff = np.asarray(ImageChops.difference(im, Image.open(buf).convert("RGB")).convert("L"), dtype=np.float32)
        H, W = diff.shape
        x, y, w, h = box
        x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
        mask = np.zeros_like(diff, dtype=bool)
        mask[y0:y1, x0:x1] = True
        if mask.sum() < 25 or (~mask).sum() < 25:
            return None
        return round(float(diff[mask].mean() / (diff[~mask].mean() + 1e-6)), 3)
    except Exception:
        return None


def build_explanation(is_fake, is_video, c, b, g, is_prov_fake, prov_result, faces_found, region=None):
    """Build a human-readable forensic explanation from per-model signals."""
    T = 0.30
    media = "video" if is_video else "image"
    findings = []
    category = "Authentic"

    if is_prov_fake:
        findings.append("Embedded metadata/C2PA credentials explicitly declare this file as AI-generated.")
    if g is not None and g > T:
        findings.append("The face-forensics model found facial inconsistencies (blending boundaries, texture or identity artifacts) typical of a face-swap or face-reenactment deepfake.")
    if c > T:
        findings.append("The Community Forensics model detected generator fingerprints in the pixel statistics, typical of fully AI-generated content (GAN/diffusion models such as Midjourney, DALL-E or Stable Diffusion).")
    if b > T:
        findings.append("The B-Free model found content-independent synthetic artifacts, which can also indicate AI inpainting or heavy manipulation/editing.")

    if is_fake:
        face_fake = g is not None and g > T
        pix_fake = c > T or b > T
        if is_prov_fake:
            category = "AI-Generated (declared)"
        elif face_fake and not pix_fake:
            category = "Deepfake (face manipulation)"
        elif face_fake and pix_fake:
            category = "Deepfake / AI-Generated"
        elif c > T and b <= T:
            category = "AI-Generated"
        elif b > T and c <= T:
            category = "Edited / Manipulated"
        else:
            category = "Synthetic or Manipulated"
        summary = f"This {media} is likely {category.lower()}."
    else:
        summary = f"This {media} appears authentic. No detector exceeded the manipulation threshold."
        findings.append("No generator fingerprints, face-manipulation artifacts, or AI provenance tags were found.")
        if faces_found and g is not None:
            findings.append("Detected faces looked consistent and natural.")

    if is_video:
        findings.append("Video verdict is averaged across sampled frames.")
    strong = max([c, b] + ([g] if g is not None else []))
    if is_fake and strong < 0.6 and not is_prov_fake:
        findings.append("Confidence is moderate; heavy compression or filters can mimic these signals, so treat with caution.")

    # Face-region analysis: face vs. rest-of-scene
    face_only = False
    if is_fake and g is not None and g <= T and max(c, b) > T and not is_prov_fake:
        findings.append("The face-swap model found no blending seams on the face. This does not make the face real: a face generated entirely by AI has no swap boundary and can pass face-swap checks, so the whole-image result takes priority.")
    if is_fake and g is not None and g > T and max(c, b) <= T and not is_prov_fake:
        face_only = True
        findings.insert(0, "Region analysis: the face region looks manipulated while the surrounding scene looks authentic \u2014 consistent with a replaced or swapped face on an otherwise real photo/video.")
    if region and region.get("ela_mismatch"):
        findings.append(f"Compression (ELA) in the face area differs from the background (ratio {region['ela_ratio']:.2f}), a classic sign of a pasted or regenerated face.")
        if is_fake and g is not None and g > T:
            face_only = True

    # Likelihood level from how far the strongest signal exceeds the threshold
    margin = (1.0 if is_prov_fake else strong) - T
    if not is_fake:
        likelihood = "Unlikely to be manipulated"
    elif margin > 0.40 or is_prov_fake:
        likelihood = "Very likely"
    elif margin > 0.20:
        likelihood = "Likely"
    else:
        likelihood = "Possible"

    if is_fake and face_only and not is_prov_fake:
        category = "Face Replaced / Swapped"
        summary = f"{likelihood}: a face in this {media} has most likely been replaced or manipulated by AI."
    elif is_fake:
        summary = f"{likelihood}: this {media} is {category.lower()}."
    return {"category": category, "summary": summary, "findings": findings, "likelihood": likelihood}


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

        gend_score, faces_found, face_box = gend_client.score(file.filename, contents, file.content_type)


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
            frame_scores = [round(float(0.5 * x + 0.5 * y), 4) for x, y in zip(c_scores, b_scores)]
            exif, ela = {"Type": "video", "Frames sampled": str(len(frames))}, None
        else:
            try:
                img = Image.open(io.BytesIO(contents))
                exif = extract_exif(img)
                img = img.convert("RGB")
                ela = make_ela(img)
            except Exception as e:
                return {"success": False, "error": f"Cannot open image: {e}"}

            c_scores = commfor_model.predict([img])
            b_scores = bfree_model.predict([img])
            
            avg_c = float(c_scores[0])
            avg_b = float(b_scores[0])
            frames_analyzed = 1
            frame_scores = []

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

        region = None
        if not is_video and face_box:
            ratio = face_region_ela(img, face_box)
            region = {"ela_ratio": ratio}
            region["ela_mismatch"] = bool(ratio is not None and (ratio > 1.6 or ratio < 0.6))

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
            "explanation": build_explanation(is_fake, is_video, avg_c, avg_b, gend_score, is_prov_fake, prov_result, faces_found, region),
            "media_type": "video" if is_video else "image",
            "frame_scores": frame_scores,
            "exif": exif,
            "ela": ela,
            "face_box": face_box if not is_video else None,
            "region": region,
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

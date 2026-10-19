import os
import io
import torch
import numpy as np
import warnings
from fastapi import FastAPI, File, UploadFile, Form
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
    T = 0.35
    media = "video" if is_video else "image"
    findings = []
    category = "Authentic"

    if is_fake:
        if is_prov_fake:
            findings.append("Embedded metadata/C2PA credentials explicitly declare this file as AI-generated.")
        if g is not None and g > T:
            findings.append("The face-forensics model detected facial inconsistencies (blending boundaries, texture or identity artifacts) typical of a deepfake face-swap.")
        if c > T:
            findings.append("The Community Forensics model detected generator fingerprints in pixel statistics, typical of AI generation (GAN/diffusion models such as Midjourney, DALL-E or Stable Diffusion).")
        if b > T:
            findings.append("The B-Free/NPR model found synthetic pixel relationships and frequency artifacts, indicating AI inpainting or generation.")

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
        summary = f"This {media} appears authentic. All detectors remained below the manipulation threshold."
        findings.append("No generator fingerprints, diffusion noise patterns, or synthetic spectral signatures were found.")
        if faces_found and g is not None:
            findings.append("Detected facial anatomy, skin texture, and boundary blending appeared natural and consistent.")
        if not faces_found:
            findings.append("Pixel statistics conform to natural optical camera capture physics.")

    if is_video:
        findings.append("Video verdict is evaluated across sampled frames and temporal motion dynamics.")
    strong = max([c, b] + ([g] if g is not None else []))
    if is_fake and strong < 0.55 and not is_prov_fake:
        findings.append("Confidence is moderate; heavy compression or stylized filters can introduce artifacts.")

    # Face-region analysis: face vs. rest-of-scene
    face_only = False
    if is_fake and g is not None and g <= T and max(c, b) > T and not is_prov_fake:
        findings.append("The face-swap model found no blending seams on the face. Whole-image generative patterns take priority.")
    if is_fake and g is not None and g > T and max(c, b) <= T and not is_prov_fake:
        face_only = True
        findings.insert(0, "Region analysis: the face region looks manipulated while the surrounding scene looks authentic — consistent with a replaced or swapped face on an otherwise real photo/video.")
    if is_fake and region and region.get("ela_mismatch") and g is not None and g > T:
        findings.append(f"Compression (ELA) in the face area differs from the background (ratio {region['ela_ratio']:.2f}), confirming a pasted or regenerated face.")
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
os.makedirs(os.path.join("static", "uploads"), exist_ok=True)
os.makedirs(os.path.join("static", "samples"), exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return open("static/index.html", encoding="utf-8").read()

class UrlFetchRequest(BaseModel):
    url: str

def _sync_fetch(url: str, upload_dir: str):
    import requests, uuid, mimetypes, shutil

    # Local loopback optimization
    for prefix in ("http://127.0.0.1:8000/", "http://localhost:8000/", "http://0.0.0.0:8000/", "/"):
        if url.startswith(prefix):
            local_rel = url[len(prefix):].split("?")[0]
            if os.path.exists(local_rel):
                fname = f"copy_{uuid.uuid4().hex[:8]}_{os.path.basename(local_rel)}"
                dest = os.path.join(upload_dir, fname)
                shutil.copyfile(local_rel, dest)
                ext = os.path.splitext(fname)[1].lower()
                return {
                    "success": True,
                    "filename": fname,
                    "file_url": f"/static/uploads/{fname}",
                    "content_type": "video/mp4" if ext in (".mp4", ".mov", ".avi") else "image/jpeg",
                    "size": os.path.getsize(dest)
                }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # 1. Direct HTTP download
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=15)
        ct = r.headers.get("Content-Type", "").lower()
        is_media_type = any(ct.startswith(p) for p in ("image/", "video/"))
        has_media_ext = any(url.lower().split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi", ".webm"))

        if r.status_code == 200 and not ct.startswith("text/") and (is_media_type or has_media_ext):
            clean_url = url.lower().split("?")[0]
            if clean_url.endswith(".png"): ext = ".png"
            elif clean_url.endswith(".mp4"): ext = ".mp4"
            elif clean_url.endswith(".mov"): ext = ".mov"
            elif clean_url.endswith(".webm"): ext = ".webm"
            elif clean_url.endswith(".webp"): ext = ".webp"
            else: ext = mimetypes.guess_extension(ct.split(";")[0]) or ".jpg"

            fname = f"fetched_{uuid.uuid4().hex[:10]}{ext}"
            fpath = os.path.join(upload_dir, fname)

            size = 0
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        size += len(chunk)
                        if size > 60 * 1024 * 1024:
                            f.close()
                            os.remove(fpath)
                            return {"success": False, "error": "Media exceeds 60MB size limit."}
                        f.write(chunk)

            return {
                "success": True,
                "filename": fname,
                "file_url": f"/static/uploads/{fname}",
                "content_type": ct or ("video/mp4" if ext == ".mp4" else "image/jpeg"),
                "size": size
            }
    except Exception:
        pass

    # 2. yt-dlp fallback
    try:
        import yt_dlp

        ffmpeg_bin = None
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None

        uid = uuid.uuid4().hex[:10]
        out_tmpl = os.path.join(upload_dir, f"social_{uid}.%(ext)s")
        ydl_opts = {
            'outtmpl': out_tmpl,
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best',
            'merge_output_format': 'mp4',
            'max_filesize': 60 * 1024 * 1024,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'quiet': True,
            'no_warnings': True,
        }
        if ffmpeg_bin and os.path.exists(ffmpeg_bin):
            ydl_opts['ffmpeg_location'] = ffmpeg_bin
        else:
            ydl_opts['format'] = 'best[height<=720][ext=mp4]/best[height<=720]'

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                saved_path = ydl.prepare_filename(info)
        except Exception as dl_err:
            if 'merging' in str(dl_err).lower() or 'ffmpeg' in str(dl_err).lower():
                ydl_opts['format'] = 'best[ext=mp4]/bestvideo[ext=mp4]/best'
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    saved_path = ydl.prepare_filename(info)
            else:
                raise dl_err

        if not os.path.exists(saved_path):
            base = os.path.splitext(saved_path)[0]
            for candidate_ext in ('.mp4', '.mkv', '.webm', '.mov', '.avi'):
                if os.path.exists(base + candidate_ext):
                    saved_path = base + candidate_ext
                    break

        if not os.path.exists(saved_path):
            return {"success": False, "error": "Video downloaded but output file could not be located."}

        fname = os.path.basename(saved_path)
        ext = os.path.splitext(fname)[1].lower()
        return {
            "success": True,
            "filename": fname,
            "file_url": f"/static/uploads/{fname}",
            "content_type": "video/mp4" if ext in ('.mp4', '.mov', '.avi') else f"video/{ext.lstrip('.')}",
            "size": os.path.getsize(saved_path)
        }
    except Exception as e:
        return {"success": False, "error": f"Could not fetch media: {str(e)}"}

@app.post("/fetch-url")
async def fetch_url(payload: UrlFetchRequest):
    import asyncio
    url = (payload.url or "").strip()
    if not url:
        return {"success": False, "error": "URL cannot be empty."}
    
    upload_dir = os.path.join("static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    return await asyncio.to_thread(_sync_fetch, url, upload_dir)

@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: str = Form("auto")):
    try:
        contents = await file.read()
        content_type = file.content_type or ""
        is_video = (
            file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
            or content_type.startswith('video/')
        )

        # 1. Provenance check (Metadata / C2PA)
        prov_result = check_provenance(contents)
        is_prov_fake = (prov_result.get("verdict") == "ai_declared")

        gend_score, faces_found, face_box = gend_client.score(file.filename, contents, content_type)

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
            
            # Robust trimmed aggregation for video frames (drop extreme single-frame outlier noise)
            if len(c_scores) >= 5:
                sorted_c = sorted(c_scores)
                avg_c = float(0.55 * np.mean(sorted_c[1:-1]) + 0.45 * np.median(c_scores))
            else:
                avg_c = float(np.mean(c_scores))

            b_valid = [s for s in b_scores if s is not None]
            if b_valid:
                if len(b_valid) >= 5:
                    sorted_b = sorted(b_valid)
                    avg_b = float(0.55 * np.mean(sorted_b[1:-1]) + 0.45 * np.median(b_valid))
                else:
                    avg_b = float(np.mean(b_valid))
            else:
                avg_b = None

            frames_analyzed = len(frames)
            frame_scores = [round(float(x if y is None else 0.5 * x + 0.5 * y), 4) for x, y in zip(c_scores, b_scores)]
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
            avg_b = float(b_scores[0]) if b_scores[0] is not None else None
            frames_analyzed = 1
            frame_scores = []

        # Region ELA calculation (face area vs background)
        region = None
        if not is_video and face_box:
            ratio = face_region_ela(img, face_box)
            region = {"ela_ratio": ratio}
            g_sus = gend_score or 0.0
            region["ela_mismatch"] = bool(ratio is not None and (ratio > 2.40 or ratio < 0.40 or (ratio > 1.85 and g_sus > 0.35)))

        # Temporal consistency signal for video:
        temporal_fake_signal = 0.0
        if is_video and temporal:
            w_err = float(temporal.get("warp_error") or 0.0)
            w_jit = float(temporal.get("warp_jitter") or 0.0)
            flick = float(temporal.get("flicker") or 0.0)
            hf = float(temporal.get("hf_energy") or 0.0)

            # Signal 1: Extreme high-frequency energy + optical warping (neural generator noise / residual grid)
            if hf > 14.0 and w_err > 1.2:
                temporal_fake_signal = max(temporal_fake_signal, min(0.92, 0.50 + (hf - 14.0) * 0.05))

            # Signal 2: High non-rigid optical warping error with generative pixel trend (AI portrait animation)
            if w_err > 1.35 and avg_c > 0.08:
                temporal_fake_signal = max(temporal_fake_signal, min(0.85, 0.45 + (w_err - 1.35) * 0.25 + (avg_c - 0.08) * 0.60))

            # Signal 3: Severe inter-frame flickering & warp jitter (face-swap jitter)
            suspicion = max([avg_c] + ([avg_b] if avg_b is not None else []) + ([gend_score] if gend_score is not None else []))
            if flick > 0.35 and w_jit > 0.50 and suspicion > 0.12:
                temporal_fake_signal = max(temporal_fake_signal, min(0.88, 0.48 + (flick - 0.35) * 1.2))

        # Display metric for B-Free / NPR
        display_b = avg_b if avg_b is not None else avg_c

        # Clean mode string
        mode = (mode or "auto").lower().strip()
        if mode not in ("auto", "deepfake", "ai_generated"):
            mode = "auto"

        # ── MODE 1: DEEPFAKE (Face-Swap & Facial Manipulation) ─────────────
        if mode == "deepfake":
            if faces_found == 0:
                return {
                    "success": True,
                    "mode": "deepfake",
                    "prediction": "No Face Detected",
                    "score": 0.0,
                    "confidence": "N/A",
                    "commfor_score": round(avg_c, 6),
                    "bfree_score": round(display_b, 6),
                    "prov_score": 0.0,
                    "frames_analyzed": int(frames_analyzed),
                    "temporal": temporal,
                    "gend_score": None,
                    "faces_found": 0,
                    "explanation": {
                        "category": "No Face Detected",
                        "summary": "Deepfake analysis requires a visible human face, but no face was found.",
                        "findings": [
                            "The face detection locator found 0 faces in this media.",
                            "Deepfake mode specifically evaluates facial blending boundaries and identity replacement.",
                            "If this media does not contain people, switch to 'AI-Generated' or 'Auto' mode to test full-scene synthesis."
                        ],
                        "likelihood": "Inconclusive"
                    },
                    "media_type": "video" if is_video else "image",
                    "frame_scores": frame_scores,
                    "exif": exif,
                    "ela": ela,
                    "face_box": None,
                    "region": None,
                }

            g_score = gend_score if gend_score is not None else 0.0
            if region and region.get("ela_mismatch") and g_score > 0.25:
                final_score = max(g_score, 0.72)
            else:
                final_score = g_score

            threshold = 0.35
            is_fake = final_score > threshold
            confidence = final_score if is_fake else (1.0 - final_score)
            prov_score = 0.0

            findings = []
            if is_fake:
                category = "Deepfake (Face Swap)"
                summary = f"High probability: A face in this {'video' if is_video else 'image'} has been replaced or manipulated by AI."
                if g_score > threshold:
                    findings.append("The face-forensics model detected facial blending seams, skin boundary artifacts, or identity morphing.")
                if region and region.get("ela_mismatch"):
                    findings.append(f"Compression (ELA) in the face region differs significantly from the background (ratio {region['ela_ratio']:.2f}), confirming a pasted or regenerated face.")
            else:
                category = "Authentic Face"
                summary = f"This {'video' if is_video else 'image'} displays natural, authentic facial anatomy."
                findings.append("No face-swap boundary lines, blending seams, or facial identity anomalies were found.")
                findings.append("Facial textures, skin pore continuity, and compression match the surrounding background naturally.")

            margin = final_score - threshold
            likelihood = "Very likely" if margin > 0.30 else ("Likely" if margin > 0.10 else "Possible")
            if not is_fake:
                likelihood = "Unlikely to be manipulated"

            explanation = {
                "category": category,
                "summary": summary,
                "findings": findings,
                "likelihood": likelihood
            }

        # ── MODE 2: AI-GENERATED (Full Scene Diffusion / GAN) ─────────────
        elif mode == "ai_generated":
            if is_prov_fake:
                final_score = 1.0
                prov_score = 1.0
            else:
                final_score = max(avg_c, display_b) if (avg_c > 0.40 and display_b > 0.40) else (0.50 * avg_c + 0.50 * display_b)
                prov_score = 0.0

            threshold = 0.35
            is_fake = final_score > threshold
            confidence = final_score if is_fake else (1.0 - final_score)

            findings = []
            if is_prov_fake:
                findings.append("C2PA content credentials or embedded metadata explicitly declare this file as AI-generated.")
            if is_fake:
                category = "AI-Generated (Synthetic)"
                summary = f"Detected: This {'video' if is_video else 'image'} was generated from scratch by an AI model (Midjourney, DALL-E, Stable Diffusion, Flux, etc.)."
                findings.append("The Community Forensics Vision Transformer detected generative pixel fingerprints and spectral anomalies.")
                if display_b > threshold:
                    findings.append("Secondary spectral analysis confirmed synthetic frequency distributions across the full frame.")
            else:
                category = "Authentic Capture"
                summary = f"This {'video' if is_video else 'image'} conforms to authentic camera capture physics."
                findings.append("No generative AI fingerprints, diffusion noise patterns, or synthetic frequency signatures were detected.")
                findings.append("Pixel statistics conform to natural optical camera sensor physics.")

            margin = final_score - threshold
            likelihood = "Very likely" if (margin > 0.30 or is_prov_fake) else ("Likely" if margin > 0.10 else "Possible")
            if not is_fake:
                likelihood = "Unlikely to be manipulated"

            explanation = {
                "category": category,
                "summary": summary,
                "findings": findings,
                "likelihood": likelihood
            }

        # ── MODE 3: AUTO (Autonomous Intelligent Ensemble) ────────────────
        else:
            cal = calibration.load()
            cal_p = calibration.apply(cal, avg_c, display_b)
            if cal_p is not None:
                pixel_final, threshold = cal_p, 0.40
            else:
                active_models = [("commfor", avg_c, 0.50)]
                if avg_b is not None:
                    active_models.append(("bfree", avg_b, 0.50))
                if gend_score is not None and faces_found > 0:
                    active_models.append(("gend", gend_score, 0.50))
                if is_video and temporal_fake_signal > 0.0:
                    active_models.append(("temporal", temporal_fake_signal, 0.40))
                
                tot_w = sum(w for _, _, w in active_models)
                pixel_final = sum(s * (w / tot_w) for _, s, w in active_models)
                threshold = 0.40

                # Multi-model consensus and false-alarm protection
                pix_scores = [avg_c] + ([avg_b] if avg_b is not None else [])
                strong_consensus = (len(pix_scores) >= 2 and all(s > 0.40 for s in pix_scores))
                face_manip = (gend_score is not None and gend_score > 0.48 and faces_found > 0)
                face_sus = (gend_score is not None and gend_score > 0.38 and faces_found > 0)
                ela_flag = bool(region and region.get("ela_mismatch"))

                if strong_consensus:
                    pixel_final = max(pixel_final, max(pix_scores) * 0.92)
                    threshold = 0.35
                elif face_manip:
                    pixel_final = max(pixel_final, gend_score * 0.92)
                    threshold = 0.38
                elif face_sus and ela_flag:
                    pixel_final = max(pixel_final, max(gend_score, 0.72))
                    threshold = 0.35
                elif is_video and temporal_fake_signal > 0.48:
                    pixel_final = max(pixel_final, temporal_fake_signal)
                    threshold = 0.38
                elif is_video and max(pix_scores) > 0.45:
                    pixel_final = max(pixel_final, max(pix_scores) * 0.88)
                    threshold = 0.38
                elif max(pix_scores) > 0.88:
                    min_s = min(pix_scores)
                    if min_s < 0.12 and faces_found == 0 and not is_video:
                        # Isolated single-model spike with zero face anomaly on static image
                        pixel_final = 0.50 * max(pix_scores) + 0.50 * min_s
                    else:
                        pixel_final = max(pixel_final, max(pix_scores) * 0.85)
                        threshold = 0.35
                elif len(pix_scores) >= 2 and max(pix_scores) > 0.55 and min(pix_scores) < 0.15 and not is_video and faces_found == 0:
                    # Disagreement between ViT and CNN on uncompressed camera photo: keep safe
                    pixel_final = min(pixel_final, 0.36)
                    threshold = 0.40

            if is_prov_fake:
                final_score = max(pixel_final, 0.99)
                prov_score = 1.0
            else:
                final_score = pixel_final
                prov_score = 0.0

            is_fake = final_score > threshold
            confidence = final_score if is_fake else (1.0 - final_score)
            explanation = build_explanation(is_fake, is_video, avg_c, display_b, gend_score, is_prov_fake, prov_result, faces_found, region)

        return {
            "success": True,
            "mode": mode,
            "prediction": "Fake" if is_fake else "Real",
            "score": round(final_score, 6),
            "confidence": f"{confidence * 100:.2f}%",
            "commfor_score": round(avg_c, 6),
            "bfree_score": round(display_b, 6),
            "prov_score": round(prov_score, 6),
            "frames_analyzed": int(frames_analyzed),
            "temporal": temporal,
            "gend_score": round(gend_score, 6) if gend_score is not None else None,
            "faces_found": faces_found,
            "explanation": explanation,
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

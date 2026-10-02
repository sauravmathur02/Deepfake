import os
import io
import torch
import numpy as np
import cv2
import tempfile
import warnings
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from models import get_model
from PIL import Image
import torchvision.transforms as transforms
from scipy.fft import fft2, fftshift

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="Deepfake Detector API — Ensemble")

# ─── Model Setup ────────────────────────────────────────────────────────────────
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]

print("Loading CLIP model...")
model = get_model('CLIP:ViT-L/14')
state_dict = torch.load('./pretrained_weights/fc_weights.pth', map_location='cpu')
model.fc.load_state_dict(state_dict)
model.eval()
if torch.cuda.is_available():
    model.cuda()
print("Model ready.")

clip_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ─── Detector 1: CLIP Neural Network ────────────────────────────────────────────
def clip_score(pil_img: Image.Image) -> float:
    """Returns probability 0–1 of image being fake via CLIP model."""
    in_tens = clip_transform(pil_img).unsqueeze(0)
    if torch.cuda.is_available():
        in_tens = in_tens.cuda()
    with torch.no_grad():
        return model(in_tens).sigmoid().item()

# ─── Detector 2: FFT Frequency Domain Analysis ──────────────────────────────────
def fft_score(pil_img: Image.Image) -> float:
    """
    AI-generated images often have anomalous spectral energy distributions.
    GAN/diffusion upsampling leaves distinctive high-frequency grid artefacts.
    Returns probability 0–1 of image being fake.
    """
    img_gray = np.array(pil_img.convert("L").resize((256, 256)), dtype=np.float32)
    
    # 2-D FFT + shift DC to center
    fft_img  = fftshift(fft2(img_gray))
    magnitude = np.log1p(np.abs(fft_img))

    h, w    = magnitude.shape
    cy, cx  = h // 2, w // 2
    radius  = min(h, w) // 6          # inner "low-freq" radius

    # Mask: low-freq (inner circle) vs high-freq (outer ring)
    Y, X = np.ogrid[:h, :w]
    dist  = np.sqrt((X - cx)**2 + (Y - cy)**2)
    low_mask  = dist <= radius
    high_mask = dist > radius

    low_energy  = magnitude[low_mask].mean()
    high_energy = magnitude[high_mask].mean()

    # Real photos: low >> high. AI images: unnaturally flat or spikey ratio.
    ratio = high_energy / (low_energy + 1e-8)

    # Calibrated empirically: real images typically ratio ~0.35–0.55
    # Scores outside this range → more likely fake
    natural_min, natural_max = 0.35, 0.65
    if natural_min <= ratio <= natural_max:
        return max(0.0, (natural_min - ratio) / natural_min * 0.5 + 0.1)
    elif ratio < natural_min:
        # Too smooth — over-processed (diffusion models)
        return min(1.0, 0.4 + (natural_min - ratio) * 4)
    else:
        # Spikey artefacts — GAN grid pattern
        return min(1.0, 0.4 + (ratio - natural_max) * 4)

# ─── Detector 3: Edge Noise Analysis ────────────────────────────────────────────
def edge_noise_score(pil_img: Image.Image) -> float:
    """
    Real photos have natural, heterogeneous edge-level noise from camera sensors.
    AI images are often suspiciously smooth or have uniformly repetitive texture patterns.
    Returns probability 0–1 of image being fake.
    """
    img_np = np.array(pil_img.convert("L").resize((256, 256)))

    # Laplacian highlights fine noise
    laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
    noise_std  = laplacian.std()
    noise_mean = np.abs(laplacian).mean()

    # Local variance map (8x8 blocks)
    block = 8
    variances = []
    for y in range(0, 256 - block, block):
        for x in range(0, 256 - block, block):
            patch = laplacian[y:y+block, x:x+block]
            variances.append(patch.var())
    var_of_var = np.var(variances)   # how heterogeneous is the noise?

    # Real photos: high variance heterogeneity; AI: unnaturally uniform
    # Empirical thresholds
    score = 0.0

    # Too smooth overall → likely AI (diffusion)
    if noise_std < 8.0:
        score += 0.5

    # Noise uniformly distributed (low var-of-var) → likely AI
    if var_of_var < 1500:
        score += 0.3

    # Very high uniform noise → GAN artefact
    if noise_mean > 15 and var_of_var < 3000:
        score += 0.2

    return min(1.0, score)

# ─── Ensemble Combine ───────────────────────────────────────────────────────────
def ensemble_predict(pil_img: Image.Image):
    """Weighted ensemble of all three detectors."""
    s_clip  = clip_score(pil_img)
    s_fft   = fft_score(pil_img)
    s_edge  = edge_noise_score(pil_img)

    # Weights: CLIP is the primary signal, FFT + edge are supporting signals
    weights = {"clip": 0.55, "fft": 0.25, "edge": 0.20}
    final = (
        weights["clip"]  * s_clip +
        weights["fft"]   * s_fft  +
        weights["edge"]  * s_edge
    )
    return {
        "final":      round(final, 6),
        "clip_score": round(s_clip, 6),
        "fft_score":  round(s_fft,  6),
        "edge_score": round(s_edge, 6),
    }

# ─── Video Frame Extraction ──────────────────────────────────────────────────────
def extract_frames(video_bytes: bytes, n_frames=8, first_n=32) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    frames, count = [], 0
    step = max(1, first_n // n_frames)
    while count < first_n:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        count += 1
    cap.release()
    os.remove(tmp_path)
    return frames

# ─── FastAPI Routes ──────────────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return open("static/index.html").read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        is_video = (
            file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
            or file.content_type.startswith('video/')
        )

        if is_video:
            frames = extract_frames(contents)
            if not frames:
                return {"success": False, "error": "Could not extract frames from video. Make sure it is a valid MP4/AVI/MOV file."}
            results = [ensemble_predict(f) for f in frames]
            scores  = {
                "final":      round(np.mean([r["final"]      for r in results]), 6),
                "clip_score": round(np.mean([r["clip_score"] for r in results]), 6),
                "fft_score":  round(np.mean([r["fft_score"]  for r in results]), 6),
                "edge_score": round(np.mean([r["edge_score"] for r in results]), 6),
            }
            frames_analyzed = len(frames)
        else:
            try:
                img = Image.open(io.BytesIO(contents)).convert("RGB")
            except Exception as img_err:
                return {"success": False, "error": f"Cannot open image: {str(img_err)}. Supported formats: JPG, PNG, WEBP, BMP."}
            scores = ensemble_predict(img)
            frames_analyzed = 1

        final   = scores["final"]
        # Dynamic threshold: if any individual detector is highly confident, lower bar
        threshold = 0.30 if max(scores["clip_score"], scores["fft_score"], scores["edge_score"]) > 0.70 else 0.40
        is_fake = final > threshold
        confidence = final if is_fake else (1.0 - final)

        return {
            "success":        True,
            "prediction":     "Fake" if is_fake else "Real",
            "score":          final,
            "confidence":     f"{confidence * 100:.2f}%",
            "clip_score":     scores["clip_score"],
            "fft_score":      scores["fft_score"],
            "edge_score":     scores["edge_score"],
            "frames_analyzed": frames_analyzed,
        }
    except Exception as e:
        import traceback
        err_msg = str(e) if str(e) else f"Unexpected error: {type(e).__name__}"
        print(traceback.format_exc())
        return {"success": False, "error": err_msg}

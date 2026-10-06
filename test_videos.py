import os
import io
import torch
import numpy as np
import cv2
import tempfile
import warnings
from PIL import Image
from transformers import pipeline

warnings.filterwarnings("ignore")

device = 0 if torch.cuda.is_available() else -1
hf_pipe = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection", device=device)

def get_hf_score(pil_img: Image.Image) -> float:
    results = hf_pipe(pil_img)
    fake_prob = 0.0
    for res in results:
        label = str(res['label']).lower()
        if 'fake' in label:
            fake_prob = res['score']
    if fake_prob == 0.0 and len(results) > 0:
        top_label = str(results[0]['label']).lower()
        top_score = results[0]['score']
        if 'real' in top_label:
            fake_prob = 1.0 - top_score
        elif 'fake' in top_label:
            fake_prob = top_score
    return float(fake_prob)

def fft_score(pil_img: Image.Image) -> float:
    try:
        from scipy.fft import fft2, fftshift
    except ImportError:
        return 0.0
    img_gray = np.array(pil_img.convert("L").resize((256, 256)), dtype=np.float32)
    fft_img  = fftshift(fft2(img_gray))
    magnitude = np.log1p(np.abs(fft_img))
    h, w  = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6
    Y, X     = np.ogrid[:h, :w]
    dist     = np.sqrt((X - cx)**2 + (Y - cy)**2)
    low_mask = dist <= radius
    low_energy  = magnitude[low_mask].mean()
    high_energy = magnitude[~low_mask].mean()
    ratio = high_energy / (low_energy + 1e-8)
    natural_min, natural_max = 0.28, 0.78
    if natural_min <= ratio <= natural_max:
        return 0.05
    elif ratio < natural_min:
        return float(min(0.95, 0.20 + (natural_min - ratio) * 3.5))
    else:
        return float(min(0.95, 0.20 + (ratio - natural_max) * 3.5))

def edge_noise_score(pil_img: Image.Image) -> float:
    img_gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    if np.sum(edges) == 0: return 0.0
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    variance = laplacian.var()
    score = 1.0 - min(1.0, variance / 500.0)
    return float(score)

def extract_frames(video_path: str, n_frames: int = 8, first_n: int = 32) -> list:
    cap = cv2.VideoCapture(video_path)
    frames, count = [], 0
    step = max(1, first_n // n_frames)
    while count < first_n:
        ret, frame = cap.read()
        if not ret: break
        if count % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        count += 1
    cap.release()
    return frames

folder = "test image and videos"
files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

for f in files:
    path = os.path.join(folder, f)
    frames = extract_frames(path)
    if not frames: continue
    
    hf = float(np.mean([get_hf_score(fr) for fr in frames]))
    fft = float(np.mean([fft_score(fr) for fr in frames]))
    edge = float(np.mean([edge_noise_score(fr) for fr in frames]))
    
    final = 0.60 * hf + 0.25 * fft + 0.15 * edge
    max_ind = max(hf, fft, edge)
    
    override = max_ind > 0.85
    is_fake = override or final > 0.40
    
    print(f"VIDEO: {f}")
    print(f"HF: {hf:.2f}, FFT: {fft:.2f}, Edge: {edge:.2f}")
    print(f"Final: {final:.2f}, Override: {override}, Fake: {is_fake}\n")

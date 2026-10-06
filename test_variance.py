import os
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

hf_pipe = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection", device=-1)

def get_hf_score(pil_img: Image.Image) -> float:
    results = hf_pipe(pil_img)
    fake_prob = 0.0
    for res in results:
        label = str(res['label']).lower()
        if 'fake' in label:
            fake_prob = res['score']
    if fake_prob == 0.0 and len(results) > 0:
        top_label = str(results[0]['label']).lower()
        if 'real' in top_label:
            fake_prob = 1.0 - results[0]['score']
        elif 'fake' in top_label:
            fake_prob = results[0]['score']
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
    ratio = magnitude[~low_mask].mean() / (magnitude[low_mask].mean() + 1e-8)
    natural_min, natural_max = 0.28, 0.78
    if natural_min <= ratio <= natural_max:
        return 0.05
    elif ratio < natural_min:
        return float(min(0.95, 0.20 + (natural_min - ratio) * 3.5))
    else:
        return float(min(0.95, 0.20 + (ratio - natural_max) * 3.5))

def extract_frames(video_path: str, n_frames: int = 15, first_n: int = 60) -> list:
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
files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4'))]

for f in files:
    path = os.path.join(folder, f)
    frames = extract_frames(path)
    if not frames: continue
    
    hf_scores = [get_hf_score(fr) for fr in frames]
    fft_scores = [fft_score(fr) for fr in frames]
    
    # Let's also look at pixel-level frame differences (optical flow proxy)
    frame_diffs = []
    for i in range(len(frames)-1):
        f1 = np.array(frames[i].convert("L"))
        f2 = np.array(frames[i+1].convert("L"))
        diff = np.mean(np.abs(f1.astype(np.float32) - f2.astype(np.float32)))
        frame_diffs.append(diff)
        
    avg_hf = np.mean(hf_scores)
    std_hf = np.std(hf_scores)
    avg_fft = np.mean(fft_scores)
    std_fft = np.std(fft_scores)
    avg_diff = np.mean(frame_diffs) if frame_diffs else 0
    std_diff = np.std(frame_diffs) if frame_diffs else 0
    
    print(f"VIDEO: {f}")
    print(f"  HF:  avg={avg_hf:.4f}, std={std_hf:.4f}")
    print(f"  FFT: avg={avg_fft:.4f}, std={std_fft:.4f}")
    print(f"  Pixel Diff: avg={avg_diff:.4f}, std={std_diff:.4f}\n")

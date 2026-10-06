import os
import io
import torch
import numpy as np
import cv2
import tempfile
import warnings
from PIL import Image
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'UniversalFakeDetect'))
from models import get_model

warnings.filterwarnings("ignore")

print("Loading Universal Fake Detect Model (CLIP:ViT-L/14)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model('CLIP:ViT-L/14')
state_dict = torch.load('UniversalFakeDetect/pretrained_weights/fc_weights.pth', map_location='cpu')
model.fc.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("Model ready.")

MEAN = {"clip": [0.48145466, 0.4578275, 0.40821073]}
STD = {"clip": [0.26862954, 0.26130258, 0.27577711]}

img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN['clip'], std=STD['clip']),
])

def get_ufd_score(pil_img: Image.Image) -> float:
    with torch.no_grad():
        img_t = img_transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
        score = model(img_t).sigmoid().item()
    return float(score)

def extract_frames(video_path: str, n_frames: int = 8, first_n: int = 32) -> list:
    cap = cv2.VideoCapture(video_path)
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
    return frames

folder = "test image and videos"
files = os.listdir(folder)

for f in files:
    path = os.path.join(folder, f)
    is_video = f.lower().endswith(('.mp4', '.avi', '.mov'))
    
    # Encode correctly for windows terminal printing
    safe_f = f.encode('ascii', 'replace').decode('ascii')
    
    if is_video:
        frames = extract_frames(path)
        if not frames:
            print(f"{safe_f}: Could not extract frames")
            continue
        scores = [float(get_ufd_score(fr)) for fr in frames]
        avg = float(np.mean(scores))
        std = float(np.std(scores))
        print(f"VIDEO {safe_f} | Avg UFD Score: {avg:.4f} | Std: {std:.4f}")
    else:
        try:
            img = Image.open(path).convert("RGB")
            score = get_ufd_score(img)
            print(f"IMAGE {safe_f} | UFD Score: {score:.4f}")
        except Exception as e:
            print(f"{safe_f}: Error {e}")

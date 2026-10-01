import os
import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from models import get_model
from PIL import Image
import torchvision.transforms as transforms
import warnings
import cv2
import tempfile
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="Deepfake Detector API")

MEAN = {"clip":[0.48145466, 0.4578275, 0.40821073]}
STD = {"clip":[0.26862954, 0.26130258, 0.27577711]}

print("Loading model for the web app...")
model = get_model('CLIP:ViT-L/14')
state_dict = torch.load('./pretrained_weights/fc_weights.pth', map_location='cpu')
model.fc.load_state_dict(state_dict)
model.eval()
if torch.cuda.is_available():
    model.cuda()
print("Model loaded successfully.")

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN["clip"], std=STD["clip"]),
])

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')) or file.content_type.startswith('video/')
        
        if is_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
                
            cap = cv2.VideoCapture(tmp_path)
            frames = []
            count = 0
            while count < 32:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % 4 == 0: # Extracts 8 frames from the first 32
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    frames.append(img)
                count += 1
            cap.release()
            os.remove(tmp_path)
            
            if len(frames) == 0:
                return {"success": False, "error": "Could not extract frames from video."}
                
            scores = []
            for img in frames:
                in_tens = transform(img).unsqueeze(0)
                if torch.cuda.is_available():
                    in_tens = in_tens.cuda()
                with torch.no_grad():
                    output = model(in_tens).sigmoid().item()
                    scores.append(output)
                    
            final_score = sum(scores) / len(scores)
        else:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            in_tens = transform(img).unsqueeze(0)
            
            if torch.cuda.is_available():
                in_tens = in_tens.cuda()
                
            with torch.no_grad():
                final_score = model(in_tens).sigmoid().item()
                
        is_fake = final_score > 0.15
        confidence = final_score if is_fake else (1.0 - final_score)
        
        return {
            "success": True,
            "prediction": "Fake" if is_fake else "Real",
            "score": final_score,
            "confidence": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

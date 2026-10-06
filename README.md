# Universal Synthetic Media Detector

A web-based deepfake detection tool that detects **AI-generated images and videos** using a 3-layer ensemble detection engine.

## Features
- 🖼️ **Image Detection** — Upload any JPG, PNG, WEBP image
- 🎬 **Video Detection** — Upload MP4, AVI, MOV, MKV videos (analyzes 8 frames)
- 🧠 **3-Layer Ensemble** — B-Free (DINOv2) + Community Forensics (ViT) + GenD (Face-based CLIP)
- 🎨 **Modern UI** — Glassmorphism design with live score breakdown

## How to Run

### Option 1: Double-click (easiest)
Just double-click **`run.bat`** — it handles everything automatically.

### Option 2: Terminal
```bash
cd UniversalFakeDetect
uvicorn app:app --port 8000
```

Then open **http://localhost:8000** in your browser.

## Install Dependencies
```bash
pip install -r requirements.txt
```

## How Detection Works

| Detector | Method | Weight |
|---|---|---|
| 🤖 GenD (2026) | Face-based CLIP-L, detects AI-generated faces | 34% |
| 🧠 B-Free (2025) | DINOv2 Bias-Free Vision Transformer | 33% |
| 〰️ CommFor (2025)| Community Forensics SotA ViT | 33% |

For **videos**: extracts 8 evenly-spaced frames from the video, runs all 3 detectors (extracting face crops for GenD), and averages the results.

## Project Structure
```
Deepfake/
├── run.bat                        ← Start the app (double-click)
├── requirements.txt               ← All Python dependencies
└── UniversalFakeDetect/
    ├── app.py                     ← FastAPI server + ensemble detection logic
    ├── models/                    ← CLIP model architecture
    ├── pretrained_weights/        ← Trained model weights
    └── static/                    ← Web UI (HTML, CSS, JS)
```

## Credits
- [B-Free](https://arxiv.org/abs/2410.02758) — Bias-Free Vision Transformer
- [Community Forensics](https://github.com/cf) — Universal Image Forensics
- [GenD](https://github.com/yermandy/GenD) — Generalizable Deepfake Detection (WACV 2026)

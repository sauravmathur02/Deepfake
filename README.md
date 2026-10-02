# Deepfake Detector

A web-based deepfake detection tool that detects **AI-generated images and videos** using a 3-layer ensemble detection engine.

## Features
- 🖼️ **Image Detection** — Upload any JPG, PNG, WEBP image
- 🎬 **Video Detection** — Upload MP4, AVI, MOV, MKV videos (analyzes 8 frames)
- 🧠 **3-Layer Ensemble** — CLIP Neural Net + FFT Frequency Analysis + Edge Noise Analysis
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
| 🧠 CLIP Neural Net | CLIP:ViT-L/14 trained on fake images (CVPR 2023) | 55% |
| 〰️ FFT Frequency | Detects GAN grid artefacts in the frequency domain | 25% |
| ⬛ Edge Noise | Detects AI smoothness vs natural camera sensor noise | 20% |

For **videos**: extracts 8 evenly-spaced frames from the first 32 frames, runs all 3 detectors on each frame, and averages the results.

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
- [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect) — CVPR 2023
- [DeCoF](https://github.com/LongMa-2025/DeCoF) — Frame consistency methodology

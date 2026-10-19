# Deepfake Detector Test Dataset (5,400+ Images & Videos)

This directory contains over **5,400 labeled real and fake media files** downloaded and structured specifically for benchmarking the Universal Synthetic Media Detector.

---

## Folder Structure

```
test_samples/
├── real_images/         # 2,707 Authentic Photos (Real human faces and photography)
├── fake_images/         # 2,700 Synthetic & Deepfake Images (AI-generated faces and manipulations)
├── real_videos/         # 6 Authentic Real Videos (MP4 format)
├── fake_videos/         # 6 Deepfake Manipulated Videos (MP4 format)
├── benchmark.py         # Automated evaluation script (supports --limit and --all)
└── run_test.bat         # 1-Click batch script to run the benchmark on Windows
```

---

## Media Inventory

| Category | Count | Source & Details |
|---|---|---|
| 🟢 **`real_images/`** | **2,707** images | High-quality authentic human face portraits (`real_00000.png` ... `real_02706.png`) |
| 🔴 **`fake_images/`** | **2,700** images | StyleGAN, diffusion & deepfake facial manipulations (`fake_00000.png` ... `fake_02699.png`) |
| 🟢 **`real_videos/`** | **6** videos | Authentic pedestrian & portrait recordings (`real_01_exit_phone_room.mp4`, `real_vid_01_...mp4`, etc.) |
| 🔴 **`fake_videos/`** | **6** videos | Face-swapped and synthetic facial motion videos (`fake_01_outside_talking.mp4`, `fake_vid_01_...mp4`, etc.) |

---

## How to Test

### Option 1: Drag & Drop in Web UI (Single or Bulk)
1. Open the UI at `http://localhost:8000`.
2. To test single files: Select **"Single Analysis"**, drop any file from `fake_images/`, `real_images/`, `fake_videos/`, or `real_videos/`.
3. To test a folder in batch: Select **"Bulk Analysis"**, drop multiple files or a whole folder, and inspect batch metrics!

### Option 2: 1-Click Automated Benchmark
Open terminal in `test_samples/` or double-click `run_test.bat`:
```bash
# Quick test (10 samples per category):
python benchmark.py

# Test custom limit (e.g., 50 samples):
python benchmark.py --limit 50

# Test all 5,400+ samples:
python benchmark.py --all
```

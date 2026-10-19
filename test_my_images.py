import os
import json
import time
import requests
from pathlib import Path

API_URL = "http://localhost:8000/predict"
IMAGE_DIR = Path(r"d:\Projects\Deepfake\Image")
ROOT_DIR = Path(r"d:\Projects\Deepfake")

files_to_test = []
for f in sorted(IMAGE_DIR.glob("*")):
    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        files_to_test.append(f)

for root_file in ["F.png", "R.jpg"]:
    rf = ROOT_DIR / root_file
    if rf.exists():
        files_to_test.append(rf)

results = []

for fpath in files_to_test:
    fname = fpath.name
    ctype = "image/png" if fname.lower().endswith(".png") else "image/jpeg"
    
    t0 = time.time()
    try:
        with open(fpath, "rb") as f:
            resp = requests.post(API_URL, files={"file": (fname, f, ctype)}, timeout=120)
        elapsed = time.time() - t0
        
        if resp.status_code == 200:
            data = resp.json()
            results.append({
                "filename": fname,
                "prediction": data.get("prediction"),
                "score": data.get("score"),
                "confidence": data.get("confidence"),
                "faces_found": data.get("faces_found"),
                "category": data.get("category"),
                "details": data.get("details"),
                "explanation": data.get("explanation"),
                "elapsed_sec": round(elapsed, 2)
            })
    except Exception as e:
        results.append({"filename": fname, "error": str(e)})

with open(r"d:\Projects\Deepfake\test_my_images_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Processed {len(results)} images. Saved to test_my_images_results.json")

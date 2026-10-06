"""
Evaluate GenD (face-based, CLIP-L/14) on vbench/real vs vbench/fake.
Run with the SEPARATE env:  ..\\venv_gend\\Scripts\\python eval_gend.py
Nothing here is tuned; it only reports AUC and the confusion at p=0.5.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "detectors", "GenD"))
from src.hf.modeling_gend import GenD  # noqa: E402

ROOT = os.path.join(os.path.dirname(HERE), "vbench")
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = GenD.from_pretrained("yermandy/GenD_CLIP_L_14").to(dev).eval()
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def frames_of(path, n=8):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def face_crop(bgr, margin=0.3):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(g, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    m = int(margin * max(w, h))
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, bgr.shape[1]), min(y + h + m, bgr.shape[0])
    return Image.fromarray(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))


def auc(pos, neg):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


rows = []
for label in ("real", "fake"):
    for p in sorted(glob.glob(os.path.join(ROOT, label, "*"))):
        crops = [c for c in (face_crop(f) for f in frames_of(p)) if c is not None]
        if not crops:
            rows.append(dict(label=label, file=os.path.basename(p), gend=None, faces=0))
            print(label, os.path.basename(p), "NO FACE", flush=True)
            continue
        with torch.no_grad():
            x = torch.stack([model.feature_extractor.preprocess(c) for c in crops]).to(dev)
            probs = model(x).softmax(dim=-1)[:, 1].cpu().numpy()  # index 1 = fake (verify below)
        rows.append(dict(label=label, file=os.path.basename(p), gend=float(probs.mean()), faces=len(crops)))
        print(label, os.path.basename(p), round(rows[-1]["gend"], 3), len(crops), flush=True)

json.dump(rows, open(os.path.join(ROOT, "gend_scores.json"), "w"), indent=1)
ok = [r for r in rows if r["gend"] is not None]
real = [r["gend"] for r in ok if r["label"] == "real"]
fake = [r["gend"] for r in ok if r["label"] == "fake"]
print(f"\nvideos with a face found: {len(ok)}/{len(rows)}  (real={len(real)} fake={len(fake)})")
a = auc(fake, real)
print(f"GenD AUC (class 1 = fake) = {a:.3f}   [if far below 0.5, the class index is flipped: AUC would be {1 - a:.3f}]")
print(f"At p>0.5: fakes caught {sum(v > .5 for v in fake)}/{len(fake)}, reals flagged {sum(v > .5 for v in real)}/{len(real)}")

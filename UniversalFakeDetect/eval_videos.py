"""Score vbench/real and vbench/fake with every signal and report AUC (no tuning)."""
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors.pixel import CommForDetector, BFreeDetector
from detectors.video import sample_video_frames
from detectors.temporal import temporal_features

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vbench")


def auc(pos, neg):
    """P(score_fake > score_real), ties = 0.5."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


c, b = CommForDetector(), BFreeDetector()
rows = []
for label in ("real", "fake"):
    for p in sorted(glob.glob(os.path.join(ROOT, label, "*"))):
        try:
            frames = sample_video_frames(p, 8)
            if not frames:
                continue
            t = temporal_features(p) or {}
            rows.append(dict(label=label, file=os.path.basename(p),
                             commfor=float(np.mean(c.predict(frames))),
                             bfree=float(np.mean(b.predict(frames))), **t))
            print(label, rows[-1]["file"], round(rows[-1]["commfor"], 3), round(rows[-1]["bfree"], 3), flush=True)
        except Exception as e:
            print("skip", p, e)

json.dump(rows, open(os.path.join(ROOT, "scores.json"), "w"), indent=1)
real = [r for r in rows if r["label"] == "real"]
fake = [r for r in rows if r["label"] == "fake"]
print(f"\nreal={len(real)} fake={len(fake)}  (AUC 0.5 = random, 1.0 = perfect)")
for k in ("commfor", "bfree", "warp_error", "warp_jitter", "flicker", "hf_energy"):
    a = auc([r[k] for r in fake if k in r], [r[k] for r in real if k in r])
    print(f"{k:12s} AUC(higher=fake) = {a:.3f}")
ens = lambda r: 0.5 * r["commfor"] + 0.5 * r["bfree"]
print(f"{'ensemble':12s} AUC = {auc([ens(r) for r in fake], [ens(r) for r in real]):.3f}")
tp = sum(ens(r) > 0.30 for r in fake)
fp = sum(ens(r) > 0.30 for r in real)
print(f"At threshold 0.30: fakes caught {tp}/{len(fake)}, reals flagged {fp}/{len(real)}")

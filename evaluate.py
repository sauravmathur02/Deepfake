"""
Measure how accurate each detector really is on YOUR labelled files.

Folder layout (create it next to this script's parent folder):

    eval_data/
        real/   <- real photos/videos (phone, webcam, WhatsApp-forwarded, ...)
        fake/   <- AI-generated or AI-edited images/videos

Usage (from d:\\Project\\Deepfake):
    python evaluate.py
    python evaluate.py --data "eval_data" --no-ufd

It prints, for every detector:
    accuracy @0.5, real accuracy, fake accuracy, AUC, best threshold,
and saves per-file scores to eval_results.csv so you can see which files fail.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "UniversalFakeDetect"))

from detectors.provenance import check_provenance  # noqa: E402
from detectors.pixel import CommForDetector, UFDDetector  # noqa: E402
from detectors.video import sample_video_frames  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def auc_score(y, s):
    """ROC AUC without sklearn (Mann-Whitney U). 0.5 = random guessing."""
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def best_threshold(y, s):
    y, s = np.asarray(y), np.asarray(s)
    best_t, best_bal = 0.5, -1
    for t in np.unique(np.concatenate([s, [0.5]])):
        pred = s >= t
        tpr = pred[y == 1].mean() if (y == 1).any() else 0
        tnr = (~pred[y == 0]).mean() if (y == 0).any() else 0
        bal = (tpr + tnr) / 2  # balanced accuracy: fair even if classes are uneven
        if bal > best_bal:
            best_bal, best_t = bal, t
    return float(best_t), float(best_bal)


def report(name, y, s):
    y, s = np.asarray(y), np.asarray(s)
    pred = s >= 0.5
    acc = (pred == (y == 1)).mean()
    r_acc = (~pred[y == 0]).mean() if (y == 0).any() else float("nan")
    f_acc = pred[y == 1].mean() if (y == 1).any() else float("nan")
    t, bal = best_threshold(y, s)
    print(f"{name:32s} acc={acc:6.1%}  real_acc={r_acc:6.1%}  fake_acc={f_acc:6.1%}  "
          f"AUC={auc_score(y, s):.3f}  best_thr={t:.3f} (bal_acc={bal:.1%})")


def load_frames(path, ext):
    if ext in VID_EXT:
        return sample_video_frames(path, n_frames=16)
    return [Image.open(path).convert("RGB")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "eval_data"))
    ap.add_argument("--no-ufd", action="store_true", help="skip the slow 2023 CLIP model")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_results.csv"))
    args = ap.parse_args()

    files = []
    for label_name, label in (("real", 0), ("fake", 1)):
        d = os.path.join(args.data, label_name)
        if not os.path.isdir(d):
            print(f"Missing folder: {d}")
            continue
        for f in sorted(os.listdir(d)):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXT + VID_EXT:
                files.append((os.path.join(d, f), label, ext))
    if not files:
        print("No files found. Create eval_data/real and eval_data/fake first.")
        return
    print(f"{sum(1 for _, l, _ in files if l == 0)} real, "
          f"{sum(1 for _, l, _ in files if l == 1)} fake files\n")

    detectors = [CommForDetector()]
    if not args.no_ufd:
        detectors.append(UFDDetector())

    rows, scores = [], {d.name: [] for d in detectors}
    prov_hits, labels = [], []
    t0 = time.time()
    for i, (path, label, ext) in enumerate(files, 1):
        name = os.path.basename(path)
        try:
            frames = load_frames(path, ext)
            if not frames:
                raise ValueError("no frames")
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        prov = check_provenance(open(path, "rb").read()) if ext in IMG_EXT else {"verdict": "n/a"}
        row = {"file": name, "label": "fake" if label else "real", "provenance": prov["verdict"]}
        for d in detectors:
            s = float(np.mean(d.predict(frames)))
            scores[d.name].append(s)
            row[d.name] = round(s, 4)
        labels.append(label)
        prov_hits.append(prov["verdict"] == "ai_declared")
        rows.append(row)
        print(f"  [{i}/{len(files)}] {name[:50]}", flush=True)

    print(f"\nDone in {time.time() - t0:.0f}s\n")
    print("=" * 110)
    for d in detectors:
        report(d.name, labels, scores[d.name])
    if len(detectors) > 1:
        report("Average of detectors", labels, np.mean([scores[d.name] for d in detectors], axis=0))

    y = np.asarray(labels)
    ph = np.asarray(prov_hits)
    print("-" * 110)
    print(f"Provenance (C2PA/metadata) flagged {ph[y == 1].sum()}/{(y == 1).sum()} fakes "
          f"and {ph[y == 0].sum()}/{(y == 0).sum()} reals (reals flagged = false alarms)")
    print("=" * 110)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Per-file scores saved to {args.out}")


if __name__ == "__main__":
    main()

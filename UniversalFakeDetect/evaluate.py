"""
Honest evaluation of the ensemble on images it was NOT tuned on.

Usage (from UniversalFakeDetect/):
  # public dataset from Hugging Face (streams only N images)
  python evaluate.py --hf julienlucas/midjourney-dalle-sd-dataset --split test --n 200

  # your own labeled folders
  python evaluate.py --real_dir path/to/real --fake_dir path/to/fake

Reports accuracy at the live threshold, AUC (threshold-free), the confusion matrix, and
accuracy on a held-out half using a threshold chosen only on the other half.
"""
import argparse
import glob
import os
import random
import numpy as np
from PIL import Image

from detectors.pixel import CommForDetector, BFreeDetector

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_folder(real_dir, fake_dir):
    items = []
    for d, label in ((real_dir, 0), (fake_dir, 1)):
        if not d:
            continue
        for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if p.lower().endswith(IMG_EXT):
                items.append((p, label))
    return items


def load_hf(name, split, n, seed, data_file=None):
    from datasets import load_dataset
    if data_file:
        if name == "local":
            ds = load_dataset("parquet", data_files={split: data_file}, split=split, streaming=True)
        else:
            url = f"hf://datasets/{name}/{data_file}"
            ds = load_dataset("parquet", data_files={split: url}, split=split, streaming=True)
    else:
        ds = load_dataset(name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=500)
    first = next(iter(ds))
    img_col = next(k for k, v in first.items() if isinstance(v, Image.Image))
    cands = [k for k in first if k != img_col and isinstance(first[k], (int, bool, str))]
    lab_col = next((k for k in cands if k.lower() in ("label_a", "label", "is_fake", "labels")), cands[0])
    print(f"columns: {list(first.keys())}")
    print(f"using image='{img_col}', label='{lab_col}', example label={first[lab_col]!r}")
    return ds, img_col, lab_col


def is_fake_label(v):
    if isinstance(v, str):
        s = v.lower()
        return not any(w in s for w in ("real", "human", "natural", "authentic"))
    return int(v) == 1  # assumes 1 = fake; check printed example label


def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    cmp = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(cmp)


def best_threshold(scores, labels):
    cands = np.linspace(0.05, 0.95, 91)
    accs = [(((scores > t) == (labels == 1)).mean(), t) for t in cands]
    return max(accs)[1]


def report(scores, labels, thr=0.45):
    pred = scores > thr
    tp = int((pred & (labels == 1)).sum()); tn = int((~pred & (labels == 0)).sum())
    fp = int((pred & (labels == 0)).sum()); fn = int((~pred & (labels == 1)).sum())
    print(f"  threshold {thr:.2f}: accuracy {(tp+tn)/len(labels)*100:.1f}%  "
          f"| fakes caught {tp}/{tp+fn}  | reals passed {tn}/{tn+fp}")
    print(f"  confusion: TP={tp} FN(missed fakes)={fn} FP(real flagged)={fp} TN={tn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf"); ap.add_argument("--split", default="test")
    ap.add_argument("--data_file", help="single file inside the HF dataset repo, e.g. data/test-00000-of-00008.parquet")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--real_dir"); ap.add_argument("--fake_dir")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    commfor, bfree = CommForDetector(), BFreeDetector()
    scores_c, scores_b, labels = [], [], []

    def run(img, label):
        img = img.convert("RGB")
        scores_c.append(commfor.predict([img])[0])
        scores_b.append(bfree.predict([img])[0])
        labels.append(label)
        if len(labels) % 25 == 0:
            print(f"  processed {len(labels)}")

    if a.hf:
        ds, img_col, lab_col = load_hf(a.hf, a.split, a.n, a.seed, a.data_file)
        for ex in ds:
            run(ex[img_col], int(is_fake_label(ex[lab_col])))
            if len(labels) >= a.n:
                break
    else:
        items = load_folder(a.real_dir, a.fake_dir)
        random.Random(a.seed).shuffle(items)
        for p, lab in items[:a.n]:
            try:
                run(Image.open(p), lab)
            except Exception as e:
                print("skip", p, e)

    c, b, y = np.array(scores_c), np.array(scores_b), np.array(labels)
    ens = 0.5 * c + 0.5 * b
    import json
    os.makedirs("data", exist_ok=True)
    json.dump({"commfor": c.tolist(), "bfree": b.tolist(), "label": y.tolist()},
              open(os.path.join("data", "bench_scores.json"), "w"))
    print(f"\n=== {len(y)} images ({int(y.sum())} fake / {int((y==0).sum())} real) ===")
    for name, s in (("CommFor", c), ("B-Free", b), ("Ensemble (live: 0.5*C+0.5*B)", ens)):
        print(f"{name}: AUC {auc(s, y):.3f}")
    print("\nEnsemble at live threshold 0.30:")
    report(ens, y, 0.30)

    idx = np.random.RandomState(a.seed).permutation(len(y))
    dev, test = idx[:len(y)//2], idx[len(y)//2:]
    t = best_threshold(ens[dev], y[dev])
    print(f"\nHeld-out check: threshold {t:.2f} chosen on first half, scored on unseen second half:")
    report(ens[test], y[test], t)


if __name__ == "__main__":
    main()

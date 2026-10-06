"""
Choose the decision threshold fairly.

- Tune on half of the benchmark (dev).
- Test on the other half (unseen) AND on your own labelled photos/videos from
  data/feedback.jsonl, which are NEVER used to choose the threshold.
Uses balanced accuracy so the 82%-fake benchmark can't reward "always fake".
Run evaluate.py first (it writes data/bench_scores.json).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
b = json.load(open(os.path.join(HERE, "data", "bench_scores.json")))
ens = 0.5 * np.array(b["commfor"]) + 0.5 * np.array(b["bfree"])
y = np.array(b["label"])

mine = [json.loads(l) for l in open(os.path.join(HERE, "data", "feedback.jsonl")) if l.strip()]
mine = [r for r in mine if "bfree" in r]
my_s = np.array([0.5 * r["commfor"] + 0.5 * r["bfree"] for r in mine])
my_y = np.array([int(r["is_fake"]) for r in mine])


def stats(s, lab, t):
    p = s > t
    tpr = (p & (lab == 1)).sum() / max((lab == 1).sum(), 1)
    tnr = (~p & (lab == 0)).sum() / max((lab == 0).sum(), 1)
    return tpr, tnr, (tpr + tnr) / 2


rng = np.random.RandomState(0)
idx = rng.permutation(len(y))
dev, test = idx[: len(y) // 2], idx[len(y) // 2:]
ts = np.linspace(0.02, 0.9, 89)
best = max(ts, key=lambda t: stats(ens[dev], y[dev], t)[2])

print(f"Threshold chosen on bench dev half (balanced accuracy): {best:.2f}\n")
print(f"{'set':28s}{'fakes caught':>14s}{'reals passed':>14s}{'bal.acc':>10s}")
for name, s, lab in (("bench dev (tuning)", ens[dev], y[dev]),
                     ("bench test (unseen)", ens[test], y[test]),
                     (f"my own files (n={len(my_y)})", my_s, my_y)):
    for t in (0.30, best):
        tpr, tnr, ba = stats(s, lab, t)
        print(f"{name + ' @' + format(t, '.2f'):28s}{tpr * 100:13.1f}%{tnr * 100:13.1f}%{ba * 100:9.1f}%")

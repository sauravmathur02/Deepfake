"""Download a small subset of angads24/deepfake-video into vbench/real and vbench/fake."""
import os
import sys
from huggingface_hub import HfApi, hf_hub_download

REPO = "angads24/deepfake-video"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vbench")

api = HfApi()
files = api.list_repo_files(REPO, repo_type="dataset")
for label in ("real", "fake"):
    vids = sorted(f for f in files if f.startswith(f"dataset/{label}/")
                  and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")))
    print(f"{label}: {len(vids)} videos in repo, downloading {min(N, len(vids))}")
    dest = os.path.join(OUT, label)
    os.makedirs(dest, exist_ok=True)
    for f in vids[:N]:
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=os.path.join(OUT, "_cache"))
        os.replace(p, os.path.join(dest, os.path.basename(f)))
print("Done ->", OUT)

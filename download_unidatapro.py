import os
from huggingface_hub import HfApi, hf_hub_download

REPO = "UniDataPro/deepfake-videos-dataset"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data", "UniDataPro")

api = HfApi()
files = api.list_repo_files(REPO, repo_type="dataset")

print("Downloading dataset to:", OUT_DIR)

# Map HuggingFace folders to local folders
folders = {
    "deepfake": "fake",
    "video": "real",
    "image": "real_images"
}

count = 0
for f in files:
    parts = f.split("/")
    if len(parts) == 2 and parts[0] in folders:
        remote_folder = parts[0]
        filename = parts[1]
        local_folder = folders[remote_folder]
        
        dest = os.path.join(OUT_DIR, local_folder)
        os.makedirs(dest, exist_ok=True)
        
        print(f"Downloading {f} -> {dest}...")
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=os.path.join(OUT_DIR, "_cache"))
        os.replace(p, os.path.join(dest, filename))
        count += 1

print(f"Done! Downloaded {count} files to {OUT_DIR}")

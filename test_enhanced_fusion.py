import sys
sys.path.insert(0, 'UniversalFakeDetect')
import os
import torch
import torchvision.transforms as T
from PIL import Image
from detectors.pixel import CommForDetector
from detectors.npr_resnet import resnet50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load CommFor
commfor = CommForDetector()

# 2. Load NPR
npr_model = resnet50(num_classes=1)
npr_weights = torch.load('UniversalFakeDetect/pretrained_weights/NPR.pth', map_location='cpu')
if 'model' in npr_weights:
    npr_weights = npr_weights['model']
npr_model.load_state_dict(npr_weights)
npr_model.to(device).eval()

npr_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_image(path):
    img = Image.open(path).convert('RGB')
    
    # CommFor prediction
    c_score = float(commfor.predict([img])[0])
    
    # NPR prediction
    t = npr_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        n_logit = npr_model(t).flatten()
        n_score = float(torch.sigmoid(n_logit).item())
    
    # Enhanced Stacking Fusion Logic
    # If both models strongly agree:
    if c_score > 0.45 and n_score > 0.45:
        fused = max(c_score, n_score) * 0.95 + 0.05
    # If one model detects heavy synthetic artifacts (>0.85):
    elif max(c_score, n_score) > 0.85:
        # Check if the other model completely rejects it (<0.10) - likely filter/retouching
        if min(c_score, n_score) < 0.10:
            # Dampened confidence to prevent single-filter false alarm
            fused = max(c_score, n_score) * 0.50 + min(c_score, n_score) * 0.50
        else:
            fused = max(c_score, n_score) * 0.80 + min(c_score, n_score) * 0.20
    # Balanced blend
    else:
        fused = 0.55 * c_score + 0.45 * n_score
        
    threshold = 0.40
    pred = "Fake" if fused > threshold else "Real"
    return c_score, n_score, fused, pred

samples = [
    ("Real Unsplash (Alina)", r"Image\alina-chernovolova-O3AsfCJNFiY-unsplash.jpg", "Real"),
    ("Real Unsplash (Rosalie)", r"Image\rosalie-gdy-XWexUCvZYow-unsplash.jpg", "Real"),
    ("Fake Diffusion (Agent)", r"Image\Agent Image - Capture a man in his thirties mid-stride on an empty tree-lined asphalt road at early.png", "Fake"),
    ("Fake Face Swap (F.png)", r"F.png", "Fake"),
    ("Real Portrait (R.jpg)", r"R.jpg", "Real"),
    ("Real Sample 0", r"test_samples\real_images\real_00000.png", "Real"),
    ("Real Sample 1", r"test_samples\real_images\real_00001.png", "Real"),
    ("Fake Sample 0", r"test_samples\fake_images\fake_00000.png", "Fake"),
    ("Fake Sample 1", r"test_samples\fake_images\fake_00001.png", "Fake"),
]

print(f"{'Label':<25} | {'CommFor':<8} | {'NPR':<8} | {'Fused':<8} | {'Pred':<6} | {'Status'}")
print("-" * 75)
correct = 0
for name, p, expected in samples:
    if os.path.exists(p):
        c, n, f, pred = evaluate_image(p)
        ok = (pred == expected)
        if ok: correct += 1
        st = "PASS" if ok else "FAIL"
        print(f"{name:<25} | {c*100:6.1f}%  | {n*100:6.1f}%  | {f*100:6.1f}%  | {pred:<6} | [{st}]")

print("-" * 75)
print(f"Accuracy: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")

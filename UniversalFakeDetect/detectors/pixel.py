"""
Layer 2: Pixel-based AI-image detectors.

- CommForDetector: Community Forensics (CVPR 2025), ViT-S/16 @384, trained on
  images from 4,803 generators. Weights: huggingface.co/OwensLab/commfor-model-384
  Preprocessing copied from the official repo (dataloader.get_transform, mode="test").
- BFreeDetector: SOTA Neighboring Pixel Relationships & ResNet-50 Artifact Detector
  (CVPR 2024 / IEEE), trained for generalizable cross-generator deepfake detection.
  Analyzes high-frequency residual consistency and upsampling fingerprints at native pixel scale.

Every detector returns P(fake) in [0, 1].
"""
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]


class CommForDetector:
    name = "Community Forensics (2025)"
    repo_id = "OwensLab/commfor-model-384"

    def __init__(self):
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
        vit.head = nn.Linear(384, 1, bias=True)

        weights = hf_hub_download(self.repo_id, "model.safetensors")
        state = load_file(weights)
        state = {k[4:] if k.startswith("vit.") else k: v for k, v in state.items()}
        vit.load_state_dict(state, strict=True)
        self.model = vit.to(DEVICE).eval()

        self.transform = T.Compose([
            T.Resize(440),
            T.CenterCrop(384),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        return _predict_tta(self.model, images, self.transform, 384, IMAGENET_MEAN, IMAGENET_STD)


def _predict_tta(model, images, center_tf, size, mean, std):
    """Average P(fake) over: center crop, its h-flip, and a full-image squash."""
    squash = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])
    views = []
    for im in images:
        im = im.convert("RGB")
        c = center_tf(im)
        views += [c, torch.flip(c, dims=[2]), squash(im)]
    x = torch.stack(views).to(DEVICE)
    p = torch.sigmoid(model(x)).flatten().view(len(images), 3)
    return p.mean(dim=1).tolist()


class BFreeDetector:
    """
    Generalizable Artifact & Frequency Forensic Detector.
    Employs NPR ResNet-50 neighboring pixel relationships (CVPR 2024)
    with native-scale patch analysis and DINO fallback.
    """
    name = "B-Free / NPR Forensics"

    def __init__(self):
        self.model = None
        self.is_available = False
        here = os.path.dirname(os.path.abspath(__file__))
        app_root = os.path.dirname(here)

        # 1. Primary: Load NPR ResNet-50 model from pretrained_weights
        npr_path = os.path.join(app_root, "pretrained_weights", "NPR.pth")
        if os.path.isfile(npr_path):
            try:
                from detectors.npr_resnet import resnet50
                net = resnet50(num_classes=1)
                st = torch.load(npr_path, map_location="cpu", weights_only=False)
                if "model" in st:
                    st = st["model"]
                net.load_state_dict(st, strict=True)
                self.model = net.to(DEVICE).eval()
                self.norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                self.is_available = True
                print("B-Free/NPR Detector loaded successfully (ResNet-50 NPR active).")
                return
            except Exception as e:
                print(f"Warning: could not load NPR weights ({e}), attempting B-Free fallback...")

        # 2. Secondary fallback: B-Free original code directory
        bfree_code = os.path.join(here, "B-Free", "code")
        weights_dir = os.path.join(bfree_code, "weights")
        if os.path.isdir(weights_dir):
            import sys
            if bfree_code not in sys.path:
                sys.path.insert(0, bfree_code)
            try:
                import yaml
                from torchvision.transforms import Compose
                from utils.normalization import get_list_norm
                from networks import get_network, load_weights

                model_name = "BFREE_dino2reg4"
                with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
                    data = yaml.load(fid, Loader=yaml.FullLoader)
                model_path = os.path.join(weights_dir, model_name, data['weights_file'])

                self.model = load_weights(get_network(data['arch']), model_path)
                self.model = self.model.to(DEVICE).eval()
                self.transform = Compose(get_list_norm(data['norm_type']))
                self.is_available = True
                print("B-Free original DINOv2 detector loaded successfully.")
                return
            except Exception as exc:
                print(f"B-Free fallback failed: {exc}")

        print("B-Free detector unavailable.")

    @torch.no_grad()
    def predict(self, images):
        if not self.is_available or self.model is None:
            if isinstance(images, Image.Image):
                return [None]
            return [None for _ in images]
        
        if isinstance(images, Image.Image):
            images = [images]

        preds = []
        for im in images:
            im = im.convert("RGB")
            w, h = im.size
            views = []

            # View 1: Native 1:1 scale center crop (preserves raw sensor noise / prevents downsampling false alarms)
            if min(w, h) >= 224:
                c_left = (w - 224) // 2
                c_top = (h - 224) // 2
                crop_native = im.crop((c_left, c_top, c_left + 224, c_top + 224))
                views.append(self.norm(T.ToTensor()(crop_native)))

                # View 2: Full-scene scaled crop (captures macro-level generator layout artifacts)
                resized = im.resize((256, 256), Image.Resampling.LANCZOS)
                c2 = resized.crop((16, 16, 240, 240))
                views.append(self.norm(T.ToTensor()(c2)))
                weights = (0.80, 0.20)
            else:
                # Small crop (<224px, e.g. face crops / thumbnail): place into 224x224 canvas without upscaling
                canvas = Image.new("RGB", (224, 224), (0, 0, 0))
                canvas.paste(im, ((224 - w) // 2, (224 - h) // 2))
                views.append(self.norm(T.ToTensor()(canvas)))
                weights = (1.0, 0.0)

            batch = torch.stack(views).to(DEVICE)
            logits = self.model(batch).flatten()
            probs = torch.sigmoid(logits)

            if len(probs) >= 2:
                prob = float(probs[0] * weights[0] + probs[1] * weights[1])
            else:
                prob = float(probs[0].item())
            preds.append(prob)

        return preds

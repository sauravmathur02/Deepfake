"""
Layer 2: Pixel-based AI-image detectors.

- CommForDetector: Community Forensics (CVPR 2025), ViT-S/16 @384, trained on
  images from 4,803 generators. Weights: huggingface.co/OwensLab/commfor-model-384
  Preprocessing copied from the official repo (dataloader.get_transform, mode="test").
- UFDDetector: UniversalFakeDetect (CVPR 2023), CLIP ViT-L/14 + linear probe.
  Older; kept as an extra vote.

Every detector returns P(fake) in [0, 1].
"""
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

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

        # Same architecture as the official ViTClassifier (model_size=small, 384, patch16).
        # pretrained=False: the HF checkpoint contains the full fine-tuned backbone.
        vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
        vit.head = nn.Linear(384, 1, bias=True)

        weights = hf_hub_download(self.repo_id, "model.safetensors")
        state = load_file(weights)
        # Checkpoint keys are prefixed with "vit." (attribute name in ViTClassifier)
        state = {k[4:] if k.startswith("vit.") else k: v for k, v in state.items()}
        vit.load_state_dict(state, strict=True)  # raises if keys don't match
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
    name = "B-Free (2025)"

    def __init__(self):
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        bfree_code = os.path.join(here, "B-Free", "code")
        if bfree_code not in sys.path:
            sys.path.insert(0, bfree_code)
            
        import yaml
        from torchvision.transforms import Compose
        from utils.normalization import get_list_norm
        from networks import get_network, load_weights
        
        weights_dir = os.path.join(bfree_code, "weights")
        model_name = "BFREE_dino2reg4"
        with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
            data = yaml.load(fid, Loader=yaml.FullLoader)
        model_path = os.path.join(weights_dir, model_name, data['weights_file'])
        
        self.model = load_weights(get_network(data['arch']), model_path)
        self.model = self.model.to(DEVICE).eval()
        self.transform = Compose(get_list_norm(data['norm_type']))

    @torch.no_grad()
    def predict(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        preds = []
        for im in images:
            x = self.transform(im.convert("RGB")).unsqueeze(0).to(DEVICE)
            out = self.model(x).cpu().numpy()
            if out.shape[1] == 1:
                logit = out[:, 0]
            elif out.shape[1] == 2:
                logit = out[:, 1] - out[:, 0]
            else:
                logit = out[:, 0]
            prob = torch.sigmoid(torch.tensor(logit)).item()
            preds.append(prob)
        return preds

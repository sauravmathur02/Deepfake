import os
import torch
from models import get_model
from PIL import Image
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

MEAN = {"clip":[0.48145466, 0.4578275, 0.40821073]}
STD = {"clip":[0.26862954, 0.26130258, 0.27577711]}

print("Loading model...")
model = get_model('CLIP:ViT-L/14')
state_dict = torch.load('./pretrained_weights/fc_weights.pth', map_location='cpu')
model.fc.load_state_dict(state_dict)
model.eval()
if torch.cuda.is_available():
    model.cuda()
print("Model loaded.")

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN["clip"], std=STD["clip"]),
])

img_dir = 'd:/Project/Deepfake/Image'
for filename in os.listdir(img_dir):
    img_path = os.path.join(img_dir, filename)
    if os.path.isfile(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            in_tens = transform(img).unsqueeze(0)
            if torch.cuda.is_available():
                in_tens = in_tens.cuda()
            
            with torch.no_grad():
                output = model(in_tens).sigmoid().item()
            
            pred = "Fake" if output > 0.5 else "Real"
            print(f"File: {filename}\nScore: {output:.4f} -> Prediction: {pred}\n")
        except Exception as e:
            print(f"Failed to process {filename}: {e}\n")

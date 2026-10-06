import torch
from transformers import pipeline
from PIL import Image

try:
    print("Loading HF model...")
    pipe = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
    
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color = 'red')
    
    result = pipe(img)
    print("Success! Result format:", result)
except Exception as e:
    print("Error:", e)

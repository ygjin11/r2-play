import torch
import clip
from PIL import Image
import numpy as np
import os


def get_vision_clip(model, preprocess, img, device):
    if isinstance(img, str):
        image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    elif isinstance(img, np):
        image = Image.fromarray(np.uint8(img))
        image = preprocess(image).unsqueeze(0)
    else:
        pass
    with torch.no_grad():
        image_features = model.encode_image(image)
    return  image_features

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    img = '' 
    for i in range(20):
        result = get_vision_clip(model, preprocess, img, device)
    print(result.shape)
    




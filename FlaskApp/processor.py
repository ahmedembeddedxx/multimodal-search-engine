from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
from tqdm import tqdm
import os
from IPython.display import display
import numpy as np
import torch


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

clip_model = clip_model.to(device)
def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)

    image_features = clip_model.get_image_features(pixel_values=inputs.pixel_values)

    return image_features.cpu().detach()
def process_all_images(folder_name):
    if not os.path.exists("image_embeddings"):
        os.makedirs("image_embeddings")

    files = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_files = len(image_files)

    with tqdm(total=total_files, desc="Processing images", leave=False) as pbar:
        for file in image_files:
            item_path = os.path.join(folder_name, file)
            with open(item_path, "rb") as f:
                image_bytes = f.read()

            image_features = process_image(image_bytes)
            torch.save(image_features, os.path.join("image_embeddings", f"{file}.pt"))

            pbar.update(1)
process_all_images(folder_name="MiniData/")
print("All images converted to embeddings")

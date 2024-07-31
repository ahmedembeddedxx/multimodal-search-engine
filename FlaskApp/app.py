from flask import Flask, request, jsonify, render_template
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import torch
import os
from tqdm import tqdm

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

clip_model = clip_model.to(device)

def process_text(text):
    inputs = clip_processor(
        text=text,
        return_tensors="pt"
    )
    
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)
    
    text_features = clip_model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

    return text_features

def get_images(text_query, folder_name = "static"):
    text_features = process_text(text_query)
    
    top_images = []
    total_files = sum(len(files) for _, _, files in os.walk("image_embeddings"))
    with tqdm(total=total_files, desc="Searching files") as pbar:
        for root, dirs, files in os.walk("image_embeddings"):
            for file in files:
                if file.endswith('.pt'):
                    image_embedding = torch.load(os.path.join(root, file)).to(device)
                    similarity_score = torch.nn.functional.cosine_similarity(image_embedding, text_features)
                    
                    if len(top_images) < 3:
                        top_images.append((os.path.join(root, file), similarity_score.item()))
                        top_images.sort(key=lambda x: x[1], reverse=True)
                    else:
                        min_score_index = min(range(len(top_images)), key=lambda i: top_images[i][1])
                        if similarity_score > top_images[min_score_index][1]:
                            top_images[min_score_index] = (os.path.join(root, file), similarity_score.item())
                            top_images.sort(key=lambda x: x[1], reverse=True)
                    
                    pbar.update(1)
                    
    result_images = []
    for image_path, score in top_images:
        formatted_image_path = image_path.replace('.pt', '').replace('image_embeddings', folder_name)
        image = Image.open(formatted_image_path)
        
        result_images.append((formatted_image_path, image, score))
 

    return result_images

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_query = request.form['text_query']

        folder_name = "static"
        top_images = get_images(text_query, folder_name)

        if top_images:
            return render_template('index.html', text_query = text_query, top_images=top_images)
        else:
            return "No matching images found."
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

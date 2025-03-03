from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import io
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)

embeddings = torch.load("static/catname_to_vector.pt")

def load_model():
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    return mtcnn, resnet

def get_image_embedding(image):
    mtcnn, resnet = load_model()
    boxes, _  = mtcnn.detect(image)
    if boxes is None:
        return None
    aligned = mtcnn(image)
    embeddings = resnet(aligned).detach()
    return embeddings

def find_best_match(user_embeddings, stuff_embeddings):
    best_match = None
    best_distance = float('inf')
    for key, value in stuff_embeddings.items():
        distance = (user_embeddings - value).norm().item()
        if distance < best_distance:
            best_match = key
            best_distance = distance
    return best_match

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return "No file part", 400
        file = request.files['photo']
        if file.filename == '':
            return "No selected file", 400
        
        # Read image and convert to processing format
        image = Image.open(io.BytesIO(file.read()))
        user_embeddings = get_image_embedding(image)
        best_match = find_best_match(user_embeddings, embeddings)
        return render_template('result.html', meme_cat=best_match)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

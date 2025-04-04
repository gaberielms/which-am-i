from flask import Flask, request, render_template

app = Flask(__name__)

import face_recognition
import numpy as np
import random
from collections import defaultdict

embeddings = defaultdict(list) # {img_embedding: int(1-20)}

def get_image_embedding(image):
    img = face_recognition.load_image_file(image)
    face_locations = face_recognition.face_locations(img)
    if len(face_locations) == 0:
        return None
    face_encodings = face_recognition.face_encodings(img, face_locations)
    return face_encodings[0]

def lowest_distance(image):
    image_embedding = get_image_embedding(image)
    if image_embedding is None:
        return "No face detected"
    best_match = None
    best_distance = float("inf")
    for val, img in embeddings.items():
        distance = np.linalg.norm(image_embedding - img)
        if distance < best_distance:
            best_distance = distance
            best_match = val
    return (best_match, best_distance)

def get_best_match(image):
    best_match, best_distance = lowest_distance(image)
    if best_match is None:
        return update_embeddings(get_image_embedding(image))
    elif best_distance > 0.5:
        return update_embeddings(get_image_embedding(image))
    return best_match

def update_embeddings(image):
    val = random.randint(1, 20)
    if len(embeddings) == 20:
        embeddings.popitem()
    while val in embeddings:
        val = random.randint(1, 20)
    embeddings[val] = image
    return val


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return "No file part", 400
        file = request.files['photo']
        if file.filename == '':
            return "No selected file", 400
        
        # Read image and convert to processing format
        image = file
        user_embeddings = get_image_embedding(image)
        if user_embeddings is None:
            return "No face detected", 400
        best_match = get_best_match(image)
        cat = 'cat' + str(best_match)
        return render_template('result.html', meme_cat=cat)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

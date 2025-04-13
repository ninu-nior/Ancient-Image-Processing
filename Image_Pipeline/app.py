from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
from flask_cors import CORS
from main_image_processing import main_func
import cv2
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    print("fd")
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    print(filepath)
    # Process the image (dummy processing: create grayscale and inverted version)
    image = Image.open(filepath)
    op_path=main_func(filepath)
    image=cv2.imread(op_path)
    # grayscale = ImageOps.grayscale(image)
    # inverted = ImageOps.invert(grayscale)
    
    processed_paths = []
    for i, img in enumerate([image, image], start=1):
        processed_filename = f'processed_{i}_{filename}'
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        img.save(processed_filepath)
        processed_paths.append(processed_filename)
    
    return jsonify({
        "language":"Eng",
        'message': 'Processing complete',
        'images': [f'/api/processed/{img}' for img in processed_paths],
        'description': 'Generated grayscale and inverted images.'
    })

@app.route('/api/processed/<filename>')
def get_processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

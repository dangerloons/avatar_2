from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(  # type: ignore
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and get face mesh
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        raise ValueError("No face detected in the image")
    
    # Get the first face's landmarks
    face_landmarks = results.multi_face_landmarks[0]
    
    # Convert landmarks to 3D points
    points_3d = []
    for landmark in face_landmarks.landmark:
        points_3d.append([landmark.x, landmark.y, landmark.z])
    
    points_3d = np.array(points_3d)
    
    # Create the OBJ file
    obj_path = image_path.replace(Path(image_path).suffix, '.obj')
    with open(obj_path, 'w') as f:
        # Write vertices
        for point in points_3d:
            f.write(f'v {point[0]} {point[1]} {point[2]}\n')
        
        # Write faces using MediaPipe's face mesh triangles
        for triangle in mp_face_mesh.FACEMESH_TRIANGLES:
            # OBJ files are 1-indexed
            f.write(f'f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n')
    
    return obj_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/generate3d', methods=['POST'])
def generate_3d():
    print("Received generate3d request")  # Debug log
    if 'image' not in request.files:
        print("No image file in request")  # Debug log
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    print(f"Received file: {file.filename}")  # Debug log
    
    if file.filename == '':
        print("Empty filename")  # Debug log
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {image_path}")  # Debug log
            file.save(image_path)
            
            print("Processing image...")  # Debug log
            obj_path = process_image(image_path)
            print(f"Generated OBJ file at: {obj_path}")  # Debug log
            
            return send_file(obj_path, as_attachment=True, download_name='model.obj')
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")  # Debug log
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
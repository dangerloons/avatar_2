from flask import Flask, request, jsonify, send_file
import mediapipe as mp
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from pathlib import Path
from scipy.spatial import Delaunay
import zipfile

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def create_better_mesh(points_3d, image):
    # Project 3D points to 2D for better triangulation
    points_2d = points_3d[:, :2]
    
    # Create Delaunay triangulation in 2D
    tri = Delaunay(points_2d)
    
    # Get image dimensions for UV mapping
    height, width = image.shape[:2]
    
    # Calculate UVs
    uvs = points_2d.copy()
    uvs[:, 0] /= width  # Normalize X coordinates
    uvs[:, 1] = 1 - (uvs[:, 1] / height)  # Normalize and flip Y coordinates
    
    return tri.simplices, uvs

@app.route('/')
def home():
    return '''
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*">
        <input type="submit" value="Generate 3D Model">
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        
        # Read and process image
        image = cv2.imread(image_path)
        if image is None:
            return 'Could not read image', 400
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return 'No face detected in image', 400
        
        # Get landmarks and convert to numpy array
        face_landmarks = results.multi_face_landmarks[0]
        points_3d = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
        
        # Scale and center the mesh
        points_3d = points_3d - np.mean(points_3d, axis=0)
        scale = 1.0 / np.max(np.abs(points_3d))
        points_3d = points_3d * scale
        
        # Create better triangulation and UV mapping
        triangles, uvs = create_better_mesh(points_3d, image)
        
        # Save the texture image
        texture_path = image_path.replace(Path(image_path).suffix, '_texture.png')
        cv2.imwrite(texture_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        
        # Create OBJ file with materials
        obj_path = image_path.replace(Path(image_path).suffix, '.obj')
        mtl_path = image_path.replace(Path(image_path).suffix, '.mtl')
        
        # Write MTL file
        with open(mtl_path, 'w') as f:
            f.write("newmtl material0\n")
            f.write("Ka 1.000000 1.000000 1.000000\n")
            f.write("Kd 1.000000 1.000000 1.000000\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write(f"map_Kd {Path(texture_path).name}\n")
        
        # Write OBJ file
        with open(obj_path, 'w') as f:
            f.write(f"mtllib {Path(mtl_path).name}\n")
            
            # Write vertices
            for point in points_3d:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
            
            # Write texture coordinates
            for uv in uvs:
                f.write(f"vt {uv[0]} {uv[1]}\n")
            
            # Write faces with material and UV mapping
            f.write("usemtl material0\n")
            for triangle in triangles:
                # OBJ indices are 1-based
                f.write(f"f {triangle[0]+1}/{triangle[0]+1} {triangle[1]+1}/{triangle[1]+1} {triangle[2]+1}/{triangle[2]+1}\n")
        
        print(f"OBJ file created at: {obj_path}")
        
        # Create ZIP file with all necessary files
        zip_path = image_path.replace(Path(image_path).suffix, '.zip')
        base_dir = os.path.dirname(obj_path)
        files_to_zip = [Path(obj_path).name, Path(mtl_path).name, Path(texture_path).name]

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in files_to_zip:
                file_path = os.path.join(base_dir, file)
                zipf.write(file_path, file)
        
        return send_file(zip_path, as_attachment=True, download_name='face_model.zip')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return f'Error: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)
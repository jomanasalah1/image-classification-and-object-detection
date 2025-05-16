import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from object_detection import (
    Yolo_v3, load_images, load_class_names, 
    draw_boxes, load_weights
)
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Constants
_MODEL_SIZE = (416, 416)
WEIGHTS_PATH = 'yolov3.weights'
CLASS_NAMES_PATH = 'coco.names'

# Global variables for the detection model
detection_session = None
detection_graph = None
input_placeholder = None
output_tensor = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_detection_model():
    global detection_session, detection_graph, input_placeholder, output_tensor
    
    if detection_session is None:
        class_names = load_class_names(CLASS_NAMES_PATH)
        n_classes = len(class_names)
        
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            model = Yolo_v3(
                n_classes=n_classes,
                model_size=_MODEL_SIZE,
                max_output_size=10,
                iou_threshold=0.5,
                confidence_threshold=0.5
            )
            
            input_placeholder = tf.compat.v1.placeholder(
                tf.float32, [1, 416, 416, 3], name='input')
            
            detections = model(input_placeholder, training=False)
            
            model_vars = tf.compat.v1.global_variables(scope='yolo_v3_model')
            assign_ops = load_weights(model_vars, WEIGHTS_PATH)
            
            detection_session = tf.compat.v1.Session(graph=detection_graph)
            detection_session.run(assign_ops)
            
            output_tensor = detections
            
        print("Object detection model initialized successfully")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            init_detection_model()
            
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
            
            file.save(upload_path)
            
            try:
                batch = load_images([upload_path], model_size=_MODEL_SIZE)
                
                with detection_graph.as_default():
                    detection_result = detection_session.run(
                        output_tensor,
                        feed_dict={input_placeholder: batch}
                    )
                
                class_names = load_class_names(CLASS_NAMES_PATH)
                draw_boxes([upload_path], detection_result, class_names, _MODEL_SIZE)
                
                os.replace(os.path.join('output', filename), output_path)
                
                return render_template('results.html',
                                    original_image=url_for('static', filename=f'uploads/{filename}'),
                                    processed_image=url_for('static', filename=f'outputs/{filename}'))
            
            except Exception as e:
                flash(f'Error during detection: {str(e)}')
                return redirect(url_for('index'))
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/outputs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port= 5001)
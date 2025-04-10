from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
import numpy as np
import cv2

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="terrain_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ['Grassy_Terrain', 'Marshy_Terrain', 'Other_Image', 'Rocky_Terrain', 'Sandy_Terrain']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def detect_roughness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    return np.std(gradient_magnitude) / np.mean(gradient_magnitude)

def slipperiness_percentage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    threshold_value = 5000
    return max(0, ((threshold_value - variance) / threshold_value) * 100)

def predict_terrain(img_path):
    image_for_analysis = cv2.imread(img_path)
    img_array = preprocess_image(img_path)

    # Set the tensor to the image
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_labels[np.argmax(output_data)].lower()
    roughness_lvl = detect_roughness(image_for_analysis)
    slipperiness_lvl = slipperiness_percentage(image_for_analysis)
    return predicted_class, roughness_lvl, slipperiness_lvl

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    predicted_class, roughness, slipperiness = predict_terrain(file_path)
    
    return jsonify({
        'terrain': predicted_class,
        'roughness': roughness,
        'slipperiness': slipperiness,
        'image_url': file_path
    })

if __name__ == '__main__':
    app.run(debug=True)

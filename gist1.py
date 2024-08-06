from flask import Flask, request, render_template_string, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  

# folder for uploaded files 
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('BrainTumour10Epochs.h5')


INPUT_SIZE = 64

# test data loaded 
def load_test_data():
    test_images = []
    test_labels = []

    for i, image_name in enumerate(os.listdir('brain_tumor_dataset/yes/')):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread('brain_tumor_dataset/yes/'+image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            test_images.append(np.array(image))
            test_labels.append(1)

    for i, image_name in enumerate(os.listdir('brain_tumor_dataset/no/')):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread('brain_tumor_dataset/no/'+image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            test_images.append(np.array(image))
            test_labels.append(0)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return test_images, test_labels


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result, accuracy = predict_tumor(file_path)
            return render_template_string(result_template, prediction=result, accuracy=accuracy)
    return render_template_string(upload_template)

#processing image and prediction 
def predict_tumor(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    img = np.array(image)
    input_img = np.expand_dims(img, axis=0)
    input_img = input_img / 255.0
    predictions = model.predict(input_img)
    result = 'Tumor' if predictions[0][0] > 0.5 else 'No Tumor'
    
    test_images, test_labels = load_test_data()
    
    y_pred = []
    for image in test_images:
        prediction = model.predict(np.expand_dims(image, axis=0))
        y_pred.append(int(np.round(prediction[0, 0])))
    
    accuracy = calculate_accuracy(test_labels, y_pred)
    
    return result, accuracy

upload_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <title>Brain Tumor Detection</title>
    <style>
      body {
        background-color: #f8f9fa;
      }
      .container {
        margin-top: 50px;
        max-width: 500px;
      }
      .card {
        padding: 20px;
        border-radius: 10px;
      }
      .logo {
        max-width: 100px; 
        margin-bottom: 20px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <img src="{{ url_for('static', filename='logoamity.png') }}" alt="Logo" class="logo">
        <h1 class="text-center">Brain Tumor Detection</h1>
        <form method="post" enctype="multipart/form-data">
          <div class="form-group">
            <label for="file">Upload an MRI Image</label>
            <input type="file" class="form-control-file" name="file" accept="image/*" required>
          </div>
          <button type="submit" class="btn btn-primary btn-block">Upload</button>
        </form>
      </div>
    </div>
  </body>
</html>
"""

result_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <title>Prediction Result</title>
    <style>
      body {
        background-color: #f8f9fa;
      }
      .container {
        margin-top: 50px;
        max-width: 500px;
      }
      .card {
        padding: 20px;
        border-radius: 10px;
      }
      .result {
        margin-top: 20px;
      }
      .logo {
        max-width: 100px; 
        margin-bottom: 20px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <img src="{{ url_for('static', filename='logoamity.png') }}" alt="Logo" class="logo">
        <h1 class="text-center">Prediction Result</h1>
        <p class="text-center result">Prediction: {{ prediction }}</p>
        <p class="text-center result">Accuracy: {{ accuracy }}</p>
        <a href="{{ url_for('upload_file') }}" class="btn btn-primary btn-block">Upload Another Image</a>
      </div>
    </div>
  </body>
</html>
"""

if __name__ == "__main__":
    app.run()

from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory
import json
import cv2
import numpy as np
import io
import keras

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')


@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files['file']
    processed_data = preprocessing(file)
    res = json.dumps({"image": processed_data.tolist()})
    return res


@app.route('/model')
def model():
    json_data = json.load(open("./web_model/model.json"))
    return jsonify(json_data)


@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('web_model', path)


def preprocessing(file):
    # Save file to memory
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)

    # Get data from file
    # original_data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    # original_img_bgr = cv2.imdecode(original_data, 1)     # flag=1: Load bgr channels
    # original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
    # resized_img = cv2.resize(original_img_rgb, dsize=(224, 224))
    resized_img = keras.preprocessing.image.load_img('../image-quality-assessment/src/tests/test_images/42039.jpg', target_size=(224, 224))

    img_data = np.asarray(resized_img)
    preprocessed_img = keras.applications.mobilenet.preprocess_input(img_data)
    return preprocessed_img


if __name__ == "__main__":
    app.run()

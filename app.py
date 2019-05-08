from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory
import json
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')


@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files['file']
    res = preprocessing(file)
    return json.dumps({"image": res.tolist()})


@app.route('/model')
def model():
    json_data = json.load(open("./web_model/model.json"))
    return jsonify(json_data)


@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('web_model', path)


def preprocessing(file):
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 0)
    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    return res


if __name__ == "__main__":
    app.run()

from flask import Flask, request, jsonify, send_from_directory
import utility
import os

app = Flask(__name__, static_folder='../UI')

@app.route('/')
def serve_index():
    return send_from_directory('../UI', 'app.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../UI', path)

@app.route('/classify_image', methods = ['POST'])
def classify_image():
    image_data = request.form['image_data']

    response = jsonify(utility.classify_image(image_data))

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Server for Bollywood Actor Classification")
    utility.load_saved_artifacts()
    app.run(host='0.0.0.0', port=5000)
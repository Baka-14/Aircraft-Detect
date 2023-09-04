import os
import io
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import base64

app = Flask(__name__)

# Load your YOLOv8 model with the appropriate model file path
model = YOLO('model.pt')

@app.route("/", methods=["POST"])
def detect_objects():
    # Check if a POST request with an image is received
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    image_file = request.files['image']

    # Check if the file name is empty
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the image file and perform object detection
    image = Image.open(io.BytesIO(image_file.read()))
    results = model(image)

    # Extract bounding boxes and class labels
    boxes = results.xyxy[0].cpu().numpy()
    classes = results.pred[0][:, -1].cpu().numpy()

    # Get class names (replace with your own class names if needed)
    class_names = {
        0: 'person',
        1: 'bicycle',
        # Add more class names here...
    }

    # Prepare the response with detected objects and their class labels
    detected_objects = []
    for bbox, class_id in zip(boxes, classes):
        class_label = class_names.get(int(class_id), 'unknown')
        detected_objects.append({
            "class": class_label,
            "bbox": bbox.tolist()
        })

    # Save the processed image with bounding boxes (you can customize this)
    processed_image = results.render()[0]

    # Convert the processed image to a base64-encoded string
    buffered = io.BytesIO()
    processed_image.save(buffered, format="PNG")
    processed_image_base64 = base64.b64encode(buffered.getvalue()).decode()

    response_data = {
        "detected_objects": detected_objects,
        "processed_image": processed_image_base64
    }

    return jsonify(response_data), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

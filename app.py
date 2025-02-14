from flask import Flask, request, render_template, jsonify
import requests
import base64
from PIL import Image, ImageDraw
import io

app = Flask(__name__)

PREDICTION_KEY = "EBU533jfLqn2TevjXaqfAD2i0WTiTAiFjQcNwFLp2AmgUnfvQWmwJQQJ99BBACYeBjFXJ3w3AAAIACOGEZx2"
PROJECT_ID = "9328bc74-432d-4f21-8c6a-bbaf614c81cc"
ITERATION_NAME = "Iteration17"
ENDPOINT = f"https://jiawen-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/{PROJECT_ID}/detect/iterations/{ITERATION_NAME}/image"


HEADERS = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream"
}

ALLOWED_TAGS = {"Angle Carrier Board", "Insert in correct position", "Misplaced Insert", "Missing insert"}

def encode_image(img):
    """Convert image to Base64 for sending to frontend"""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.route('/')
def main_home():
    return render_template('UI.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['frame']
    image_bytes = file.read()

    try:
        # Send to Azure for prediction
        response = requests.post(ENDPOINT, headers=HEADERS, data=image_bytes)
        response.raise_for_status()
        result = response.json()

        # Open image for drawing
        img = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(img)

        filtered_predictions = []
        for prediction in result.get("predictions", []):
            tag_name = prediction["tagName"]
            probability = prediction["probability"]
            bounding_box = prediction["boundingBox"]

            if tag_name in ALLOWED_TAGS and probability > 0.5:
                # Convert bounding box to image coordinates
                left = bounding_box["left"] * img.width
                top = bounding_box["top"] * img.height
                right = (bounding_box["left"] + bounding_box["width"]) * img.width
                bottom = (bounding_box["top"] + bounding_box["height"]) * img.height

                # Draw bounding box
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                draw.text((left, top), f"{tag_name}: {probability:.2f}", fill="red")

                filtered_predictions.append({
                    "tagName": tag_name,
                    "probability": probability,
                    "boundingBox": bounding_box
                })

        # Convert image to Base64
        img_base64 = encode_image(img)

        return jsonify({"image_data": img_base64, "predictions": filtered_predictions})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

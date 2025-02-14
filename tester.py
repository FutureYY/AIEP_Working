from flask import Flask, request, render_template, jsonify
import requests

app = Flask(__name__)

# Azure Custom Vision API details
PREDICTION_KEY = "10jnoKfrdfDiVn6w35ngOnpx0hEAA8k84A0gxToPCNH1qUKWyuoOJQQJ99BAACYeBjFXJ3w3AAAIACOGzGnt"
PROJECT_ID = "9328bc74-432d-4f21-8c6a-bbaf614c81cc"
ITERATION_NAME = "Iteration12"
ENDPOINT = f"https://abcdetectorresource-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/{PROJECT_ID}/detect/iterations/{ITERATION_NAME}/image"

HEADERS = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["file"]
        if image:
            # Send the image to Azure Custom Vision for object detection
            response = requests.post(
                ENDPOINT, 
                headers=HEADERS, 
                data=image.read()
            )
            result = response.json()  # Get the response from the API
            return jsonify(result)  # Return the result as JSON response
    
    return render_template("UI.html")

if __name__ == "__main__":
    app.run(debug=True)

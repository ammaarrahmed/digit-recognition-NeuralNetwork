from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

# Load the trained model
model = load_model("digit_recognition_model.h5")

# Initialize Flask app
app = Flask(__name__)

# Define a route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded. Please upload an image."
    
    # Get the uploaded file
    file = request.files["file"]
    if file.filename == "":
        return "No file selected. Please upload an image."
    
    # Save the uploaded file temporarily
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Preprocess the image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image_resized = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
    image_normalized = image_resized.astype("float32") / 255.0  # Normalize
    image_flattened = image_normalized.reshape(1, 28 * 28)  # Flatten the image

    # Predict the digit
    predictions = model.predict(image_flattened)
    predicted_label = np.argmax(predictions, axis=1)[0]

    # Remove the temporary file
    os.remove(file_path)

    return f"The predicted digit is: {predicted_label}"

if __name__ == "__main__":
    app.run(debug=True)
import cv2
import pytesseract
import numpy as np
from flask import Flask, request, jsonify, render_template

# Set the path for Tesseract executable (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the uploaded image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(img)

    # Return the extracted text as a JSON response
    return jsonify({'extracted_text': extracted_text})

if __name__ == '__main__':
    app.run(debug=True)

# Medicine Recommendation System

A web-based system designed to recommend medicines based on user input. The system supports recommendations through three modes:
- **Medicine Name Suggestion**: Suggests medicines based on partial input.
- **Condition-Based Recommendation**: Recommends medicines for a specified condition or symptom.
- **OCR (Optical Character Recognition)**: Extracts text from uploaded images to recommend medicines based on the recognized text.

## Features
- **Search by Medicine Name**: Suggests alternative medicines based on partial input using exact and fuzzy matching techniques.
- **Recommendation by Use**: Recommends medicines based on a given condition or symptom.
- **OCR Extraction**: Upload an image, and the system will use OCR to recognize text on a tablet or packaging, then suggest medicines based on the recognized text.

## Tech Stack
- **Backend**: Flask (Python Web Framework)
- **Machine Learning**: TfidfVectorizer for similarity computation, cosine similarity for finding similar medicines.
- **OCR**: Tesseract (for extracting text from images)
- **Frontend**: HTML, CSS, JavaScript (for handling the UI and AJAX-based dynamic suggestions)

## Installation

### Prerequisites
Ensure you have Python installed. The recommended version is 3.6 or higher.

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/medicine-recommendation-system.git
    ```

2. Navigate to the project folder:
    ```bash
    cd medicine-recommendation-system
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Install Tesseract (OCR tool):
    - **Windows**: Download and install Tesseract from [here](https://github.com/UB-Mannheim/tesseract/wiki).
    - After installation, make sure to set the path to the Tesseract executable in the `app.py` file:
        ```python
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ```

### Running the Application

1. Run the Flask application:
    ```bash
    python app.py
    ```

2. The application will be accessible at `http://127.0.0.1:5000/`.

## Usage

### 1. **Medicine Name Suggestion**
- Enter a medicine name or a partial name in the search bar to get suggestions of alternative medicines.

### 2. **Condition-Based Recommendation**
- Enter a medical condition or symptom to receive recommendations for medicines that treat that condition.

### 3. **OCR-based Recommendations**
- Upload an image of a medicine or its packaging. The system will use Optical Character Recognition (OCR) to extract the text and recommend related medicines.

## Files Overview
- `app.py`: Main Flask application handling the routes and business logic.
- `templates/`: Contains HTML files for rendering the frontend (`index.html`, `result.html`).
- `static/`: Contains static files like CSS (`Home.css`) and JavaScript (`script.js`).
- `data/med_data.csv`: CSV file containing the medicine data (e.g., name, composition, uses, side effects).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
- **Name**: Your Name
- **Email**: your-email@example.com
- **GitHub**: [GitHub Link](https://github.com/your-username)
- **LinkedIn**: [LinkedIn Link](https://www.linkedin.com/in/your-profile/)


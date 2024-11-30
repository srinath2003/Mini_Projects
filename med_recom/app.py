import cv2
import pytesseract
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("data/med_data")


# Generate TF-IDF matrix for similarity calculations
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Suggest medicine names based on partial input
def suggest_medicine_names(partial_name):
    medicine_names = [name for name in df['Medicine Name'].tolist()]
    
    # Step 1: Look for exact matches first
    exact_matches = [name for name in medicine_names if partial_name in name]
    if exact_matches:
        return [df['Medicine Name'].iloc[medicine_names.index(m)] for m in exact_matches]
    
    # Step 2: If no exact match, find close matches using difflib
    matches = difflib.get_close_matches(partial_name, medicine_names, n=5, cutoff=0.3)
    return [df['Medicine Name'].iloc[medicine_names.index(m)] for m in matches]
# Recommend medicines based on a specific condition (use)
def recommend_medicine_by_use(condition):
    # Filter medicines that contain the specified condition in the 'Uses' column
    relevant_medicines = df[df['Uses'].str.contains(condition, case=False, na=False)]
    
    if relevant_medicines.empty:
        return "No medicines found for this condition."
    
    # Return a list of relevant medicines (top 5, you can change the number if needed)
    return relevant_medicines[['Medicine Name', 'Composition', 'Uses', 'Side_effects']].head(10).to_dict(orient='records')


# Recommend similar medicines based on a selected medicine
def recommend_medicines(medicine_name):
    medicine_name = medicine_name.lower()

     # Check if the input medicine exists in the dataset
    if medicine_name not in df['Medicine Name'].str.lower().values:
        return "Medicine not found in the dataset."

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the 'Uses' column
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Uses'])

    # Get the index of the input medicine
    idx = df[df['Medicine Name'] == medicine_name].index[0]

    # Compute cosine similarity with all medicines
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get the top 5 similar medicines excluding the input medicine itself
    similar_indices = cosine_sim.argsort()[-6:][::-1]  # Get top 6, reverse for descending order
    similar_indices = [i for i in similar_indices if i != idx][:5]  # Exclude the input medicine

    # Create a list of recommendations with relevant details
    recommendations = []
    for i in similar_indices:
        recommendations.append({
            'Medicine Name': df.iloc[i]['Medicine Name'],
            'Composition': df.iloc[i]['Composition'],
            'Uses': df.iloc[i]['Uses'],
            'Side_effects': df.iloc[i]['Side_effects'],
            'Similarity Score': cosine_sim[i]
        })

    return recommendations
def suggest_medicine_for_ocr(extracted_text):
    try:
        # Extract all entries from the matching column
        matching_entries = df['col_for_ocr'].fillna('').tolist()
        
        # Calculate similarity scores
        scores = difflib.get_close_matches(extracted_text, matching_entries, n=5, cutoff=0.5)
        
        if not scores:
            return {"error": "No matching medicines found for the extracted text."}

        # Find the indices of the top matches
        matches = df[df['col_for_ocr'].isin(scores)]

        # Convert matching entries to a list of dictionaries
        recommendations = matches[['Medicine Name', 'Composition', 'Uses', 'Side_effects']].to_dict(orient='records')
        
        return recommendations

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggest', methods=['POST'])
def suggest():
    partial_name = request.form['medicine_name']
    suggestions = suggest_medicine_names(partial_name)

    # Fetch full details for the suggestions
    suggestion_details = []
    for suggestion in suggestions:
        medicine_details = df[df['Medicine Name'] == suggestion].iloc[0]
        suggestion_details.append({
            'Medicine Name': medicine_details['Medicine Name'],
            'Composition': medicine_details['Composition'],
            'Uses': medicine_details['Uses'],
            'Side_effects': medicine_details['Side_effects'],
        })

    # Show alternative medicines based on the first suggestion
    if suggestion_details:
        # Get alternative recommendations based on the composition or other features of the first suggestion
        first_suggestion = suggestion_details[0]['Medicine Name']
        recommendations = recommend_medicines(first_suggestion)
    else:
        recommendations = []
    # print(suggestion_details,"NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
    # print("ooooooooooooooooo",recommendations)

    return render_template('result.html', suggestions=suggestion_details, recommendations=recommendations, search_type="name", query=partial_name)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    partial_name = request.args.get('query', '').lower()
    suggestions = suggest_medicine_names(partial_name)
    return jsonify(suggestions=suggestions)

@app.route('/recommend', methods=['POST'])
def recommend():
    medicine_name = request.form['selected_medicine']
    recommendations = recommend_medicines(medicine_name)
    return render_template('result.html', recommendations=recommendations, search_type="recommend", query=medicine_name)

@app.route('/recommend_by_use', methods=['POST'])
def recommend_by_use():
    condition = request.form['condition'].strip().lower()  # Convert condition to lowercase and strip extra spaces
    recommendations = recommend_medicine_by_use(condition)
    
    # If no recommendations, return an appropriate message to the frontend
    if isinstance(recommendations, str):  # If it's a string (error message)
        return render_template('result.html', recommendations=[{'Medicine Name': recommendations}], search_type="condition", query=condition)
    
    return render_template('result.html', recommendations=recommendations, search_type="condition", query=condition)


# Set the path for Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# OCR Extraction Route
@app.route('/ocr_extract', methods=['POST'])
def ocr_extract():
    if 'file' not in request.files:
        return render_template('result.html', error="No file uploaded", search_type="ocr", query="")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="No file selected", search_type="ocr", query="")

    try:
        # Read the uploaded image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(img).strip().lower()

        if not extracted_text:
            return render_template('result.html', error="No text found in the image", search_type="ocr", query="")

        # Suggest medicines based on extracted text
        suggestions = suggest_medicine_for_ocr(extracted_text)
    
        if isinstance(suggestions, dict) and "error" in suggestions:
            return render_template('result.html', error=suggestions["error"], search_type="ocr", query=extracted_text)

        # Render the result template with suggestions
        return render_template(
            'result.html',
            query=extracted_text,
            search_type="ocr",
            suggestions=suggestions
        )

    except Exception as e:
        return render_template('result.html', error=f"An error occurred: {str(e)}", search_type="ocr", query="")


# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
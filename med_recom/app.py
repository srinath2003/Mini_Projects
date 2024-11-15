from flask import Flask, render_template, request, jsonify
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("data/data.csv")

# Ensure the 'combined_features' column exists in the dataset
if 'combined_features' not in df.columns:
    df['combined_features'] = df['Medicine Name'] + ' ' + df['Composition'] + ' ' + df['Uses'] + ' ' + df['Side_effects']

# Generate TF-IDF matrix for similarity calculations
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Suggest medicine names based on partial input
def suggest_medicine_names(partial_name):
    partial_name = partial_name.lower()
    medicine_names = [name.lower() for name in df['Medicine Name'].tolist()]
    
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
    return relevant_medicines[['Medicine Name', 'Composition', 'Uses', 'Side_effects']].head(5).to_dict(orient='records')


# Recommend similar medicines based on a selected medicine
def recommend_medicines(medicine_name):
    medicine_name = medicine_name.lower()

    if medicine_name not in df['Medicine Name'].str.lower().values:
        return "Medicine not found in the dataset."

    idx = df[df['Medicine Name'].str.lower() == medicine_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores[1:6], key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in sim_scores:
        recommendations.append({
            'Medicine Name': df.iloc[i]['Medicine Name'],
            'Composition': df.iloc[i]['Composition'],
            'Uses': df.iloc[i]['Uses'],
            'Side_effects': df.iloc[i]['Side_effects'],
            'Similarity Score': score
        })

    return recommendations

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
        medicine_details = df[df['Medicine Name'].str.lower() == suggestion.lower()].iloc[0]
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


if __name__ == '__main__':
    app.run(debug=True)

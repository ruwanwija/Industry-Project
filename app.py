from flask import Flask, request, render_template, jsonify, redirect, url_for
import pickle
import firebase_admin
from firebase_admin import credentials, firestore
import nltk
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from calendar import monthrange

app = Flask(__name__)

# Configure NLTK path and download resources if needed
nltk_data_path = 'C:\\Users\\Ruwan Wijayasundara\\AppData\\Roaming\\nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# Load the pre-trained SVM model and vectorizer
model, vectorizer = None, None
try:
    with open('svm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
except FileNotFoundError as e:
    print(f"Error loading model/vectorizer: {e}")

# Initialize Firebase Admin SDK
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Error initializing Firebase: {e}")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def get_month_end_date(year_month):
    year, month = map(int, year_month.split("-"))
    last_day = monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last_day}"

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/insert')
def insert_page():
    return render_template('review.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded."}), 500

    review = request.form.get('review', '')
    date = request.form.get('date', '')

    # Validate date format (YYYY-MM-DD) for Firestore consistency
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    processed_review = preprocess_text(review)
    review_vector = vectorizer.transform([processed_review])

    # Get prediction probabilities and labels
    try:
        probabilities = model.predict_proba(review_vector)[0]
    except AttributeError:
        return jsonify({"error": "Model does not support probability prediction."}), 500

    labels = ["Location", "Food Quality", "Value for Money", "Comfort", "Staff Behavior"]
    probability_dict = {labels[i]: probabilities[i] for i in range(len(labels))}
    highest_label = max(probability_dict, key=probability_dict.get)
    highest_probability = probability_dict[highest_label]

    # Add to Firestore
    try:
        db.collection('reviews').add({
            'review': review,
            'date': date,
            'highest_label': highest_label,
            'highest_probability': highest_probability,
            'probabilities': probability_dict
        })
    except Exception as e:
        return jsonify({"error": f"Failed to save to Firestore: {e}"}), 500

    # Redirect to success page instead of returning JSON
    return redirect(url_for('success_page'))

@app.route('/success')
def success_page():
    return render_template('success.html')

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

@app.route('/get_data', methods=['GET'])
def get_data():
    month = request.args.get('month', 'all')
    label_counts = {label: 0 for label in ["Location", "Food Quality", "Value for Money", "Comfort", "Staff Behavior"]}

    try:
        if month == 'all':
            docs = db.collection('reviews').stream()
        else:
            start_date = f"{month}-01"
            end_date = get_month_end_date(month)
            docs = db.collection('reviews').where('date', '>=', start_date).where('date', '<=', end_date).stream()

        for doc in docs:
            highest_label = doc.to_dict().get('highest_label')
            if highest_label in label_counts:
                label_counts[highest_label] += 1
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve data from Firestore: {e}"}), 500

    return jsonify(label_counts)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify
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

app = Flask(__name__)

# Set NLTK Data Path
os.environ['NLTK_DATA'] = 'C:\\Users\\Ruwan Wijayasundara\\AppData\\Roaming\\nltk_data'

# Download the required resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained SVM model and vectorizer
try:
    with open('svm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    print(f"Processed text: {tokens}")
    return ' '.join(tokens)

# Home route
@app.route('/')
def home_page():
    return render_template('home.html')

# Insert review route
@app.route('/insert')
def home():
    return render_template('review.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        review = request.form['review']
        date = request.form['date']
        print(f"Received review: {review} with date: {date}")

        # Preprocess the review text
        processed_review = preprocess_text(review)
        review_vector = vectorizer.transform([processed_review])

        # Get prediction probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(review_vector)[0]
        else:
            return jsonify({"error": "Model does not support probability predictions."}), 500

        labels = ["Location", "Food Quality", "Value for Money", "Comfort", "Staff Behavior"]
        probability_dict = {labels[i]: probabilities[i] for i in range(len(labels))}
        highest_label = max(probability_dict, key=probability_dict.get)
        highest_probability = probability_dict[highest_label]
        print(f"Prediction result - Label: {highest_label}, Probability: {highest_probability}")

        # Save data to Firestore
        db.collection('reviews').add({
            'review': review,
            'date': date,
            'highest_label': highest_label,
            'highest_probability': highest_probability,
            'probabilities': probability_dict
        })
        print("Data saved to Firebase.")

        return jsonify({
            'highest_label': highest_label,
            'highest_probability': highest_probability,
            'probabilities': probability_dict
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Visualization route
@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

# Route to fetch review data by month or all data
@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        month = request.args.get('month', 'all')
        print(f"Fetching data for month: {month}")
        label_counts = {
            "Location": 0, 
            "Food Quality": 0, 
            "Value for Money": 0, 
            "Comfort": 0, 
            "Staff Behavior": 0
        }

        # Query Firestore based on the specified month or get all data
        if month == 'all':
            docs = db.collection('reviews').stream()
        else:
            start_date = f"{month}-01"
            end_date = f"{month}-31"
            docs = db.collection('reviews').where('date', '>=', start_date).where('date', '<=', end_date).stream()

        # Count labels for the specified time period
        for doc in docs:
            highest_label = doc.to_dict().get('highest_label')
            if highest_label in label_counts:
                label_counts[highest_label] += 1

        print(f"Label counts for the specified period: {label_counts}")
        return jsonify(label_counts)
    except Exception as e:
        print(f"Error fetching data from Firestore: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

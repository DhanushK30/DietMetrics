from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import torch
import pytesseract
from PIL import Image
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__, static_folder='static')

# Load Nutri-Score Model using pickle instead of joblib
with open('nutri_score_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('nutri_score_features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('nutri_score_model_optimized.pkl', 'rb') as f:
    model_nutri = pickle.load(f)

# Load Food Ingredient Classifier Model (keeping this part unchanged)
MODEL_PATH = "distilbert_food_classifier.pth"
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model_nova = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
# Modified to use weights_only=True to avoid the security warning
model_nova.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True))
model_nova.eval()

# Features used in the model
FEATURES = [
    'energy-kcal_100g', 'proteins_100g', 'fat_100g', 'saturated-fat_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'salt_100g', 'sodium_100g'
]

# Function to calculate missing features
def calculate_missing_features(data):
    data['energy_density'] = data['energy-kcal_100g'] / 100
    data['protein_to_fat_ratio'] = data['proteins_100g'] / (data['fat_100g'] + 0.01)
    data['sugar_to_carb_ratio'] = data['sugars_100g'] / (data['carbohydrates_100g'] + 0.01)
    return data

# Function to determine Nutri-Score grade
def get_nutri_grade(score):
    if -15 <= score <= 0.99:
        return "A", "Green"
    elif 1 <= score <= 3.99:
        return "B", "#90EE90"
    elif 4 <= score <= 11.99:
        return "C", "#FFFF00"
    elif 12 <= score <= 18.99:
        return "D", "Orange"
    elif score >= 19:
        return "E", "Red"
    return "Unknown", "Gray"

# -------- ROUTES -------- #

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/choose')
def choose():
    return render_template('choose.html')

@app.route('/nova')
def nova():
    return render_template('nova.html')

@app.route('/score')
def score():
    return render_template('score.html')

@app.route('/predict_nutri', methods=['POST'])
def predict_nutri():
    try:
        # Get user input
        user_input = {f: float(request.form.get(f)) for f in FEATURES if request.form.get(f) is not None}
        
        # Convert to DataFrame
        X_df = pd.DataFrame([user_input])
        
        # Add missing features
        X_df = calculate_missing_features(X_df)
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in X_df.columns:
                X_df[feature] = 0
                
        # Ensure correct column order
        X_df = X_df[feature_names]

        # Apply pre-processing
        X_processed = preprocessor.transform(X_df)

        # Predict Nutri-Score
        score = float(model_nutri.predict(X_processed)[0])

        # Get grade and color
        grade, color = get_nutri_grade(score)

        return jsonify({'nutri_score': round(score, 2), 'grade': grade, 'color': color})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# ---- Food Ingredient Classification (NOVA) ---- #
def classify_ingredients(ingredients):
    inputs = tokenizer(ingredients, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model_nova(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction + 1

@app.route('/predict_nova', methods=['POST'])
def predict_nova():
    ingredients = request.form.get('ingredients')
    prediction = classify_ingredients(ingredients)
    return jsonify({"prediction": prediction})

@app.route('/ocr', methods=['POST'])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files["image"])
    extracted_text = pytesseract.image_to_string(img).strip()

    if not extracted_text:
        return jsonify({"error": "No text detected"}), 400

    prediction = classify_ingredients(extracted_text)
    return jsonify({"extracted_text": extracted_text, "prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
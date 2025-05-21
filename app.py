from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import torch
import pytesseract
from PIL import Image
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import requests

# --- Configuration for Asset Downloading ---
ASSET_DIR = "downloaded_assets" # Directory to store downloaded files
os.makedirs(ASSET_DIR, exist_ok=True)

# URLs for models and assets
DISTILBERT_MODEL_URL = "https://huggingface.co/DhanushK30/DietMetrics-Models/resolve/main/distilbert_food_classifier.pth"

# !!! IMPORTANT: Replace these with your ACTUAL GitHub Release URLs for the PKL files !!!
# These are placeholders. You need to create a release and get the real URLs.
NUTRI_PREPROCESSOR_URL = "https://github.com/DhanushK30/DietMetrics/releases/download/v0.1-assets/nutri_score_preprocessor.pkl"
NUTRI_FEATURES_URL = "https://github.com/DhanushK30/DietMetrics/releases/download/v0.1-assets/nutri_score_features.pkl"
NUTRI_MODEL_OPTIMIZED_URL = "https://github.com/DhanushK30/DietMetrics/releases/download/v0.1-assets/nutri_score_model_optimized.pkl"

# Local paths for downloaded assets
DISTILBERT_MODEL_PATH = os.path.join(ASSET_DIR, "distilbert_food_classifier.pth")
NUTRI_PREPROCESSOR_PATH = os.path.join(ASSET_DIR, "nutri_score_preprocessor.pkl")
NUTRI_FEATURES_PATH = os.path.join(ASSET_DIR, "nutri_score_features.pkl")
NUTRI_MODEL_OPTIMIZED_PATH = os.path.join(ASSET_DIR, "nutri_score_model_optimized.pkl")

def download_file(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {os.path.basename(destination)} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for HTTP errors
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{os.path.basename(destination)} downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"CRITICAL: Error downloading {url}: {e}")
            print(f"Application might not work correctly without {os.path.basename(destination)}")
            raise # Re-raise the exception to halt startup if a critical file fails
    else:
        print(f"{os.path.basename(destination)} already exists at {destination}.")

app = Flask(__name__, static_folder='static')

# --- Download all assets at startup ---
print("Checking and downloading model assets...")
try:
    download_file(DISTILBERT_MODEL_URL, DISTILBERT_MODEL_PATH)
    download_file(NUTRI_PREPROCESSOR_URL, NUTRI_PREPROCESSOR_PATH)
    download_file(NUTRI_FEATURES_URL, NUTRI_FEATURES_PATH)
    download_file(NUTRI_MODEL_OPTIMIZED_URL, NUTRI_MODEL_OPTIMIZED_PATH)
    print("Asset download check complete.")
except Exception as e:
    print(f"Failed to download one or more critical assets. Exiting. Error: {e}")
    exit(1) # Exit if downloads fail, as the app can't function

# Load Nutri-Score Model (using paths to downloaded files)
preprocessor = joblib.load(NUTRI_PREPROCESSOR_PATH)
feature_selector = joblib.load(NUTRI_FEATURES_PATH)
model_nutri = joblib.load(NUTRI_MODEL_OPTIMIZED_PATH)
print("Nutri-Score models loaded.")

# Load Food Ingredient Classifier Model (using path to downloaded file)
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model_nova = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4) # num_labels=4 as per your code
model_nova.load_state_dict(torch.load(DISTILBERT_MODEL_PATH, map_location=torch.device("cpu")))
model_nova.eval()
print("Food Ingredient Classifier (NOVA) model loaded.")

# -------- ROUTES -------- #
# (Your routes remain the same)
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

# ---- Nutri-Score Prediction ---- #
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

@app.route('/predict_nutri', methods=['POST'])
def predict_nutri():
    try:
        features_form = ['energy-kcal_100g', 'proteins_100g', 'fat_100g', 'saturated-fat_100g',
                         'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'salt_100g', 'sodium_100g']
        
        # Ensure all features are present in the request form, provide default 0 if not.
        # This makes it robust to missing form fields, though ideally, frontend ensures all are sent.
        input_data = []
        for f_name in features_form:
            try:
                value = float(request.form.get(f_name, 0)) # Default to 0 if missing
            except (ValueError, TypeError):
                value = 0.0 # Default to 0.0 if conversion fails
            input_data.append(value)

        X_input = np.array([input_data])
        X_df = pd.DataFrame(X_input, columns=features_form)
        
        X_processed = preprocessor.transform(X_df)
        X_selected = feature_selector.transform(X_processed) # Assumes feature_selector is a transformer
        
        score_prediction = model_nutri.predict(X_selected)
        score_value = float(score_prediction[0])
        
        grade, color = get_nutri_grade(score_value)

        return jsonify({'nutri_score': round(score_value, 2), 'grade': grade, 'color': color})
    except Exception as e:
        app.logger.error(f"Error in /predict_nutri: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# ---- Food Ingredient Classification ---- #
def classify_ingredients(ingredients_text):
    inputs = tokenizer(ingredients_text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(model_nova.device) # Ensure tensors are on same device as model
    with torch.no_grad():
        outputs = model_nova(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction + 1 # Assuming your classes are 1-indexed

@app.route('/predict_nova', methods=['POST'])
def predict_nova():
    try:
        ingredients = request.form.get('ingredients')
        if not ingredients:
            return jsonify({"error": "No ingredients provided"}), 400
        prediction = classify_ingredients(ingredients)
        return jsonify({"prediction": prediction})
    except Exception as e:
        app.logger.error(f"Error in /predict_nova: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        img = Image.open(image_file)
        extracted_text = pytesseract.image_to_string(img).strip()

        if not extracted_text:
            return jsonify({"extracted_text": "", "error": "No text detected"}), 200 # Return 200 but indicate no text

        prediction = classify_ingredients(extracted_text)
        return jsonify({"extracted_text": extracted_text, "prediction": prediction})
    except Exception as e:
        app.logger.error(f"Error in /ocr: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local development. Gunicorn will be used in production.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
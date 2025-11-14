"""
Vercel Serverless Function for Fast Delivery Review Classification
TF-IDF Model Only (lightweight for serverless deployment)
"""
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Import only if available (for local testing)
try:
    import joblib
    JOBLIB_AVAILABLE = True
except:
    JOBLIB_AVAILABLE = False

app = Flask(__name__, template_folder='../templates')
CORS(app)

# Paths - adjust for Vercel serverless environment
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "fast_delivery_ml_project" / "models"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# Globals
LOGREG_MODEL = None
TFIDF_VECTORIZER = None
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def load_tfidf():
    """Load TF-IDF model if available"""
    global LOGREG_MODEL, TFIDF_VECTORIZER
    if not JOBLIB_AVAILABLE:
        return False
    
    if TFIDF_MODEL_PATH.exists() and TFIDF_VECTORIZER_PATH.exists():
        try:
            LOGREG_MODEL = joblib.load(TFIDF_MODEL_PATH)
            TFIDF_VECTORIZER = joblib.load(TFIDF_VECTORIZER_PATH)
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    return False

def predict_tfidf(review_text):
    """Predict using TF-IDF + Logistic Regression"""
    if LOGREG_MODEL is None or TFIDF_VECTORIZER is None:
        return {"error": "TF-IDF model not loaded. Please train the model first."}
    
    try:
        X = TFIDF_VECTORIZER.transform([review_text])
        prediction = LOGREG_MODEL.predict(X)[0]
        probabilities = LOGREG_MODEL.predict_proba(X)[0]
        
        label = LABEL_MAP.get(prediction, "unknown")
        confidence = float(probabilities[prediction])
        
        return {
            "label": label,
            "confidence": confidence,
            "model": "TF-IDF + Logistic Regression",
            "all_scores": {
                "negative": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "positive": float(probabilities[2])
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.route("/")
def home():
    """Serve the frontend"""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict sentiment from review text"""
    try:
        data = request.get_json()
        if not data or "review" not in data:
            return jsonify({"error": "No review text provided"}), 400
        
        review = data.get("review", "").strip()
        if not review:
            return jsonify({"error": "Review text is empty"}), 400
        
        result = predict_tfidf(review)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models": {
            "tfidf": LOGREG_MODEL is not None
        }
    })

# Try to load models on startup
load_tfidf()

# Vercel handler
app.debug = False

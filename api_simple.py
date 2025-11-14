"""
Simple Flask API for Fast Delivery Review Classification
TF-IDF Model Only (Python 3.13 compatible)
"""
import logging
import json
import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("fast-delivery-api")

# Paths
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "fast_delivery_ml_project" / "models"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# Globals
LOGREG_MODEL = None
TFIDF_VECTORIZER = None

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def load_tfidf():
    global LOGREG_MODEL, TFIDF_VECTORIZER
    if TFIDF_MODEL_PATH.exists() and TFIDF_VECTORIZER_PATH.exists():
        LOGREG_MODEL = joblib.load(TFIDF_MODEL_PATH)
        TFIDF_VECTORIZER = joblib.load(TFIDF_VECTORIZER_PATH)
        logger.info("✅ TF-IDF model loaded successfully")
        return True
    logger.warning("⚠️  TF-IDF files missing")
    return False

def predict_tfidf(review_text):
    """Predict using TF-IDF + Logistic Regression"""
    if LOGREG_MODEL is None or TFIDF_VECTORIZER is None:
        return {"error": "TF-IDF model not loaded"}
    
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
        logger.error(f"TF-IDF prediction error: {e}")
        return {"error": str(e)}

@app.route("/")
def home():
    """Serve the frontend"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict sentiment from review text"""
    try:
        data = request.get_json()
        if not data or "review" not in data:
            return jsonify({"error": "No review text provided"}), 400
        
        review = data.get("review", "").strip()
        if not review:
            return jsonify({"error": "Review text is empty"}), 400
        
        model_choice = data.get("model", "tfidf").lower()
        
        if model_choice == "tfidf":
            result = predict_tfidf(review)
        else:
            result = {"error": f"Model '{model_choice}' not available. Only TF-IDF is supported."}
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models": {
            "tfidf": LOGREG_MODEL is not None,
            "distilbert": False
        }
    })

if __name__ == "__main__":
    logger.info("🚀 Starting Fast Delivery Review Classification API...")
    logger.info(f"📁 Models directory: {MODELS_DIR}")
    
    # Load models
    tfidf_loaded = load_tfidf()
    
    if not tfidf_loaded:
        logger.error("❌ No models loaded! Please train the model first.")
        logger.info("Run: python -m fast_delivery_ml_project.src.train_tfidf")
        exit(1)
    
    logger.info("✅ API ready!")
    logger.info("🌐 Open http://127.0.0.1:5000 in your browser")
    
    # Get port from environment variable for deployment platforms
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

import logging, json, re
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

# Optional transformers/torch
try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
try:
    import torch
except Exception:
    torch = None

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("fast-delivery-api")

# Paths
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "fast_delivery_ml_project" / "models"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
DISTILBERT_3CLASS_DIR = MODELS_DIR / "distilbert_3class"

# Globals
LOGREG_MODEL = None
TFIDF_VECTORIZER = None
BERT_PIPELINE = None

def load_tfidf():
    global LOGREG_MODEL, TFIDF_VECTORIZER
    if TFIDF_MODEL_PATH.exists() and TFIDF_VECTORIZER_PATH.exists():
        LOGREG_MODEL = joblib.load(TFIDF_MODEL_PATH)
        TFIDF_VECTORIZER = joblib.load(TFIDF_VECTORIZER_PATH)
        logger.info("TF-IDF loaded")
        return True
    logger.warning("TF-IDF files missing")
    return False

def load_distilbert():
    global BERT_PIPELINE
    if not HF_AVAILABLE:
        logger.warning("transformers not installed")
        return False
    
    local_cfg = DISTILBERT_3CLASS_DIR / "config.json"
    local_bin = DISTILBERT_3CLASS_DIR / "pytorch_model.bin"
    local_safetensors = DISTILBERT_3CLASS_DIR / "model.safetensors"
    
    if not local_cfg.exists():
        logger.error(f"config.json missing in {DISTILBERT_3CLASS_DIR}")
        return False
    
    if not (local_bin.exists() or local_safetensors.exists()):
        logger.error(f"DistilBERT model weights missing!")
        logger.error(f"  pytorch_model.bin: {local_bin.exists()}")
        logger.error(f"  model.safetensors: {local_safetensors.exists()}")
        logger.error(f"Run: python -m fast_delivery_ml_project.src.train_distilbert_3class")
        return False
    
    model_path = str(DISTILBERT_3CLASS_DIR)
    device = 0 if (torch and torch.cuda.is_available()) else -1
    
    try:
        BERT_PIPELINE = hf_pipeline(
            "text-classification", 
            model=model_path, 
            tokenizer=model_path, 
            device=device,
            return_all_scores=False
        )
        
        cfg = getattr(getattr(BERT_PIPELINE, "model", None), "config", None)
        num_labels = getattr(cfg, "num_labels", None)
        id2label = getattr(cfg, "id2label", None)
        
        logger.info(f"✅ DistilBERT 3-class loaded successfully!")
        logger.info(f"   Model path: {model_path}")
        logger.info(f"   num_labels: {num_labels}")
        logger.info(f"   id2label: {id2label}")
        logger.info(f"   Format: {'pytorch_model.bin' if local_bin.exists() else 'model.safetensors'}")
        
        if num_labels != 3:
            logger.error(f"❌ Model has {num_labels} labels, expected 3!")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load DistilBERT: {e}")
        return False

def predict_distilbert_3class(text: str):
    if BERT_PIPELINE is None:
        raise ValueError("DistilBERT pipeline is not loaded")
    
    out = BERT_PIPELINE(text, truncation=True, max_length=256)
    item = out[0] if isinstance(out, list) else out
    
    raw = str(item.get("label", "unknown"))
    score = float(item.get("score", 0.0))
    
    # Direct label mapping for 3-class model
    label_map = {
        "LABEL_0": "negative",
        "Negative": "negative",
        "LABEL_1": "neutral",
        "Neutral": "neutral",
        "LABEL_2": "positive",
        "Positive": "positive"
    }
    
    label = label_map.get(raw, "unknown")
    
    if label == "unknown":
        logger.warning(f"Unknown label from model: {raw}")
    
    return {
        "label": label,
        "confidence": score,
        "model": "DistilBERT-3class",
        "raw_label": raw
    }

def normalize_label_from_pipeline(raw_label: str, pipe) -> str:
    l = str(raw_label or "").lower().strip()
    if "neutral" in l: return "neutral"
    if "positive" in l: return "positive"
    if "negative" in l: return "negative"
    m = re.search(r"label[_\- ]?(\d+)", l)
    idx = int(m.group(1)) if m else (int(l) if l.isdigit() else None)
    cfg = getattr(getattr(pipe, "model", None), "config", None)
    num = getattr(cfg, "num_labels", None)
    if idx is not None:
        if num == 3: return ["negative","neutral","positive"][min(idx,2)]
        if num == 2: return ["negative","positive"][min(idx,1)]
    return "unknown"

LABEL_NAMES = ["negative", "neutral", "positive"]

def predict_tfidf(text: str):
    if TFIDF_VECTORIZER is None or LOGREG_MODEL is None:
        raise ValueError("TF-IDF model or vectorizer is not loaded")
    X = TFIDF_VECTORIZER.transform([text])
    pred = int(LOGREG_MODEL.predict(X)[0])
    proba = LOGREG_MODEL.predict_proba(X)[0] if hasattr(LOGREG_MODEL,"predict_proba") else [0.33,0.33,0.34]
    
    # Map 0=negative, 1=neutral, 2=positive
    label = LABEL_NAMES[pred] if pred < len(LABEL_NAMES) else "unknown"
    confidence = float(proba[pred])
    
    return {"label": label, "confidence": confidence, "model": "TF-IDF"}

def predict_distilbert(text: str):
    if BERT_PIPELINE is None:
        raise ValueError("DistilBERT pipeline is not loaded")
    out = BERT_PIPELINE(text, truncation=True, max_length=256)
    item = out[0] if isinstance(out, list) else out
    raw = str(item.get("label","unknown")); score = float(item.get("score",0.0))
    label = normalize_label_from_pipeline(raw, BERT_PIPELINE)
    return {"label": label, "confidence": score, "model": "DistilBERT", "raw_label": raw}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    text = str(data.get("text","")).strip()
    model = str(data.get("model","distilbert")).lower()
    if not text:
        return jsonify({"error":"No text","label":"unknown","confidence":0.0}), 400
    try:
        if model=="tfidf" and LOGREG_MODEL and TFIDF_VECTORIZER:
            return jsonify(predict_tfidf(text))
        if BERT_PIPELINE:
            return jsonify(predict_distilbert(text))
        if LOGREG_MODEL and TFIDF_VECTORIZER:
            return jsonify(predict_tfidf(text))
        return jsonify({"label":"neutral","confidence":0.5,"model":"Keyword"})
    except Exception as e:
        logger.exception("Predict error")
        return jsonify({"error":str(e),"label":"unknown","confidence":0.0}), 500

@app.route("/api/test")
def api_test():
    cfg = getattr(getattr(BERT_PIPELINE,"model",None),"config",None) if BERT_PIPELINE else None
    return jsonify({
        "tfidf_available": bool(LOGREG_MODEL and TFIDF_VECTORIZER),
        "distilbert_available": bool(BERT_PIPELINE),
        "bert_num_labels": getattr(cfg,"num_labels",None) if cfg else None,
        "bert_id2label": getattr(cfg,"id2label",None) if cfg else None,
        "models_dir": str(MODELS_DIR),
        "paths_exist": {
            "tfidf_model": TFIDF_MODEL_PATH.exists(),
            "tfidf_vectorizer": TFIDF_VECTORIZER_PATH.exists(),
            "distilbert_3class_config": (DISTILBERT_3CLASS_DIR / "config.json").exists(),
            "distilbert_3class_bin": (DISTILBERT_3CLASS_DIR / "pytorch_model.bin").exists(),
            "distilbert_3class_safetensors": (DISTILBERT_3CLASS_DIR / "model.safetensors").exists(),
            "distilbert_3class_tokenizer": (DISTILBERT_3CLASS_DIR / "tokenizer.json").exists()
        }
    })

if __name__ == "__main__":
    logger.info("Starting API...")
    load_tfidf()
    load_distilbert()
    app.run(host="127.0.0.1", port=5000, debug=True)
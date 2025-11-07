import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "fast_delivery_ml_project" / "data" / "raw" / "delivery_reviews_enhanced.csv"
MODELS_DIR = ROOT / "fast_delivery_ml_project" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# 3-CLASS MAPPING for Customer Feedback Type
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

def load_data():
    """Load dataset using Customer Feedback Type as 3-class label"""
    logger.info(f"Loading data from {DATA_FILE}")
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    logger.info(f"Loaded {len(df)} reviews")
    
    if "Review Text" not in df.columns or "Customer Feedback Type" not in df.columns:
        raise ValueError("Missing required columns: 'Review Text' or 'Customer Feedback Type'")
    
    df = df[["Review Text", "Customer Feedback Type"]].copy()
    df.columns = ["text", "label"]
    
    df = df.dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()
    
    # Map to 3-class numeric labels
    df = df[df["label"].isin(["Negative", "Neutral", "Positive"])]
    df["label"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    
    logger.info(f"After cleaning: {len(df)} reviews")
    logger.info(f"\nLabel distribution:")
    for label_id, count in df["label"].value_counts().sort_index().items():
        label_idx = int(label_id) if isinstance(label_id, (int, float, np.integer)) else 0
        logger.info(f"  {LABEL_NAMES[label_idx]}: {count}")
    
    return df

def train_tfidf_model(df, test_size=0.2):
    """Train TF-IDF + Logistic Regression with adjusted features"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], 
        df["label"], 
        test_size=test_size, 
        random_state=42, 
        stratify=df["label"],
        shuffle=True  # Ensure proper shuffling
    )
    
    logger.info(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Adjusted TF-IDF vectorizer to prevent overfitting
    logger.info("\nCreating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Reduced to avoid overfitting
        ngram_range=(1, 2),  # Reduced to bigrams for generalization
        min_df=2,
        max_df=0.9,
        lowercase=True,
        strip_accents='unicode',
        sublinear_tf=True,
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    logger.info(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Hyperparameter tuning
    logger.info("\nTraining Logistic Regression with hyperparameter tuning...")
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'max_iter': [2000],
        'multi_class': ['multinomial'],
        'solver': ['lbfgs'],
        'penalty': ['l2'],
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(class_weight='balanced', random_state=42),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=1,  # Sequential to avoid Windows issues
        verbose=1
    )
    
    try:
        grid_search.fit(X_train_tfidf, y_train)
    except Exception as e:
        logger.error(f"GridSearchCV failed: {e}")
        logger.info("Falling back to default LogisticRegression")
        model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_tfidf, y_train)
    else:
        model = grid_search.best_estimator_
        logger.info(f"\nBest parameters: {grid_search.best_params_}")
        logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("EVALUATION")
    logger.info("="*60)
    
    y_train_pred = model.predict(X_train_tfidf)
    train_acc = accuracy_score(y_train, y_train_pred)
    logger.info(f"\nTrain Accuracy: {train_acc:.4f}")
    
    y_test_pred = model.predict(X_test_tfidf)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    logger.info(f"\nTest Accuracy:  {test_acc:.4f}")
    logger.info(f"Test F1 Score:  {test_f1:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall:    {test_recall:.4f}")
    
    logger.info("\nClassification Report:")
    report = classification_report(y_test, y_test_pred, target_names=LABEL_NAMES, zero_division=0, output_dict=False)
    logger.info(f"\n{report}")
    
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                Predicted")
    logger.info(f"           Neg  Neu  Pos")
    for i, row in enumerate(cm):
        logger.info(f"Actual {LABEL_NAMES[i]:8s} {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")
    
    # Save
    logger.info(f"\nSaving model to {TFIDF_MODEL_PATH}")
    joblib.dump(model, TFIDF_MODEL_PATH)
    
    logger.info(f"Saving vectorizer to {TFIDF_VECTORIZER_PATH}")
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    
    # Save metrics
    metrics = {
        "tfidf": {
            "accuracy": float(test_acc),
            "f1": float(test_f1),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "num_classes": 3
        }
    }
    
    if METRICS_PATH.exists():
        try:
            with METRICS_PATH.open("r") as f:
                existing = json.load(f)
            existing.update(metrics)
            metrics = existing
        except Exception:
            pass
    
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\nMetrics saved to: {METRICS_PATH}")
    
    return model, vectorizer, metrics["tfidf"]

def main():
    logger.info("="*60)
    logger.info("TF-IDF + LOGISTIC REGRESSION TRAINING (3-CLASS)")
    logger.info("="*60)
    
    df = load_data()
    model, vectorizer, metrics = train_tfidf_model(df, test_size=0.2)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*60)
    logger.info(f"Model saved to: {TFIDF_MODEL_PATH}")
    logger.info(f"Vectorizer saved to: {TFIDF_VECTORIZER_PATH}")
    logger.info(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final Test F1 Score: {metrics['f1']:.4f}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
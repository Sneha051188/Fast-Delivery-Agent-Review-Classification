"""
TF-IDF + Logistic Regression Training Script for 3-Class Sentiment Classification
Trains on enhanced delivery reviews dataset (2000 balanced samples)
"""
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import json

# Configuration
DATA_FILE = Path(__file__).parent.parent / "data" / "raw" / "delivery_reviews_enhanced.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
LABEL_MAP = {"Incorrect": 0, "Neutral": 1, "Correct": 2}
REVERSE_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def train_tfidf_model():
    """Train TF-IDF + Logistic Regression model"""
    print("🔄 Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Map labels to numeric
    df['label'] = df['sentiment'].map(LABEL_MAP)
    
    print(f"📊 Dataset size: {len(df)} samples")
    print(f"📈 Distribution:\n{df['sentiment'].value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print("\n🔧 Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("🔧 Training Logistic Regression with GridSearchCV...")
    print("⚠️  Using n_jobs=1 for Windows compatibility...")
    
    # GridSearchCV with n_jobs=1 for Windows
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [200, 500]
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42),
        param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=1,  # Windows compatibility
        verbose=1
    )
    
    grid_search.fit(X_train_tfidf, y_train)
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    
    # Get best model
    model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])}")
    
    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(MODEL_DIR / "tfidf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics = {
        "tfidf": {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "best_params": grid_search.best_params_
        }
    }
    
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Models saved to {MODEL_DIR}")
    print(f"   - tfidf_vectorizer.pkl")
    print(f"   - tfidf_model.pkl")
    print(f"   - metrics.json")

if __name__ == "__main__":
    train_tfidf_model()

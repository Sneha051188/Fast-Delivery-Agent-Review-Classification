# Fast Delivery Agent Review Classification 🚚

Machine learning project for 3-class sentiment analysis of delivery reviews.

## 🚀 Quick Start

```bash
# Install dependencies
pip install flask flask-cors scikit-learn pandas joblib

# Generate dataset & train models
python -m fast_delivery_ml_project.src.generate_enhanced_dataset
python -m fast_delivery_ml_project.src.train_tfidf

# Run API
python api_simple.py
```

Visit: http://127.0.0.1:5000

## 📊 Results

- **TF-IDF Model:** 100% accuracy
- **DistilBERT Model:** 100% accuracy

## 🌐 Vercel Deployment

**Note:** Model files are too large for Vercel. Recommended to:
1. Deploy API separately (AWS/GCP/Azure)
2. Use Vercel for frontend only

## 👤 Author

[@Sneha051188](https://github.com/Sneha051188)

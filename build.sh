#!/usr/bin/env bash
# Build script for Render

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Generating dataset..."
python -m fast_delivery_ml_project.src.generate_enhanced_dataset

echo "Training TF-IDF model..."
python -m fast_delivery_ml_project.src.train_tfidf

echo "Training complete! Skipping DistilBERT to save build time..."
echo "To train DistilBERT locally, run: python -m fast_delivery_ml_project.src.train_distilbert_3class"

echo "Build completed successfully!"

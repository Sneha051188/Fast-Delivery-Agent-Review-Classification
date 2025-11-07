# India's Fast Delivery Agents — Reviews & Ratings (ML Research Project)

This repository contains a **publishable ML pipeline** built for the Kaggle dataset:  
**"India's Fast Delivery Agents Reviews and Ratings"**.

Dataset link: https://www.kaggle.com/datasets/kanakbaghel/indias-fast-delivery-agents-reviews-and-ratings/data

## What this project does
- **Preprocess & clean** the reviews
- **EDA** and charts saved to `figures/`
- **Baselines (ML)**: Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost/LightGBM (classification & regression)
- **Transformer (DL)**: DistilBERT (sentiment & rating prediction)
- **Evaluation**: Accuracy/F1 (classification), MAE/RMSE (regression)
- **Reproducible** via `config.yaml` and CLI scripts

## Quickstart
1. **Create environment**  
   ```bash
   python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   ```

2. **Place data**  
   Put the Kaggle CSV inside: `data/raw/`  
   Update `config.yaml` with the exact **file name** and **column names**.

3. **Run EDA**  
   ```bash
   python src/eda.py --config config.yaml
   ```

4. **Prepare & split data**  
   ```bash
   python src/preprocess.py --config config.yaml
   ```

5. **Train baselines (sentiment & rating)**  
   ```bash
   # Sentiment (classification, derived from rating thresholds in config)
   python src/train_baselines.py --config config.yaml --task sentiment
   # Rating prediction (regression)
   python src/train_baselines.py --config config.yaml --task rating
   ```

6. **Train Transformer (optional, GPU recommended)**  
   ```bash
   # Sentiment classification
   python src/train_bert.py --config config.yaml --task sentiment
   # Rating regression
   python src/train_bert.py --config config.yaml --task rating
   ```

7. **Results & models**  
   Metrics JSON & confusion matrices: `experiments/results/`  
   Saved models/pipelines: `experiments/models/`

## Paper-friendly assets
- Include EDA figures from `figures/`
- Use the metrics saved as JSON for tables
- Cite the dataset and your chosen model papers

## Config tips
- If review column != `review_text`, set it in `config.yaml`
- If rating scale is not 1–5, adjust thresholds

---

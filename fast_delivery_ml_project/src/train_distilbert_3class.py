"""
DistilBERT Training Script for 3-Class Sentiment Classification
Fine-tunes DistilBERT on enhanced delivery reviews dataset
"""
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import numpy as np

# Configuration
DATA_FILE = Path(__file__).parent.parent / "data" / "raw" / "delivery_reviews_enhanced.csv"
MODEL_DIR = Path(__file__).parent.parent / "models" / "distilbert_3class"
LABEL_MAP = {"Incorrect": 0, "Neutral": 1, "Correct": 2}

class ReviewDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for reviews"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

class WeightedTrainer(Trainer):
    """Custom Trainer with class weights"""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_distilbert_model():
    """Train DistilBERT model"""
    print("🔄 Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Map labels
    df['label'] = df['sentiment'].map(LABEL_MAP)
    
    print(f"📊 Dataset size: {len(df)} samples")
    print(f"📈 Distribution:\n{df['sentiment'].value_counts()}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    print("\n🔧 Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print("🔧 Tokenizing data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)
    
    # Calculate class weights
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor(len(class_counts) / class_counts)
    
    print(f"\n⚖️  Class weights: {class_weights.tolist()}")
    
    print("\n🔧 Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    )
    
    # Training arguments with save_safetensors=False
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(MODEL_DIR / 'logs'),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        save_safetensors=False,  # Force pytorch_model.bin format
        report_to="none"
    )
    
    print("🔧 Starting training...")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights
    )
    
    trainer.train()
    
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    
    print(f"\n✅ Final Results:")
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   F1 Score: {eval_results['eval_f1']:.4f}")
    
    # Generate predictions for classification report
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    
    print(f"\n{classification_report(val_labels, preds, target_names=['negative', 'neutral', 'positive'])}")
    
    # Save model with explicit safe_serialization=False
    print(f"\n💾 Saving model to {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR, safe_serialization=False)
    tokenizer.save_pretrained(MODEL_DIR)
    
    # Verify pytorch_model.bin was created
    model_bin_path = MODEL_DIR / "pytorch_model.bin"
    if model_bin_path.exists():
        print(f"✅ Verified: pytorch_model.bin exists ({model_bin_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"⚠️  Warning: pytorch_model.bin not found!")
    
    # Save metrics
    metrics_path = MODEL_DIR.parent / "metrics.json"
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    
    metrics["distilbert"] = {
        "accuracy": float(eval_results['eval_accuracy']),
        "f1_score": float(eval_results['eval_f1'])
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {MODEL_DIR}")
    print(f"   Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    train_distilbert_model()

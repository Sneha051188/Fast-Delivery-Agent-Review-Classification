import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import json
import logging
import torch
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "fast_delivery_ml_project" / "data" / "raw" / "delivery_reviews_enhanced.csv"
MODELS_DIR = ROOT / "fast_delivery_ml_project" / "models"
DISTILBERT_DIR = MODELS_DIR / "distilbert_3class"
METRICS_PATH = MODELS_DIR / "metrics.json"

# LABEL MAPPING for Customer Feedback Type
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Simple data augmentation via synonym replacement
def augment_text(text):
    """Add slight variations to text for diversity"""
    synonyms = {
        "fast": ["quick", "swift", "prompt"],
        "great": ["excellent", "awesome", "fantastic"],
        "late": ["delayed", "tardy", "slow"],
        "poor": ["bad", "terrible", "subpar"],
        "okay": ["fine", "decent", "acceptable"],
        "service": ["support", "assistance", "delivery"],
        "driver": ["courier", "delivery person", "rider"],
        "items": ["products", "goods", "order"]
    }
    words = text.split()
    for i, word in enumerate(words):
        if random.random() < 0.2:  # 20% chance to replace
            for key, vals in synonyms.items():
                if word.lower() == key:
                    words[i] = random.choice(vals)
                    break
    return " ".join(words)

# DATA PIPELINE
def load_and_prepare_data():
    """Load dataset and apply augmentation"""
    df = pd.read_csv(DATA_FILE)
    logger.info(f"Loaded {len(df)} reviews")
    
    if "Review Text" not in df.columns or "Customer Feedback Type" not in df.columns:
        raise ValueError("Missing required columns: 'Review Text' or 'Customer Feedback Type'")
    
    df = df[["Review Text", "Customer Feedback Type"]].rename(columns={"Review Text": "text", "Customer Feedback Type": "label"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.capitalize()
    df = df[df["label"].isin(["Negative", "Neutral", "Positive"])]
    
    # Augment data by creating one varied copy per review
    augmented = df.copy()
    augmented["text"] = augmented["text"].apply(augment_text)
    df = pd.concat([df, augmented], ignore_index=True)
    df = df.drop_duplicates(subset=["text"])
    
    logger.info(f"After cleaning and augmentation: {len(df)} reviews")
    logger.info(f"Final label distribution:\n{df['label'].value_counts()}")
    return df

def create_datasets(df, test_size=0.15, val_size=0.15):
    df["label_id"] = df["label"].map(LABEL_MAP)
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label_id"], random_state=42, shuffle=True)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, stratify=train_val["label_id"], random_state=42, shuffle=True)
    train_ds = Dataset.from_pandas(train[["text","label_id"]].rename(columns={"label_id":"labels"}))
    val_ds = Dataset.from_pandas(val[["text","label_id"]].rename(columns={"label_id":"labels"}))
    test_ds = Dataset.from_pandas(test[["text","label_id"]].rename(columns={"label_id":"labels"}))
    return DatasetDict({"train":train_ds,"validation":val_ds,"test":test_ds})

def tokenize_datasets(datasets, model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tok_fn(examples): return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
    return datasets.map(tok_fn, batched=True, remove_columns=["text"]), tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(labels, preds)}")
    logger.info(classification_report(labels, preds, target_names=list(ID2LABEL.values())))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights, dtype=torch.float32).to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=w)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_model(datasets, tokenizer, output_dir=DISTILBERT_DIR, epochs=15):
    """Train DistilBERT with optimized settings for high accuracy"""
    logger.info("\n" + "="*60)
    logger.info("INITIALIZING DISTILBERT 3-CLASS MODEL")
    logger.info("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP
    )
    
    # Compute class weights
    train_labels = datasets["train"]["labels"]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    logger.info(f"\nClass weights: {dict(zip(ID2LABEL.values(), class_weights))}")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,  # Increased for better convergence
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,  # Lower for stability
        weight_decay=0.05,  # Increased to prevent overfitting
        warmup_ratio=0.2,  # Longer warmup
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        save_total_limit=2,
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        save_safetensors=False,
        gradient_accumulation_steps=8,  # Effective batch size = 8*8=64
        max_grad_norm=1.0,  # Gradient clipping
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        class_weights=class_weights.tolist()
    )
    
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    trainer.train()
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*60)
    
    test_results = trainer.evaluate(datasets["test"])
    
    logger.info(f"\nSaving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save again explicitly with PyTorch format
    model.save_pretrained(str(output_dir), safe_serialization=False)
    
    # Verify files
    config_file = output_dir / "config.json"
    model_bin_file = output_dir / "pytorch_model.bin"
    model_safetensors_file = output_dir / "model.safetensors"
    tokenizer_file = output_dir / "tokenizer.json"
    
    logger.info(f"\nVerifying saved files:")
    logger.info(f"  config.json exists: {config_file.exists()}")
    logger.info(f"  pytorch_model.bin exists: {model_bin_file.exists()}")
    logger.info(f"  model.safetensors exists: {model_safetensors_file.exists()}")
    logger.info(f"  tokenizer.json exists: {tokenizer_file.exists()}")
    
    if not (model_bin_file.exists() or model_safetensors_file.exists()):
        raise RuntimeError(f"No model weights saved! Check {output_dir}")
    
    # Save metrics
    metrics = {
        "distilbert_3class": {
            "accuracy": float(test_results.get("eval_accuracy", 0.0)),
            "f1": float(test_results.get("eval_f1", 0.0)),
            "precision": float(test_results.get("eval_precision", 0.0)),
            "recall": float(test_results.get("eval_recall", 0.0)),
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
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\nMetrics saved to: {METRICS_PATH}")
    
    return trainer, test_results

def main():
    logger.info("="*60)
    logger.info("DISTILBERT 3-CLASS SENTIMENT TRAINING")
    logger.info("="*60)
    df = load_and_prepare_data()
    ds = create_datasets(df)
    tok_ds, tokenizer = tokenize_datasets(ds)
    train_model(tok_ds, tokenizer, epochs=15)
    logger.info("Training complete ✅")

if __name__ == "__main__":
    main()
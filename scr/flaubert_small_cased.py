import os
import json
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# =========================
# Configuration
# =========================
CONFIG = {
    "model": {
        "model_name": "flaubert/flaubert_small_cased",
        "num_labels": 2,
        "max_length": 128
    },
    "training": {
        "batch_size": 16,
        "num_epochs": 5,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "grad_accum_steps": 2,
        "fp16": True,
        "gradient_checkpointing": False
    },
    "data": {
        "test_size": 0.2,
        "val_size": 0.1,
        "random_seed": 42
    }
}

# =========================
# Dataset Loader
# =========================
def load_data():
    ban_emo = pd.read_csv("/mnt/data/h0p3/hate/src/data/BanEmoHate.csv").rename(columns={'Comments': 'text', 'HateSpeech': 'label'})
    ban_emo['label'] = (ban_emo['label'] == 'hate').astype(int)

    banglish = pd.read_csv("/mnt/data/h0p3/hate/src/data/BanglishHateSpeechDataset.csv").rename(columns={'Comment': 'text', 'Hate': 'label'})
    banglish['label'] = (banglish['label'] == 'Yes').astype(int)

    bengali = pd.read_csv("/mnt/data/h0p3/hate/src/data/BengaliHeatspeechDataset.csv", encoding='utf-8')
    bengali.columns = bengali.columns.str.strip()
    bengali = bengali.rename(columns={'Text': 'text'})
    hate_cols = ['Race', 'Behaviour', 'Physical', 'Class', 'Religion', 'Disability', 'Ethnicity', 'Gender', 'Sexual Orientation', 'Political']
    bengali['label'] = (bengali[hate_cols].sum(axis=1) > 0).astype(int)

    combined = concatenate_datasets([Dataset.from_pandas(df[['text', 'label']]) for df in [ban_emo, banglish, bengali]])
    train_test = combined.train_test_split(test_size=CONFIG['data']['test_size'], seed=CONFIG['data']['random_seed'])
    train_val = train_test['train'].train_test_split(test_size=CONFIG['data']['val_size']/(1-CONFIG['data']['test_size']), seed=CONFIG['data']['random_seed'])
    return {'train': train_val['train'], 'validation': train_val['test'], 'test': train_test['test']}

# =========================
# Metrics
# =========================
def compute_metrics(pred):
    labels, preds = pred.label_ids, pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# =========================
# Save metrics to JSON
# =========================
def save_metrics(model_name, metrics, config, output_dir="./metrics"):
    os.makedirs(output_dir, exist_ok=True)
    batch = config['training']['batch_size']
    lr = config['training']['learning_rate']
    epochs = config['training']['num_epochs']
    filename = f"{model_name}_batch{batch}_lr{lr}_ep{epochs}.json"
    output_path = os.path.join(output_dir, filename)
    result = {"model_name": model_name, "metrics": metrics, "config": config}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"✅ Metrics saved to {output_path}")

# =========================
# Main Training Loop
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = FlaubertTokenizer.from_pretrained(CONFIG['model']['model_name'])
    model = FlaubertForSequenceClassification.from_pretrained(CONFIG['model']['model_name'], num_labels=CONFIG['model']['num_labels']).to(device)

    # Load and tokenize data
    datasets = load_data()
    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding=False, max_length=CONFIG['model']['max_length'])
    tokenized = {k: v.map(tokenize_fn, batched=True) for k, v in datasets.items()}

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./results_flaubert",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=CONFIG['training']['learning_rate'],
            per_device_train_batch_size=CONFIG['training']['batch_size'],
            per_device_eval_batch_size=CONFIG['training']['batch_size']*2,
            num_train_epochs=CONFIG['training']['num_epochs'],
            weight_decay=CONFIG['training']['weight_decay'],
            warmup_steps=CONFIG['training']['warmup_steps'],
            gradient_accumulation_steps=CONFIG['training']['grad_accum_steps'],
            fp16=CONFIG['training']['fp16'],
            logging_dir="./logs_flaubert",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
            report_to="none"
        ),
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    print("Training model...")
    trainer.train()

    print("Evaluating on test set...")
    results = trainer.evaluate(tokenized['test'])
    filtered_results = {k: results[k] for k in ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']}
    print("\nTest Results:", filtered_results)

    model.save_pretrained("./saved_model_flaubert")
    tokenizer.save_pretrained("./saved_model_flaubert")
    print("Model saved to ./saved_model_flaubert")

    save_metrics("Flaubert", filtered_results, CONFIG)

if __name__ == "__main__":
    main()

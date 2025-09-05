import os
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# =========================
# Configuration
# =========================
CONFIG = {
    "model": {
        "model_name": "distilbert-base-multilingual-cased",
        "num_labels": 2,
        "max_length": 128
    },
    "training": {
        "batch_size": 8,
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "grad_accum_steps": 2,
        "fp16": True,
        "gradient_checkpointing": True
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
    # Load BanEmo dataset
    ban_emo = pd.read_csv("/mnt/data/h0p3/hate/src/data/BanEmoHate.csv").rename(
        columns={'Comments': 'text', 'HateSpeech': 'label'}
    )
    ban_emo['label'] = (ban_emo['label'] == 'hate').astype(int)

    # Load Banglish dataset
    banglish = pd.read_csv("/mnt/data/h0p3/hate/src/data/BanglishHateSpeechDataset.csv").rename(
        columns={'Comment': 'text', 'Hate': 'label'}
    )
    banglish['label'] = (banglish['label'] == 'Yes').astype(int)

    # Load Bengali dataset
    bengali = pd.read_csv("/mnt/data/h0p3/hate/src/data/BengaliHeatspeechDataset.csv", encoding='utf-8')
    bengali.columns = bengali.columns.str.strip()
    bengali = bengali.rename(columns={'Text': 'text'})
    hate_cols = ['Race', 'Behaviour', 'Physical', 'Class', 'Religion', 'Disability',
                 'Ethnicity', 'Gender', 'Sexual Orientation', 'Political']
    bengali['label'] = (bengali[hate_cols].sum(axis=1) > 0).astype(int)

    # Combine datasets
    combined = concatenate_datasets([
        Dataset.from_pandas(df[['text', 'label']]) for df in [ban_emo, banglish, bengali]
    ])

    # Split train/val/test
    train_test = combined.train_test_split(
        test_size=CONFIG['data']['test_size'],
        seed=CONFIG['data']['random_seed']
    )
    train_val = train_test['train'].train_test_split(
        test_size=CONFIG['data']['val_size'] / (1 - CONFIG['data']['test_size']),
        seed=CONFIG['data']['random_seed']
    )
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
# Main Training Loop
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(CONFIG['model']['model_name'])
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG['model']['model_name'], num_labels=CONFIG['model']['num_labels']
    ).to(device)

    # Load and tokenize dataset
    print("Loading and processing data...")
    datasets = load_data()

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding='longest', max_length=CONFIG['model']['max_length'])

    tokenized = {k: v.map(tokenize_fn, batched=True, remove_columns=['text']) for k, v in datasets.items()}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results_distilbert",
        num_train_epochs=CONFIG['training']['num_epochs'],
        per_device_train_batch_size=CONFIG['training']['batch_size'],
        per_device_eval_batch_size=CONFIG['training']['batch_size'] * 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs_distilbert",
        logging_steps=50,
        **{k: v for k, v in CONFIG['training'].items() if k in ['learning_rate', 'weight_decay', 'warmup_steps', 'fp16', 'gradient_checkpointing']}
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Train
    print("Training model...")
    trainer.train()

    # Evaluate
    print("Evaluating on test set...")
    results = trainer.evaluate(tokenized['test'])
    filtered_results = {k: v for k, v in results.items() if k not in ['eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']}
    print("\nTest Results:", filtered_results)

    # Save
    model.save_pretrained("./saved_model_distilbert")
    tokenizer.save_pretrained("./saved_model_distilbert")
    print("Model saved to ./saved_model_distilbert")

if __name__ == "__main__":
    main()

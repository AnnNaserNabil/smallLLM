import os
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import MobileBertForSequenceClassification, MobileBertTokenizerFast, Trainer, TrainingArguments, DataCollatorWithPadding

# Configuration
CONFIG = {
    "model": {
        "model_name": "google/mobilebert-uncased",  # MobileBERT (English uncased)
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

def load_data():
    # Load and combine datasets
    ban_emo = pd.read_csv("/mnt/data/h0p3/hate/src/data/BanEmoHate.csv").rename(columns={'Comments': 'text', 'HateSpeech': 'label'})
    ban_emo['label'] = (ban_emo['label'] == 'hate').astype(int)
    
    banglish = pd.read_csv("/mnt/data/h0p3/hate/src/data/BanglishHateSpeechDataset.csv").rename(columns={'Comment': 'text', 'Hate': 'label'})
    banglish['label'] = (banglish['label'] == 'Yes').astype(int)
    
    bengali = pd.read_csv("/mnt/data/h0p3/hate/src/data/BengaliHeatspeechDataset.csv", encoding='utf-8').rename(columns={'Text': 'text'})
    hate_cols = ['Race', 'Behaviour', 'Physical', 'Class', 'Religion', 'Disability', 'Ethnicity ', 'Gender', 'Sexual Orientation', 'Political']
    bengali['label'] = (bengali[hate_cols].sum(axis=1) > 0).astype(int)
    
    # Combine and split
    combined = concatenate_datasets([Dataset.from_pandas(df[['text', 'label']]) for df in [ban_emo, banglish, bengali]])
    train_test = combined.train_test_split(test_size=CONFIG['data']['test_size'], seed=CONFIG['data']['random_seed'])
    train_val = train_test['train'].train_test_split(test_size=CONFIG['data']['val_size']/(1-CONFIG['data']['test_size']))
    return {'train': train_val['train'], 'validation': train_val['test'], 'test': train_test['test']}

def compute_metrics(pred):
    labels, preds = pred.label_ids, pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': accuracy_score(labels, preds), 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MobileBertTokenizerFast.from_pretrained(CONFIG['model']['model_name'])
    model = MobileBertForSequenceClassification.from_pretrained(
        CONFIG['model']['model_name'], num_labels=CONFIG['model']['num_labels']
    ).to(device)
    
    # Prepare data
    print("Loading and processing data...")
    datasets = load_data()
    
    def tokenize(batch):
        return tokenizer(batch['text'], padding='longest', truncation=True, max_length=CONFIG['model']['max_length'])
    
    tokenized_datasets = {k: v.map(tokenize, batched=True, remove_columns=['text']) for k, v in datasets.items()}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    
    # Train
    trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='./results',
        num_train_epochs=CONFIG['training']['num_epochs'],
        per_device_train_batch_size=CONFIG['training']['batch_size'],
        per_device_eval_batch_size=CONFIG['training']['batch_size']*2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay'],
        warmup_steps=CONFIG['training']['warmup_steps'],
        fp16=CONFIG['training']['fp16']
        # gradient_checkpointing removed
    ),
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    data_collator=data_collator
    )
    trainer.train()
    
    # Evaluate
    print("Evaluating on test set...")
    results = trainer.evaluate(tokenized_datasets['test'])
    print("\nTest Results:", {k: v for k, v in results.items() if k not in ['eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']})
    
    # Save
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')
    print("Model saved to ./saved_model")

if __name__ == "__main__":
    main()

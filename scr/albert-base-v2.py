import os
import json
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AlbertForSequenceClassification, AlbertTokenizerFast, Trainer, TrainingArguments, DataCollatorWithPadding

# =========================
# Configuration
# =========================
CONFIG = {
    "model": {
        "model_name": "albert-base-v2",
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
    # BanEmo
    ban_emo = pd.read_csv("/mnt/data/h0p3/hate/src/data/BanEmoHate.csv")
    ban_emo = ban_emo.rename(columns={'Comments': 'text', 'HateSpeech': 'label'})
    ban_emo['label'] = (ban_emo['label'] == 'hate').astype(int)

    # Banglish
    banglish = pd.read_csv("/mnt/data/h0p3/hate/src/data/BanglishHateSpeechDataset.csv")
    banglish = banglish.rename(columns={'Comment': 'text', 'Hate': 'label'})
    banglish['label'] = (banglish['label'] == 'Yes').astype(int)

    # Bengali
    bengali = pd.read_csv("/mnt/data/h0p3/hate/src/data/BengaliHeatspeechDataset.csv", encoding='utf-8')
    bengali = bengali.rename(columns={'Text': 'text'})
    bengali.columns = bengali.columns.str.strip()
    hate_cols = ['Race', 'Behaviour', 'Physical', 'Class', 'Religion', 'Disability',
                 'Ethnicity', 'Gender', 'Sexual Orientation', 'Political']
    bengali['label'] = (bengali[hate_cols].sum(axis=1) > 0).astype(int)

    # Combine datasets
    combined = concatenate_datasets([
        Dataset.from_pandas(df[['text', 'label']]) for df in [ban_emo, banglish, bengali]
    ])

    # Split
    train_test = combined.train_test_split(test_size=CONFIG['data']['test_size'], seed=CONFIG['data']['random_seed'])
    train_val = train_test['train'].train_test_split(
        test_size=CONFIG['data']['val_size'] / (1 - CONFIG['data']['test_size']),
        seed=CONFIG['data']['random_seed']
    )

    return {'train': train_val['train'], 'validation': train_val['test'], 'test': train_test['test']}

# =========================
# Metrics
# =========================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# =========================
# Save metrics JSON
# =========================
def save_metrics(model_name, metrics, config, output_dir="./metrics"):
    os.makedirs(output_dir, exist_ok=True)
    batch = config['training']['batch_size']
    lr = config['training']['learning_rate']
    epochs = config['training']['num_epochs']
    filename = f"{model_name}_batch{batch}_lr{lr}_ep{epochs}.json"
    output_path = os.path.join(output_dir, filename)
    result = {
        "model_name": model_name,
        "metrics": metrics,
        "config": config
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"✅ Metrics saved to {output_path}")

# =========================
# Main
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer and Model
    tokenizer = AlbertTokenizerFast.from_pretrained(CONFIG['model']['model_name'])
    model = AlbertForSequenceClassification.from_pretrained(
        CONFIG['model']['model_name'],
        num_labels=CONFIG['model']['num_labels']
    ).to(device)

    # Dataset
    print("Loading dataset...")
    datasets = load_data()

    def tokenize(batch):
        return tokenizer(batch['text'], padding='longest', truncation=True, max_length=CONFIG['model']['max_length'])

    tokenized_datasets = {k: v.map(tokenize, batched=True, remove_columns=['text']) for k, v in datasets.items()}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=CONFIG['training']['num_epochs'],
        per_device_train_batch_size=CONFIG['training']['batch_size'],
        per_device_eval_batch_size=CONFIG['training']['batch_size'] * 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay'],
        warmup_steps=CONFIG['training']['warmup_steps'],
        fp16=CONFIG['training']['fp16'],
        gradient_accumulation_steps=CONFIG['training']['grad_accum_steps'],
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Train
    print("Training model...")
    trainer.train()

    # Evaluate
    print("Evaluating on test set...")
    results = trainer.evaluate(tokenized_datasets['test'])
    filtered_results = {k: results[k] for k in ['eval_accuracy','eval_f1','eval_precision','eval_recall']}
    print("\nTest Results:", filtered_results)

    # Save model and tokenizer
    save_dir = "./saved_model_albert"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    # Save metrics JSON
    save_metrics("ALBERT", filtered_results, CONFIG)

if __name__ == "__main__":
    main()

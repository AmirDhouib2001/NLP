
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import evaluate
import torch

# Configuration
MODEL_NAME = "distilbert-base-uncased"

# Chemins - fonctionnent depuis n'importe quel dossier
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "sequence_classifier")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "question_classif - question_classif.csv")
HF_MODEL_NAME = "AmirDHOUIB/distilbert-finetuned-query-classifier-5IABD2"

# Charger les données
print("Chargement des données...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset chargé: {len(df)} exemples")
print(f"Distribution des classes:")
print(df['label_text'].value_counts())

# Split train/validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"\nTrain: {len(train_df)} exemples")
print(f"Validation: {len(val_df)} exemples")

# Créer les datasets
train_dataset = Dataset.from_pandas(train_df[['question', 'label']])
val_dataset = Dataset.from_pandas(val_df[['question', 'label']])

# Charger le tokenizer
print("\nChargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Fonction de tokenization
def tokenize_function(examples):
    return tokenizer(examples["question"], truncation=True, padding=False, max_length=128)

# Tokenizer les données
print("Tokenization des données...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Charger le modèle
print("\nChargement du modèle...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "question_rag", 1: "send_message"},
    label2id={"question_rag": 0, "send_message": 1}
)

# Geler les premières couches (transfer learning avec peu de données)
# On ne fine-tune que les 2 dernières couches du transformer
for param in model.distilbert.transformer.layer[:-2].parameters():
    param.requires_grad = False

print(f"Nombre de paramètres entraînables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Métriques
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,  
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    warmup_steps=50,
    no_cuda=True,  
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Entraînement
print("\n" + "="*50)
print("DÉBUT DE L'ENTRAÎNEMENT")
print("="*50)
trainer.train()

# Évaluation finale
print("\n" + "="*50)
print("ÉVALUATION FINALE")
print("="*50)
eval_results = trainer.evaluate()
print(f"Résultats: {eval_results}")

# Sauvegarder le modèle
print("\nSauvegarde du modèle...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modèle sauvegardé dans {OUTPUT_DIR}")

# Test rapide
print("\n" + "="*50)
print("TESTS")
print("="*50)

test_queries = [
    "Does the React course cover the use of hooks?",
    "Ask the python teacher when is the next class",
    "What are the prerequisites for machine learning?",
    "Send a message to John about the meeting"
]

for query in test_queries:
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_label = model.config.id2label[predicted_class]
    confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
    
    print(f"\nQuery: {query}")
    print(f"Prédiction: {predicted_label} (confiance: {confidence:.2%})")

print("\n" + "="*50)
print("UPLOAD SUR HUGGINGFACE")
print("="*50)

from huggingface_hub import HfApi

api = HfApi()

try:
    api.create_repo(repo_id=HF_MODEL_NAME, repo_type="model", exist_ok=True)
    print(f"✓ Repository créé/vérifié: {HF_MODEL_NAME}")
except Exception as e:
    print(f"Info: {e}")

# Uploader le modèle
api.upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=HF_MODEL_NAME,
    repo_type="model"
)

print(f"\n✅ Modèle uploadé sur HuggingFace: {HF_MODEL_NAME}")


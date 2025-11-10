

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import evaluate
import torch

MODEL_NAME = "distilbert-base-uncased"

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "token_classifier")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "train_2.csv")
HF_MODEL_NAME = "AmirDHOUIB/distilbert-finetuned-token-classifier-5IABD2"


label_list = ["O", "B-PERSON", "I-PERSON", "B-CONTENT", "I-CONTENT"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print("Configuration:")
print(f"Modèle de base: {MODEL_NAME}")
print(f"Labels: {label_list}")
print(f"Dataset: {DATA_PATH}")
print(f"Output directory: {OUTPUT_DIR}")


def convert_to_bio_format(labels):
    """
    Convertir les labels simples ("person", "content", 0) en format BIO.
    
    Args:
        labels: Liste de labels (ex: [0, "person", "person", "content", "content"])
    
    Returns:
        Liste de labels BIO (ex: ["O", "B-PERSON", "I-PERSON", "B-CONTENT", "I-CONTENT"])
    """
    bio_labels = []
    previous_label = None
    
    for label in labels:
        # Convertir en string et nettoyer
        label_str = str(label).strip().lower()
        
        if label_str == "0" or label_str == "o" or label_str == "":
            bio_labels.append("O")
            previous_label = None
        elif label_str == "person":
            if previous_label == "person":
                bio_labels.append("I-PERSON")
            else:
                bio_labels.append("B-PERSON")
            previous_label = "person"
        elif label_str == "content":
            if previous_label == "content":
                bio_labels.append("I-CONTENT")
            else:
                bio_labels.append("B-CONTENT")
            previous_label = "content"
        else:
            bio_labels.append("O")
            previous_label = None
    
    return bio_labels


def load_train_2_data(file_path):
    """
    Charger les données du fichier train_2.csv.
    
    Format attendu:
    - words: liste de mots au format JSON string
    - labels: liste de labels au format JSON string
    """
    print(f"\nChargement du fichier {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset chargé: {len(df)} exemples")
    print(f"Colonnes: {df.columns.tolist()}")
    
    sentences = []
    labels = []
    
    for idx, row in df.iterrows():
        try:
            # Parser les listes JSON
            words = ast.literal_eval(row['words'])
            row_labels = ast.literal_eval(row['labels'])
            
            # Convertir en format BIO
            bio_labels = convert_to_bio_format(row_labels)
            
            # Vérifier que les longueurs correspondent
            if len(words) == len(bio_labels):
                sentences.append(words)
                labels.append(bio_labels)
            else:
                print(f"Warning: Ligne {idx} - longueur mismatch: {len(words)} mots vs {len(bio_labels)} labels")
        
        except Exception as e:
            print(f"Erreur ligne {idx}: {e}")
            continue
    
    print(f"Exemples valides: {len(sentences)}")
    
    # Statistiques sur les labels
    label_counts = {"O": 0, "B-PERSON": 0, "I-PERSON": 0, "B-CONTENT": 0, "I-CONTENT": 0}
    for label_seq in labels:
        for label in label_seq:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nDistribution des labels:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Afficher quelques exemples
    print("\nExemples de données:")
    for i in range(min(3, len(sentences))):
        print(f"\nExemple {i+1}:")
        print(f"Phrase: {' '.join(sentences[i])}")
        print("Labels:")
        for word, label in zip(sentences[i], labels[i]):
            print(f"  {word:20s} -> {label}")
    
    return sentences, labels


# Charger les données
print("\n" + "="*70)
print("CHARGEMENT DES DONNÉES")
print("="*70)
sentences, labels = load_train_2_data(DATA_PATH)

# Créer un DataFrame
data = []
for words, tags in zip(sentences, labels):
    data.append({"tokens": words, "ner_tags": [label2id[tag] for tag in tags]})

df = pd.DataFrame(data)
print(f"\nNombre total d'exemples: {len(df)}")

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Train: {len(train_data)} exemples")
print(f"Validation: {len(val_data)} exemples")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print("\nChargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(examples):
    """
    Tokenizer les mots et aligner les labels avec les tokens
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # Les tokens spéciaux ont word_idx = None
            if word_idx is None:
                label_ids.append(-100)  # Ignorer ces tokens dans la loss
            # On met le label seulement sur le premier token de chaque mot
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Pour les tokens suivants du même mot, on ignore
            else:
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


print("\n" + "="*70)
print("TOKENIZATION")
print("="*70)
print("Tokenization des données...")
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
print("Tokenization terminée!")

print("\n" + "="*70)
print("CHARGEMENT DU MODÈLE")
print("="*70)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Geler les premières couches (transfer learning avec peu de données)
print("\nGel des premières couches pour le transfer learning...")
for param in model.distilbert.transformer.layer[:-2].parameters():
    param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Paramètres entraînables: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Retirer les tokens ignorés (label = -100)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    warmup_steps=100,
    no_cuda=True,  # Forcer l'utilisation du CPU (évite les problèmes CUDA)
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
print("\n" + "="*70)
print("DÉBUT DE L'ENTRAÎNEMENT")
print("="*70)
trainer.train()

print("\n" + "="*70)
print("ÉVALUATION FINALE")
print("="*70)
eval_results = trainer.evaluate()
print(f"Résultats: {eval_results}")

print("\nSauvegarde du modèle...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modèle sauvegardé dans {OUTPUT_DIR}")

print("\n" + "="*70)
print("TESTS avec predict_at_word_level")
print("="*70)

from models import predict_at_word_level

test_sentences = [
    "Ask the python teacher when is the next class".split(),
    "Send a message to John about the meeting".split(),
    "Write to Mom telling her I'll be home late".split(),
]

for words in test_sentences:
    predictions = predict_at_word_level(words, model, tokenizer)
    predicted_labels = [id2label[pred] for pred in predictions]
    
    print(f"\nPhrase: {' '.join(words)}")
    print("Prédictions:")
    for word, label in zip(words, predicted_labels):
        print(f"  {word:20s} -> {label}")

print("\n" + "="*70)
print("UPLOAD SUR HUGGINGFACE")
print("="*70)

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

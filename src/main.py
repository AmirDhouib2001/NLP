import click
import joblib
import pandas as pd
import os
from sklearn.model_selection import cross_val_score
import numpy as np

from data import make_dataset
from feature import make_features
from models import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/model.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    print(f"Chargement des données depuis {input_filename}")
    df = make_dataset(input_filename)
    
    print(f"Transformation des features avec CountVectorizer")
    X, y, vectorizer = make_features(df, fit_vectorizer=True)
    print(f"Shape des features: {X.shape}")
    print(f"Nombre de mots dans le vocabulaire: {len(vectorizer.vocabulary_)}")

    print("Entraînement du modèle")
    model = make_model()
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_dump_filename), exist_ok=True)
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer
    }
    
    joblib.dump(model_data, model_dump_filename)
    print(f"Modèle et vectorizer sauvegardés dans {model_dump_filename}")
    
    train_score = model.score(X, y)
    print(f"Accuracy sur les données d'entraînement: {train_score:.3f}")


@click.command()
@click.option("--input_filename", default="data/raw/test.csv", help="File with data to predict")
@click.option("--model_dump_filename", default="models/model.json", help="File with dumped model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    """Pipeline de prédiction"""
    print(f"Chargement du modèle depuis {model_dump_filename}")
    model_data = joblib.load(model_dump_filename)
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    
    print(f"Chargement des données à prédire depuis {input_filename}")
    df = make_dataset(input_filename)
    
    print("Transformation des features avec le même vectorizer")
    X, y_true, _ = make_features(df, vectorizer=vectorizer, fit_vectorizer=False)
    print(f"Shape des features: {X.shape}")
    
    print("Prédiction")
    predictions = model.predict(X)
    predictions_proba = model.predict_proba(X)
    
    # Calculer et afficher l'accuracy si on a les vraies labels
    if y_true is not None:
        accuracy = model.score(X, y_true)
        print(f"Accuracy sur les données de test: {accuracy:.3f}")
    else:
        print("Pas de labels dans le fichier, impossible de calculer l'accuracy")
    
    # Créer le DataFrame de résultats
    results = pd.DataFrame({
        'video_name': df['video_name'],
        'prediction': predictions,
        'probability_comic': predictions_proba[:, 1]  
    })
    
    # Ajouter les vraies labels si disponibles
    if 'is_comic' in df.columns:
        results['true_label'] = df['is_comic']
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    results.to_csv(output_filename, index=False)
    print(f"Prédictions sauvegardées dans {output_filename}")
    
    print(f"Nombre de vidéos prédites comme comiques: {predictions.sum()}")
    print(f"Proportion prédite comme comique: {predictions.mean():.3f}")


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--cv", default=5, help="Number of cross-validation folds")
def evaluate(input_filename, cv):
    """Évalue le modèle avec cross-validation"""
    print(f"Évaluation avec cross-validation ({cv}-fold)")
    
    print(f"Chargement des données depuis {input_filename}")
    df = make_dataset(input_filename)

    print("Transformation des features avec CountVectorizer")
    X, y, vectorizer = make_features(df, fit_vectorizer=True)
    print(f"Shape des features: {X.shape}")
    print(f"Nombre de mots dans le vocabulaire: {len(vectorizer.vocabulary_)}")

    model = make_model()

    return evaluate_model(model, X, y, cv)


def evaluate_model(model, X, y, cv=5):
    """Run k-fold cross validation et affiche les résultats"""
    print(f"\nCross-validation avec {cv} folds...")
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"\nRésultats par fold:")
    for i, score in enumerate(scores, 1):
        print(f"  Fold {i}: {score:.3f}")
    
    print(f"\nStatistiques globales:")
    print(f"  Accuracy moyenne: {scores.mean():.3f}")
    print(f"  Écart-type: {scores.std():.3f}")
    print(f"  Accuracy min: {scores.min():.3f}")
    print(f"  Accuracy max: {scores.max():.3f}")
    print(f"  Intervalle de confiance 95%: [{scores.mean() - 2*scores.std():.3f}, {scores.mean() + 2*scores.std():.3f}]")
    
    return scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

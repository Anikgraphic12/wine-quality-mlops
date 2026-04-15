"""
Script d'entraînement - Wine Quality Dataset
- Charge le dataset winequalityN.csv
- Prétraitement simple (encodage type, standardisation)
- Entraîne un RandomForestClassifier
- Évalue le modèle (accuracy, F1-score)
- Sauvegarde le modèle et le scaler dans artifacts/
"""

import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report


def main():
    # --- Chargement ---
    df = pd.read_csv("winequalityN.csv")
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # --- Prétraitement ---
    # Encodage de la colonne 'type' (white=1, red=0)
    le = LabelEncoder()
    df["type"] = le.fit_transform(df["type"])

    # Suppression des lignes avec valeurs manquantes
    df.dropna(inplace=True)

    # Features et cible
    # On binarise la qualité : >= 7 = bon (1), sinon mauvais (0)
    X = df.drop(columns=["quality"])
    y = (df["quality"] >= 7).astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Entraînement ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # --- Évaluation ---
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy  : {acc:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nRapport détaillé :")
    print(classification_report(y_test, y_pred, target_names=["Mauvais", "Bon"]))

    # --- Sauvegarde ---
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    joblib.dump(scaler, "artifacts/scaler.pkl")

    # Sauvegarde des noms de features pour l'API
    feature_names = list(X.columns)
    with open("artifacts/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    metrics = {"accuracy": round(float(acc), 4), "f1_score": round(float(f1), 4)}
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("\nModèle sauvegardé dans artifacts/model.pkl")
    print("Scaler sauvegardé dans artifacts/scaler.pkl")


if __name__ == "__main__":
    main()

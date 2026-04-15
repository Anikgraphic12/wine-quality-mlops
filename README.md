# Wine Quality MLOps

Projet MLOps — prédiction de la qualité du vin.

## Stack
- Python 3.11 · scikit-learn · FastAPI · Docker · GitHub Actions

## Lancer localement

```bash
pip install -r requirements.txt
python train.py          # entraîne et sauvegarde le modèle
python app.py            # démarre l'API sur http://localhost:8000
```

## Tester l'API

```bash
# Santé
curl http://localhost:8000/health

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": 1,
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.27,
    "citric_acid": 0.36,
    "residual_sugar": 20.7,
    "chlorides": 0.045,
    "free_sulfur_dioxide": 45.0,
    "total_sulfur_dioxide": 170.0,
    "density": 1.001,
    "pH": 3.0,
    "sulphates": 0.45,
    "alcohol": 8.8
  }'
```

## Docker

```bash
docker build -t wine-quality-api .
docker run -p 8000:8000 wine-quality-api
```

## Pipeline CI (GitHub Actions)

| Branche       | Jobs exécutés                                      |
|---------------|----------------------------------------------------|
| `feature/*`   | install deps → train                               |
| `develop`     | install deps → train → build Docker → push Docker  |

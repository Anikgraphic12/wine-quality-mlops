FROM python:3.11-slim

WORKDIR /app

# Copie des dépendances en premier (cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY train.py .
COPY app.py .
COPY winequalityN.csv .

# Entraînement du modèle au moment du build
RUN python train.py

EXPOSE 8000

CMD ["python", "app.py"]

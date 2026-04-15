FROM python:3.11-slim

WORKDIR /app

# dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code complet (IMPORTANT)
COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
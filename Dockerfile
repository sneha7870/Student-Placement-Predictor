# ---- Student Placement Predictor: Docker image ----
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Train the model at build time so model.pkl is baked into the image
RUN python train_model.py

# Render (and most PaaS) inject PORT; default to 5000 for local runs
ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app"]

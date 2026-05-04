# ── Base image (as required by question paper) ────────────────────────────────
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (enables Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY data/    ./data/
COPY src/     ./src/
COPY models/  ./models/
COPY results/ ./results/

# Unbuffered Python output (important for CLI tools)
ENV PYTHONUNBUFFERED=1

# Default entrypoint: CLI predictor
ENTRYPOINT ["python", "src/predict_cli.py"]

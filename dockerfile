FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the TensorFlow model
COPY ModelCNN2.h5 .

# Copy application code
COPY . .

# Environment variables
ENV PORT=8080

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Increase timeout for model loading
ENV UVICORN_TIMEOUT=300

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Run the application with proper worker configuration
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75
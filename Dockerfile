FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the code needed for inference/testing
COPY app/ ./app
COPY models/ ./models

# Ensure model file exists
RUN test -f models/last.pt || (echo "Model file missing: models/last.pt" && exit 1)

CMD ["python", "app/main_logic.py"]

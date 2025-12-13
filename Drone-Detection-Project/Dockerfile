# -----------------------------
# Dockerfile for Drone Detection Project (headless)
# -----------------------------

FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# -----------------------------
# Install system dependencies (minimal)
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    wget \
    unzip \
    unixodbc \
    unixodbc-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy requirements and install Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# -----------------------------
# Copy application code
# -----------------------------
COPY . .

# -----------------------------
# Create data directory (if not using mounted volume)
# -----------------------------
RUN mkdir -p /app/Data

# -----------------------------
# Verify model file exists
# -----------------------------
RUN test -f models/YOLOv5_model/weights/last.pt || \
    (echo "Model file missing!" && exit 1)

# -----------------------------
# Run headless by default
# -----------------------------
CMD ["python", "-m", "app.main_logic"]

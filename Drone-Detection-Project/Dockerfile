# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY Drone-Detection-Project/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY Drone-Detection-Project/app ./app

# Copy model weights
COPY Drone-Detection-Project/models/YOLOv5_model/weights/last.pt ./models/YOLOv5_model/weights/last.pt

# Optional: verify that the model file exists
RUN test -f models/YOLOv5_model/weights/last.pt || \
    (echo "Model file missing: models/YOLOv5_model/weights/last.pt" && exit 1)

# Default command: run the headless main logic
CMD ["python", "-m", "app.main_logic"]


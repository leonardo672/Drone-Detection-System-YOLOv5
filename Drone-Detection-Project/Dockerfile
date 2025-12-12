FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ ./app

# Copy model
COPY models/YOLOv5_model/weights/last.pt ./models/YOLOv5_model/weights/last.pt

# Verify model
RUN test -f models/YOLOv5_model/weights/last.pt || \
    (echo "Model file missing!" && exit 1)

CMD ["python", "-m", "app.main_logic"]

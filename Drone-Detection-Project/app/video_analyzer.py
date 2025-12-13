import cv2
import torch
from pathlib import Path
import time

class VideoAnalyzer:
    def __init__(self, model_path):
        # Load local YOLOv5 model via torch hub
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=model_path,
            force_reload=True
        )

    def detect_and_save_objects(self, input_path, output_folder="data/images/Output",
                                new_width=640, new_height=480):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (new_width, new_height))
            results = self.model(frame)

            detected_objects = results.xyxy[0]
            for obj in detected_objects:
                x1, y1, x2, y2, conf, cls = obj
                obj_img = frame[int(y1):int(y2), int(x1):int(x2)]
                if obj_img.size > 0:
                    timestamp = int(time.time() * 1000)
                    filename = output_folder / f"{Path(input_path).stem}_{int(x1)}_{int(y1)}_{timestamp}.jpg"
                    cv2.imwrite(str(filename), obj_img)

        cap.release()
        cv2.destroyAllWindows()

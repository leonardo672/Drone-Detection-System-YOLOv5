import cv2
import torch
from pathlib import Path

class ImageAnalyzer:
    def __init__(self, model_path):
        # Load local YOLOv5 model via torch hub
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=model_path,
            force_reload=True
        )

    def analyze_image(self, image_path, output_folder="data/images/Output"):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(image_path))
        results = self.model(img)
        detected_objects = results.xyxy[0]

        for obj in detected_objects:
            x1, y1, x2, y2, conf, cls = obj
            obj_img = img[int(y1):int(y2), int(x1):int(x2)]
            if obj_img.size > 0:
                filename = output_folder / f"{Path(image_path).stem}_{int(x1)}_{int(y1)}.jpg"
                cv2.imwrite(str(filename), obj_img)

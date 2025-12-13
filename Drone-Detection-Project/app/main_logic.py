from .image_analyzer import ImageAnalyzer
from .video_analyzer import VideoAnalyzer
from pathlib import Path

def main():
    model_path = "models/YOLOv5_model/weights/last.pt"
    image_folder = Path("data/images/TestImages")
    video_folder = Path("data/videos")
    output_folder = Path("data/images/Output")
    output_folder.mkdir(parents=True, exist_ok=True)

    analyzer = ImageAnalyzer(model_path=model_path)
    for img_path in image_folder.glob("*.jpg"):
        analyzer.analyze_image(img_path, output_folder=output_folder)

    video_analyzer = VideoAnalyzer(model_path=model_path)
    for video_name in ["Bayraktar.mp4", "Phantom4.mp4"]:
        video_file = video_folder / video_name
        if video_file.exists():
            video_analyzer.detect_and_save_objects(video_file, output_folder=output_folder)
        else:
            print(f"Video {video_name} not found in {video_folder}")

if __name__ == "__main__":
    main()

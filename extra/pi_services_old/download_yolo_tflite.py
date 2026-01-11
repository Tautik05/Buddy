"""Download YOLOv8 TensorFlow Lite model"""

import requests
import os

def download_yolo_tflite():
    """Download YOLOv8n TensorFlow Lite model"""
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # YOLOv8n TFLite model URL (you may need to update this URL)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.tflite"
    
    model_path = "models/yolov8n.tflite"
    
    if os.path.exists(model_path):
        print(f"✅ Model already exists: {model_path}")
        return
    
    print("📥 Downloading YOLOv8n TensorFlow Lite model...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded: {model_path}")
        print(f"📊 Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Alternative: Convert from PyTorch using:")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('yolov8n.pt')")
        print("   model.export(format='tflite')")

if __name__ == "__main__":
    download_yolo_tflite()
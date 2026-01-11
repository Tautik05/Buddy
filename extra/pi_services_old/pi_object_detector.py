"""
Lightweight Object Detection for Raspberry Pi
Optimized for Pi performance with reduced model complexity
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class PiObjectDetector:
    """Optimized object detector for Raspberry Pi"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        
        # Load YOLOv8 nano model (smallest/fastest)
        self.model = YOLO(model_path)
        
        # Common object classes we care about
        self.target_classes = {
            'person', 'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'
        }
    
    def detect(self, frame: np.ndarray, max_detections: int = 10) -> List[Dict[str, Any]]:
        """Detect objects in frame with Pi optimizations"""
        try:
            # Resize frame for faster processing on Pi
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
                scale = 1.0
            
            # Run inference with reduced image size
            results = self.model(frame_resized, verbose=False, conf=self.confidence_threshold)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # Only include target classes
                        if class_name in self.target_classes:
                            confidence = float(box.conf[0])
                            
                            # Scale coordinates back to original frame size
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, x2 = int(x1 / scale), int(x2 / scale)
                            y1, y2 = int(y1 / scale), int(y2 / scale)
                            
                            detections.append({
                                'name': class_name,
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2]
                            })
            
            # Sort by confidence and limit results
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            return detections[:max_detections]
            
        except Exception as e:
            print(f"Object detection error: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection boxes on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            name = detection['name']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def get_object_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get summary of detected objects"""
        summary = {}
        for detection in detections:
            name = detection['name']
            summary[name] = summary.get(name, 0) + 1
        return summary
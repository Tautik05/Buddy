from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

class ObjectDetector:
    def __init__(self, confidence_threshold=0.8):  # Higher threshold for Pi
        self.model = YOLO("yolov8n.pt")  # Nano model for Pi
        self.model.fuse()  # Optimize for inference
        self.confidence_threshold = confidence_threshold
        
        # Reduced target classes for Pi performance
        self.target_classes = {
            'bottle', 'cup', 'cell phone', 'laptop', 'book', 'apple', 'banana'
        }
        
        # Larger minimum areas to reduce false positives
        self.min_areas = {
            'cell phone': 1500,
            'laptop': 8000,
            'cup': 1200,
            'bottle': 1500,
            'book': 3000
        }

    def detect(self, frame):
        detected_objects = []
        
        try:
            # Resize frame for faster processing on Pi
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            results = self.model(frame, verbose=False, conf=self.confidence_threshold, 
                               imgsz=640, half=True)  # Use half precision for Pi
            
            for r in results:
                for box in r.boxes:
                    if box.conf[0] >= self.confidence_threshold:
                        cls_id = int(box.cls[0])
                        class_name = self.model.names[cls_id]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        
                        # Only include target objects
                        if class_name in self.target_classes:
                            # Calculate area for size filtering
                            x1, y1, x2, y2 = bbox
                            area = (x2 - x1) * (y2 - y1)
                            
                            # Apply size filter
                            min_area = self.min_areas.get(class_name, 1000)
                            if area >= min_area:
                                detected_objects.append({
                                    'name': class_name,
                                    'confidence': confidence,
                                    'bbox': bbox,
                                    'area': area
                                })
            
            # Keep only top 3 detections for Pi
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            return detected_objects[:3]
            
        except Exception as e:
            return []
    
    def _remove_overlaps(self, detections):
        """Remove overlapping detections that might be the same object"""
        if len(detections) <= 1:
            return detections
        
        filtered = []
        for i, det1 in enumerate(detections):
            is_duplicate = False
            
            for j, det2 in enumerate(detections):
                if i != j and self._calculate_iou(det1['bbox'], det2['bbox']) > 0.3:
                    # Keep the one with higher confidence
                    if det1['confidence'] < det2['confidence']:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(det1)
        
        return filtered
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            name = detection['name']
            confidence = detection['confidence']
            
            # Use different colors for different object types
            if 'phone' in name or 'laptop' in name or 'mouse' in name:
                color = (255, 0, 0)  # Blue for tech
            elif 'cup' in name or 'bottle' in name or 'glass' in name:
                color = (0, 255, 255)  # Yellow for drinks
            elif 'chair' in name or 'couch' in name:
                color = (128, 0, 128)  # Purple for furniture
            else:
                color = (0, 255, 0)  # Green for others
            
            # Draw thicker bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label = f"{name} ({confidence:.2f})"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-label_height-10), (x1+label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

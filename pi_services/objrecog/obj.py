import tensorflow as tf
import cv2
import numpy as np
from collections import Counter

class ObjectDetector:
    def __init__(self, confidence_threshold=0.6):
        # Load YOLOv8 TFLite model
        self.interpreter = tf.lite.Interpreter(model_path="models/yolov8n_float32.tflite")
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.confidence_threshold = confidence_threshold
        self.input_size = self.input_details[0]['shape'][1]  # Usually 640
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # More common objects that YOLO can actually detect
        self.target_classes = {
            'bottle', 'cup', 'cell phone', 'laptop', 'book', 'mouse', 'keyboard',
            'tv', 'remote', 'chair', 'clock', 'vase', 'potted plant',
            'spoon', 'fork', 'knife', 'bowl', 'banana', 'apple', 'orange'
        }
        
        # Lower minimum areas
        self.min_areas = {
            'cell phone': 1000,
            'laptop': 5000,
            'cup': 800,
            'bottle': 1000,
            'book': 2000,
            'mouse': 500,
            'keyboard': 3000
        }

    def preprocess_image(self, image):
        """Preprocess image for YOLO TFLite model"""
        # Resize to model input size
        input_image = cv2.resize(image, (self.input_size, self.input_size))
        # Normalize to [0, 1]
        input_image = input_image.astype(np.float32) / 255.0
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def detect(self, frame):
        detected_objects = []
        
        try:
            # Preprocess image
            input_image = self.preprocess_image(frame)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            print(f"DEBUG: YOLO output shape: {output_data.shape}")
            print(f"DEBUG: Output sample: {output_data[0][:5] if len(output_data[0]) > 5 else output_data[0]}")
            
            # Process detections
            detections = self.process_detections(output_data[0], frame.shape)
            
            print(f"DEBUG: Raw detections: {len(detections)}")
            for det in detections[:3]:  # Show first 3
                print(f"DEBUG: {det['name']} - {det['confidence']:.2f}")
            
            # Filter by target classes and minimum area
            for detection in detections:
                if detection['name'] in self.target_classes:
                    area = detection['area']
                    min_area = self.min_areas.get(detection['name'], 1500)
                    if area >= min_area:
                        detected_objects.append(detection)
            
            print(f"DEBUG: Filtered detections: {[d['name'] for d in detected_objects]}")
            
            # Keep only top 2 detections
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            return detected_objects[:2]
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def process_detections(self, output, original_shape):
        """Process YOLO output to get bounding boxes and classes"""
        detections = []
        
        # YOLO output format: [batch, num_detections, 85] where 85 = 4 (bbox) + 1 (conf) + 80 (classes)
        for detection in output:
            # Extract confidence and class scores
            confidence = detection[4]
            class_scores = detection[5:]
            
            if confidence > self.confidence_threshold:
                # Get class with highest score
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                if class_confidence > self.confidence_threshold:
                    # Extract bounding box (center_x, center_y, width, height)
                    center_x, center_y, width, height = detection[:4]
                    
                    # Convert to corner coordinates and scale to original image
                    orig_h, orig_w = original_shape[:2]
                    x1 = int((center_x - width/2) * orig_w / self.input_size)
                    y1 = int((center_y - height/2) * orig_h / self.input_size)
                    x2 = int((center_x + width/2) * orig_w / self.input_size)
                    y2 = int((center_y + height/2) * orig_h / self.input_size)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))
                    
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > 0 and class_id < len(self.class_names):
                        detections.append({
                            'name': self.class_names[class_id],
                            'confidence': float(confidence * class_confidence),
                            'bbox': [x1, y1, x2, y2],
                            'area': area
                        })
        
        return detections
    
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

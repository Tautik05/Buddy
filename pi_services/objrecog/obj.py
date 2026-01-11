import tensorflow as tf
import cv2
import numpy as np
from collections import Counter

class ObjectDetector:
    def __init__(self, confidence_threshold=0.8):  # Much higher threshold
        # Load YOLOv8 TFLite model (int8 quantized for Raspberry Pi)
        self.interpreter = tf.lite.Interpreter(model_path="models/yolov8n_int8.tflite")
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
        
        # Only most reliable objects
        self.target_classes = {
            'bottle', 'cup', 'laptop', 'book', 'spoon', 'fork', 'knife', 'bowl'
        }
        
        # Much higher minimum areas to reduce false positives
        self.min_areas = {
            'bottle': 3000,
            'cup': 2000,
            'laptop': 10000,
            'book': 4000,
            'spoon': 1000,
            'fork': 1000,
            'knife': 1000,
            'bowl': 3000
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
            
            # Process detections with strict filtering
            detections = self.process_detections_strict(output_data[0], frame.shape)
            
            print(f"DEBUG: Found {len(detections)} valid detections")
            for det in detections:
                print(f"DEBUG: {det['name']} - {det['confidence']:.2f} (area: {det['area']:.0f})")
            
            return detections[:1]  # Only return top 1 detection
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def process_detections_strict(self, output, original_shape):
        """Process YOLO output with strict filtering"""
        detections = []
        
        try:
            # Handle different output shapes
            if len(output.shape) == 1:
                # Reshape if needed
                if output.shape[0] == 8400 * 84:
                    output = output.reshape(8400, 84)
                else:
                    print(f"DEBUG: Unexpected output shape: {output.shape}")
                    return []
            
            print(f"DEBUG: Processing output shape: {output.shape}")
            
            # YOLOv8 output format: [8400, 84] where 84 = 4 (bbox) + 80 (classes)
            for i in range(min(output.shape[0], 1000)):  # Limit processing
                detection = output[i]
                
                # Extract bounding box (center_x, center_y, width, height)
                center_x, center_y, width, height = detection[:4]
                
                # Extract class scores (no objectness in YOLOv8)
                class_scores = detection[4:]
                
                # Get class with highest score
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                # Very strict confidence filtering
                if class_confidence > self.confidence_threshold:
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                        
                        # Only process target classes
                        if class_name in self.target_classes:
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
                            min_area = self.min_areas.get(class_name, 3000)
                            
                            # Strict area filtering
                            if area >= min_area:
                                detections.append({
                                    'name': class_name,
                                    'confidence': float(class_confidence),
                                    'bbox': [x1, y1, x2, y2],
                                    'area': area
                                })
            
            # Sort by confidence and return only the best
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            return detections
            
        except Exception as e:
            print(f"DEBUG: Detection processing error: {e}")
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

import tensorflow as tf
import cv2
import numpy as np
import os

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        # Use Teachable Machine model instead of broken YOLO
        model_path = "models/model.tflite"
        labels_path = "models/labels.txt"
        
        print(f"🔍 Loading Teachable Machine model: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Teachable Machine model not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model with Pi-specific fixes
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=[]  # Disable XNNPACK to prevent crashes on Pi
        )
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.confidence_threshold = confidence_threshold
        
        # Load labels
        self.class_names = self.load_labels(labels_path)
        
        print(f"✅ Teachable Machine model loaded")
        print(f"📊 Input size: {self.input_width}x{self.input_height}")
        print(f"🏷️ Classes: {self.class_names}")
    
    def load_labels(self, labels_path):
        """Load class labels"""
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            return labels
        else:
            # Default labels for your model
            return ["0 bottle", "1 cup", "2 nothing"]
    
    def preprocess_image(self, image):
        """Preprocess image for Teachable Machine model"""
        # Resize to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Handle different input types with proper np.float32 conversion
        if self.input_details[0]['dtype'] == np.uint8:
            # Keep as uint8 (0-255)
            processed = rgb_image.astype(np.uint8)
        else:
            # Ensure exactly np.float32 for Pi compatibility
            processed = np.array(rgb_image, dtype=np.float32) / 255.0
        
        # Add batch dimension - ensure shape is correct
        processed = np.expand_dims(processed, axis=0)
        return processed
    
    def detect(self, frame):
        """Detect objects using Teachable Machine model"""
        try:
            # Preprocess
            input_data = self.preprocess_image(frame)
            
            # Set input
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions = output_data[0]  # Remove batch dimension
            
            # Get top prediction
            top_index = np.argmax(predictions)
            top_confidence = float(predictions[top_index])
            top_class = self.class_names[top_index] if top_index < len(self.class_names) else f"Class_{top_index}"
            
            # Only return detection if it's not "nothing" and above threshold
            if top_confidence > self.confidence_threshold and "nothing" not in top_class.lower():
                # Extract just the object name (remove "0 ", "1 " prefixes)
                object_name = top_class.split(" ", 1)[-1] if " " in top_class else top_class
                
                return [{
                    'name': object_name,
                    'confidence': top_confidence,
                    'bbox': [0, 0, frame.shape[1], frame.shape[0]],  # Full frame as bbox
                    'area': frame.shape[0] * frame.shape[1]
                }]
            
            return []  # No objects detected
            
        except Exception as e:
            print(f"Teachable Machine detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame with proper visual feedback"""
        if not detections:
            return frame
            
        for detection in detections:
            name = detection['name']
            confidence = detection['confidence']
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - low confidence
            
            h, w = frame.shape[:2]
            
            # Draw thick colored border around entire frame
            border_thickness = 8
            cv2.rectangle(frame, (0, 0), (w, h), color, border_thickness)
            
            # Draw detection box in center (simulated object location)
            center_x, center_y = w // 2, h // 2
            box_size = 150
            x1 = center_x - box_size // 2
            y1 = center_y - box_size // 2
            x2 = center_x + box_size // 2
            y2 = center_y + box_size // 2
            
            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            label = f"{name.upper()} {confidence:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Label background
            cv2.rectangle(frame, (x1, y1-35), (x1 + label_size[0] + 10, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Draw object icon in corner
            if "bottle" in name.lower():
                cv2.putText(frame, "BOTTLE", (w-120, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            elif "cup" in name.lower():
                cv2.putText(frame, "CUP", (w-80, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add detection indicator
            cv2.circle(frame, (30, 30), 15, color, -1)
            cv2.putText(frame, "!", (25, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return frame
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
        
        # Load model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
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
        
        # Handle different input types
        if self.input_details[0]['dtype'] == np.uint8:
            # Keep as uint8 (0-255)
            processed = rgb_image.astype(np.uint8)
        else:
            # Normalize to float32 (0-1)
            processed = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension
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
        """Draw detections on frame"""
        for detection in detections:
            name = detection['name']
            confidence = detection['confidence']
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Draw colored border around frame
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (5, 5), (w-5, h-5), color, 6)
            
            # Draw label
            label = f"{name} {confidence:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Draw object icon
            if "bottle" in name.lower():
                cv2.putText(frame, "🍼", (w-60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            elif "cup" in name.lower():
                cv2.putText(frame, "☕", (w-60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        return frame
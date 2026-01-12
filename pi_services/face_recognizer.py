"""
Face Recognition Module
Handles face recognition using TensorFlow Lite model
"""

import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
import logging
import html
from typing import Dict, Tuple
from pi_memory import save_face, get_all_faces


class FaceRecognizer:
    """Handles face recognition using TensorFlow Lite model"""
    
    def __init__(self, model_path: str, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load TensorFlow Lite model with Pi-specific fixes
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, 
            experimental_delegates=[]  # Disable XNNPACK to prevent crashes on Pi
        )
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Debug input shape
        print(f"TFLite input shape: {self.input_details[0]['shape']}")
        print(f"TFLite input dtype: {self.input_details[0]['dtype']}")
        print(f"TFLite output shape: {self.output_details[0]['shape']}")
        print(f"TFLite output dtype: {self.output_details[0]['dtype']}")
        
        self.known_faces: Dict[str, np.ndarray] = {}
        self.load_known_faces()
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for TFLite model"""
        face_resized = cv2.resize(face_img, (112, 112))
        # Ensure exactly np.float32 and correct shape [1, 112, 112, 3]
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_normalized = np.expand_dims(face_normalized, axis=0)  # Add batch dimension
        return face_normalized
    
    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extract face embedding from image"""
        preprocessed = self.preprocess_face(face_img)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
        embedding_flat = embedding.flatten()
        return embedding_flat
    
    def recognize(self, face_img: np.ndarray) -> Tuple[str, float]:
        """Recognize a face from image"""
        try:
            # Enhance image
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_enhanced = cv2.equalizeHist(face_gray)
            face_enhanced = cv2.cvtColor(face_enhanced, cv2.COLOR_GRAY2BGR)
            
            embedding = self.get_embedding(face_enhanced)
            
            if not self.known_faces:
                return "Unknown", 0.0
            
            best_match = None
            min_distance = float('inf')
            all_results = []
            
            # Ensure both embeddings are 1D
            embedding = embedding.flatten() if embedding.ndim > 1 else embedding
            
            for name, known_embedding in self.known_faces.items():
                known_embedding = known_embedding.flatten() if known_embedding.ndim > 1 else known_embedding
                
                # Check if embeddings have same dimensions
                if embedding.shape[0] != known_embedding.shape[0]:
                    print(f"DEBUG: Dimension mismatch for {name}: {embedding.shape[0]} vs {known_embedding.shape[0]}")
                    continue
                    
                distance = cosine(embedding, known_embedding)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = name
            
            if best_match is None:
                return "Unknown", 0.0
            
            final_confidence = 1 - min_distance
            
            if min_distance < self.config.recognition_threshold:
                return best_match, final_confidence
            
            return "Unknown", 0.0
            
        except Exception as e:
            self.logger.error(f"Recognition error: {e}", exc_info=True)
            return "Unknown", 0.0
    
    def add_face(self, name: str, face_img: np.ndarray) -> bool:
        """Add a new face to known faces"""
        try:
            print(f"DEBUG: add_face called for '{name}' with image shape {face_img.shape}")
            embedding = self.get_embedding(face_img)
            print(f"DEBUG: Generated embedding shape: {embedding.shape}")
            
            self.known_faces[name] = embedding
            print(f"DEBUG: Added to known_faces dict, total faces: {len(self.known_faces)}")
            
            result = save_face(name, embedding, 0.95)
            print(f"DEBUG: save_face returned: {result}")
            
            if result:
                print(f"SUCCESS: Face saved for '{name}'")
                # Reload faces to confirm
                self.load_known_faces()
                return True
            else:
                print(f"FAILED: Could not save face for '{name}'")
                return False
            
        except Exception as e:
            print(f"DEBUG: Exception in add_face: {e}")
            self.logger.error(f"Failed to add face for {name}: {e}", exc_info=True)
            return False
    
    def load_known_faces(self) -> None:
        """Load known faces from database"""
        try:
            faces = get_all_faces()
            self.known_faces = {
                html.unescape(name): embedding 
                for name, embedding in faces.items()
            }
            print(f"INFO:face_recognizer:Loaded {len(self.known_faces)} known faces")
            if self.known_faces:
                print(f"Known faces: {list(self.known_faces.keys())}")
            else:
                print("WARNING: No faces found in database!")
            self.logger.info(f"Loaded {len(self.known_faces)} known faces")
        except Exception as e:
            print(f"ERROR loading faces: {e}")
            self.logger.error(f"Failed to load known faces: {e}", exc_info=True)
            self.known_faces = {}
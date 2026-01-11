"""
Lightweight Face Recognition for Raspberry Pi
Optimized for Pi performance
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional
import json
import os

class PiFaceRecognizer:
    """Optimized face recognizer for Raspberry Pi"""
    
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.threshold = threshold
        self.face_database = {}
        self.db_path = "face_database.json"
        
        # Initialize ONNX Runtime with CPU provider only
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        
        # Load existing faces
        self._load_database()
    
    def _load_database(self):
        """Load face database from JSON"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for name, embedding_list in data.items():
                        self.face_database[name] = np.array(embedding_list)
            except Exception as e:
                print(f"Error loading face database: {e}")
    
    def _save_database(self):
        """Save face database to JSON"""
        try:
            data = {}
            for name, embedding in self.face_database.items():
                data[name] = embedding.tolist()
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving face database: {e}")
    
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for recognition"""
        # Resize to model input size (112x112 for MobileFaceNet)
        face_resized = cv2.resize(face_img, (112, 112))
        
        # Normalize
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension and transpose to NCHW
        face_input = np.transpose(face_rgb, (2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)
        
        return face_input
    
    def _get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Get face embedding from preprocessed image"""
        try:
            face_input = self._preprocess_face(face_img)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: face_input})
            
            # Get embedding and normalize
            embedding = outputs[0][0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def add_face(self, name: str, face_img: np.ndarray) -> bool:
        """Add a new face to the database"""
        try:
            embedding = self._get_embedding(face_img)
            if embedding is not None:
                self.face_database[name] = embedding
                self._save_database()
                print(f"Added face for {name}")
                return True
            return False
        except Exception as e:
            print(f"Error adding face for {name}: {e}")
            return False
    
    def recognize(self, face_img: np.ndarray) -> Tuple[str, float]:
        """Recognize face and return name with confidence"""
        try:
            if len(self.face_database) == 0:
                return "Unknown", 0.0
            
            embedding = self._get_embedding(face_img)
            if embedding is None:
                return "Unknown", 0.0
            
            best_match = "Unknown"
            best_similarity = 0.0
            
            # Compare with all known faces
            for name, known_embedding in self.face_database.items():
                similarity = np.dot(embedding, known_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # Check if similarity meets threshold
            if best_similarity >= self.threshold:
                return best_match, best_similarity
            else:
                return "Unknown", best_similarity
                
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return "Unknown", 0.0
    
    def remove_face(self, name: str) -> bool:
        """Remove a face from the database"""
        if name in self.face_database:
            del self.face_database[name]
            self._save_database()
            return True
        return False
    
    def list_faces(self) -> list:
        """List all known faces"""
        return list(self.face_database.keys())
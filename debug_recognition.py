#!/usr/bin/env python3
"""
Face Recognition Debug - Step by Step
"""

import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from memory import get_all_faces

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (112, 112))
    face_normalized = (face_resized - 127.5) / 128.0
    return np.expand_dims(face_normalized.transpose(2, 0, 1), axis=0).astype(np.float32)

def get_embedding(face_img, session, input_name):
    preprocessed = preprocess_face(face_img)
    embedding = session.run(None, {input_name: preprocessed})[0]
    return embedding.flatten()

def recognize_face_debug(face_img, known_faces, session, input_name, threshold=0.3):
    print(f"🔍 Testing face recognition with threshold {threshold}")
    
    # Get embedding for current face
    embedding = get_embedding(face_img, session, input_name)
    print(f"✅ Generated embedding: shape={embedding.shape}, type={type(embedding)}")
    
    best_match = None
    min_distance = float('inf')
    
    print(f"📊 Comparing against {len(known_faces)} known faces:")
    
    for name, known_embedding in known_faces.items():
        distance = cosine(embedding, known_embedding)
        print(f"   - {name}: distance={distance:.4f}")
        
        if distance < min_distance:
            min_distance = distance
            best_match = name
    
    print(f"🎯 Best match: {best_match} (distance: {min_distance:.4f})")
    print(f"🚪 Threshold: {threshold}")
    
    if min_distance < threshold:
        confidence = 1 - min_distance
        print(f"✅ RECOGNIZED: {best_match} (confidence: {confidence:.4f})")
        return best_match, confidence
    else:
        print(f"❌ NOT RECOGNIZED: distance {min_distance:.4f} >= threshold {threshold}")
        return "Unknown", 0.0

def main():
    print("🧠 FACE RECOGNITION DEBUG")
    print("=" * 50)
    
    # Load model
    model_path = "face-recog/MobileFaceNet.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    print(f"✅ Loaded model: {model_path}")
    
    # Load known faces
    known_faces = get_all_faces()
    print(f"✅ Loaded {len(known_faces)} known faces")
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("\n📹 Camera started. Press:")
    print("   'r' - Test recognition on current face")
    print("   't' - Test with different thresholds")
    print("   'q' - Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 50))
        
        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{w}x{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(frame, f"Faces: {len(faces)} | Press 'r' to test recognition", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and len(faces) > 0:
            print(f"\n🔍 Testing recognition on {len(faces)} detected faces:")
            
            for i, (x, y, w, h) in enumerate(faces):
                print(f"\n--- Face {i+1} ({w}x{h}) ---")
                face_roi = frame[y:y+h, x:x+w]
                
                if w > 50 and h > 50:
                    name, confidence = recognize_face_debug(face_roi, known_faces, session, input_name)
                    print(f"Result: {name} ({confidence:.4f})")
                else:
                    print(f"❌ Face too small: {w}x{h}")
        
        elif key == ord('t') and len(faces) > 0:
            print(f"\n🎯 Testing different thresholds:")
            face_roi = frame[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
            
            for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
                print(f"\n--- Threshold {threshold} ---")
                name, confidence = recognize_face_debug(face_roi, known_faces, session, input_name, threshold)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
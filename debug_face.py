#!/usr/bin/env python3
"""
Face Detection Debug Tool
Test face detection and recognition
"""

import cv2
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_face_detection():
    """Test basic face detection"""
    print("🔍 FACE DETECTION DEBUG")
    print("=" * 50)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ Error: Could not load face cascade")
        return
    
    print("✅ Camera and face cascade loaded")
    print("📹 Press 'q' to quit, 's' to save current frame")
    print("🔍 Look at the camera and move around to test detection")
    print()
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            continue
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance for better detection
        gray = cv2.equalizeHist(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect faces with multiple parameters
        faces1 = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(50, 50),
            maxSize=(600, 600),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces2 = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            maxSize=(500, 500)
        )
        
        # Combine results
        all_faces = list(faces1) + list(faces2)
        
        # Remove duplicates
        filtered_faces = []
        for face in all_faces:
            x, y, w, h = face
            is_duplicate = False
            for existing in filtered_faces:
                ex, ey, ew, eh = existing
                if (abs(x - ex) < 30 and abs(y - ey) < 30 and 
                    abs(w - ew) < 30 and abs(h - eh) < 30):
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_faces.append(face)
        
        faces = filtered_faces
        
        if len(faces) > 0:
            detection_count += 1
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{w}x{h}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show stats
        detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
        status = f"Faces: {len(faces)} | Frames: {frame_count} | Detection Rate: {detection_rate:.1f}%"
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Face Detection Debug', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"debug_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved frame as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Final Stats:")
    print(f"   Total frames: {frame_count}")
    print(f"   Frames with faces: {detection_count}")
    print(f"   Detection rate: {detection_rate:.1f}%")
    
    if detection_rate < 50:
        print("\n⚠️  Low detection rate. Try:")
        print("   - Better lighting")
        print("   - Face the camera directly")
        print("   - Move closer to camera")
        print("   - Remove glasses/hat if wearing")

def test_face_recognition():
    """Test face recognition with known faces"""
    print("\n🧠 FACE RECOGNITION DEBUG")
    print("=" * 50)
    
    try:
        from memory import get_all_faces
        known_faces = get_all_faces()
        print(f"✅ Loaded {len(known_faces)} known faces:")
        for name in known_faces.keys():
            print(f"   - {name}")
        
        if len(known_faces) == 0:
            print("⚠️  No known faces found. Add some faces first.")
        
    except Exception as e:
        print(f"❌ Error loading faces: {e}")

if __name__ == "__main__":
    try:
        test_face_detection()
        test_face_recognition()
    except KeyboardInterrupt:
        print("\n👋 Debug session ended")
    except Exception as e:
        print(f"\n❌ Error: {e}")
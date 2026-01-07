import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine
import json
import os
import time
import msvcrt

class FaceRecognition:
    def __init__(self, model_path="MobileFaceNet.onnx", threshold=0.6):
        self.threshold = threshold
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}
        self.face_db_path = "face_database.json"
        self.waiting_for_name = False
        self.unknown_face_img = None
        self.user_input = ""
        self.input_ready = False
        self.load_known_faces()
    
    def preprocess_face(self, face_img):
        """Preprocess face for MobileFaceNet"""
        face_resized = cv2.resize(face_img, (112, 112))
        face_normalized = (face_resized - 127.5) / 128.0
        return np.expand_dims(face_normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
    
    def get_embedding(self, face_img):
        """Get face embedding using MobileFaceNet"""
        preprocessed = self.preprocess_face(face_img)
        embedding = self.session.run(None, {self.input_name: preprocessed})[0]
        return embedding.flatten()
    
    def detect_faces(self, img):
        """Detect faces in image - filter for larger, closer faces"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # More precise detection
            minNeighbors=8,   # Reduce false positives
            minSize=(80, 80), # Only detect faces >= 80x80 pixels
            maxSize=(400, 400) # Ignore very large detections
        )
        return faces
    
    def add_face(self, name, face_img):
        """Add known face"""
        embedding = self.get_embedding(face_img)
        self.known_faces[name] = embedding
        return embedding
    
    def recognize_face(self, face_img):
        """Recognize face using cosine distance"""
        embedding = self.get_embedding(face_img)
        
        best_match = None
        min_distance = float('inf')
        
        for name, known_embedding in self.known_faces.items():
            distance = cosine(embedding, known_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        if min_distance < self.threshold:
            return best_match, 1 - min_distance
        return "Unknown", 0.0
    
    def load_known_faces(self):
        """Load known faces from database"""
        if os.path.exists(self.face_db_path):
            try:
                with open(self.face_db_path, 'r') as f:
                    data = json.load(f)
                    for name, embedding in data.items():
                        self.known_faces[name] = np.array(embedding)
                print(f"Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                print(f"Error loading face database: {e}")
    
    def save_face_to_db(self, name, embedding):
        """Save new face to database"""
        data = {}
        if os.path.exists(self.face_db_path):
            try:
                with open(self.face_db_path, 'r') as f:
                    data = json.load(f)
            except:
                pass
        
        data[name] = embedding.tolist()
        with open(self.face_db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def process_frame(self, frame):
        """Process video frame for face recognition"""
        faces = self.detect_faces(frame)
        results = []
        
        # Only process the largest face (closest person)
        if faces is not None and len(faces) > 0:
            # Sort by area (width * height) and take the largest
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Only process if face is reasonably large (close enough)
            if w > 100 and h > 100:
                face_roi = frame[y:y+h, x:x+w]
                name, confidence = self.recognize_face(face_roi)
                results.append((name, confidence, (x, y, w, h)))
                
                # Draw rectangle and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame, results
    
    def check_keyboard_input(self):
        """Non-blocking keyboard input check"""
        if msvcrt.kbhit():
            char = msvcrt.getch().decode('utf-8')
            if char == '\r':  # Enter key
                return True
            elif char == '\b':  # Backspace
                if self.user_input:
                    self.user_input = self.user_input[:-1]
                    print(f"\rEnter name: {self.user_input} ", end="", flush=True)
            else:
                self.user_input += char
                print(char, end="", flush=True)
        return False
    
    def process_recognition_result(self, results):
        """Process face recognition results"""
        if not results:
            return None
            
        for name, confidence, bbox in results:
            if name == "Unknown":
                if not self.waiting_for_name:
                    self.waiting_for_name = True
                    return f"Unknown person detected! Enter name: "
            else:
                return f"Recognized: {name} (confidence: {confidence:.2f})"
        return None
    
    def learn_new_face(self, name, face_img):
        """Learn a new face with given name"""
        embedding = self.add_face(name, face_img)
        self.save_face_to_db(name, embedding)
        self.waiting_for_name = False
        return f"Learned new face: {name}"
    
    def run(self):
        """Main loop for face recognition"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Simple Face Recognition System")
        print("- Green box = Known person")
        print("- Red box = Unknown person")
        print("- Press 'q' in video window to quit")
        print("- When unknown person detected, type their name and press Enter\n")
        
        last_recognition_time = 0
        recognition_cooldown = 5  # Longer cooldown to reduce noise
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame for face recognition
                processed_frame, results = self.process_frame(frame)
                
                # Handle unknown faces
                current_time = time.time()
                if results and (current_time - last_recognition_time) > recognition_cooldown:
                    for name, confidence, bbox in results:
                        if name == "Unknown" and not self.waiting_for_name:
                            self.waiting_for_name = True
                            x, y, w, h = bbox
                            self.unknown_face_img = frame[y:y+h, x:x+w]
                            print(f"\nUnknown person detected! Enter name: ", end="", flush=True)
                            last_recognition_time = current_time
                            break
                        elif name != "Unknown":
                            print(f"\nRecognized: {name} (confidence: {confidence:.2f})")
                            last_recognition_time = current_time
                
                # Check for keyboard input (non-blocking)
                if self.check_keyboard_input():
                    user_text = self.user_input.strip()
                    self.user_input = ""
                    self.input_ready = False
                    
                    if user_text.lower() in ['exit', 'quit']:
                        break
                    
                    if self.waiting_for_name and self.unknown_face_img is not None:
                        response = self.learn_new_face(user_text, self.unknown_face_img)
                        print(f"\nBuddy: {response}")
                        self.unknown_face_img = None
                    else:
                        buddy_response = ask_buddy(user_text)
                        try:
                            response_data = json.loads(buddy_response)
                            print(f"\nBuddy: {response_data.get('reply', buddy_response)}")
                        except:
                            print(f"\nBuddy: {buddy_response}")
                    
                    if not self.waiting_for_name:
                        print("You: ", end="", flush=True)
                
                # Display video
                cv2.imshow('Face Recognition', processed_frame)
                
                # Handle OpenCV window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

if __name__ == "__main__":
    # Check if model file exists
    model_path = "MobileFaceNet.onnx"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        print("Please ensure MobileFaceNet.onnx model file is in this directory")
        exit(1)
    
    buddy = FaceRecognition(model_path)
    buddy.run()
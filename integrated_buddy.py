import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine
import json
import os
import time
import msvcrt
import sys
import threading
import queue
from buddy_brain import ask_buddy
from memory import save_memory, get_memory, save_face, get_all_faces, update_face_name

class IntegratedBuddy:
    def __init__(self, model_path="face-recog/MobileFaceNet.onnx", threshold=0.25):
        try:
            self.threshold = threshold
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load face cascade classifier")
                
            self.known_faces = {}
            
            # Dedicated companion state
            self.active_user = None  # Currently engaged user
            self.session_started = False  # Has conversation started
            self.waiting_for_name = False
            self.unknown_face_img = None
            self.user_input = ""
            
            # AI processing
            self.ai_queue = queue.Queue()
            self.response_queue = queue.Queue()
            self.ai_busy = False
            self.current_faces = []
            
            # User switching
            self.user_switch_threshold = 10  # seconds before considering user switch
            self.last_face_time = 0
            
            # Frame processing
            self.current_frame = None
            self.current_frame_results = []
            
            self.load_known_faces()
            print("[INIT] System initialized successfully")
        except Exception as e:
            print(f"[INIT ERROR] Failed to initialize: {e}")
            raise
    
    def preprocess_face(self, face_img):
        face_resized = cv2.resize(face_img, (112, 112))
        face_normalized = (face_resized - 127.5) / 128.0
        return np.expand_dims(face_normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
    
    def get_embedding(self, face_img):
        preprocessed = self.preprocess_face(face_img)
        embedding = self.session.run(None, {self.input_name: preprocessed})[0]
        return embedding.flatten()
    
    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(80, 80),
            maxSize=(400, 400)
        )
        return faces
    
    def recognize_face(self, face_img):
        embedding = self.get_embedding(face_img)
        
        best_match = None
        min_distance = float('inf')
        
        for name, known_embedding in self.known_faces.items():
            distance = cosine(embedding, known_embedding)
            print(f"[DEBUG] Distance to {name}: {distance:.3f}")
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        print(f"[DEBUG] Best match: {best_match}, distance: {min_distance:.3f}, threshold: {self.threshold}")
        
        if min_distance < self.threshold:
            return best_match, 1 - min_distance
        return "Unknown", 0.0
    
    def load_known_faces(self):
        """Load known faces from database"""
        try:
            self.known_faces = get_all_faces()
            print("[SYNC] Loaded faces from database:", list(self.known_faces.keys()))
        except Exception as e:
            print(f"Error loading faces from database: {e}")
    
    def add_face(self, name, face_img):
        """Add new face with comprehensive error handling"""
        try:
            if face_img is None or face_img.size == 0:
                print(f"\n[ERROR] Invalid face image for {name}")
                return None
                
            embedding = self.get_embedding(face_img)
            self.known_faces[name] = embedding
            
            # Save to faces database
            save_face(name, embedding, 0.95)
            
            # Start dedicated session with new user
            self.active_user = name
            self.session_started = True
            save_memory("name", name, 0.95)
            
            return embedding
        except Exception as e:
            print(f"\n[FACE LEARNING ERROR] Failed to learn {name}: {e}")
            return None
    
    def process_frame(self, frame):
        """Process video frame with error handling"""
        try:
            faces = self.detect_faces(frame)
            results = []
            self.current_faces = []  # Reset current faces
            self.current_frame = frame  # Store for unknown face capture
            
            if faces is not None and len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                if w > 100 and h > 100:
                    try:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.recognize_face(face_roi)
                        results.append((name, confidence, (x, y, w, h)))
                        
                        # Store face info for JSON output
                        self.current_faces.append({
                            "name": name,
                            "confidence": round(confidence, 3),
                            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                        })
                        
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"\n[FACE PROCESSING ERROR] {e}")
            
            self.current_frame_results = results
            return frame, results
        except Exception as e:
            print(f"\n[FRAME PROCESSING ERROR] {e}")
            return frame, []
    
    def check_keyboard_input(self):
        """Safe keyboard input handling"""
        try:
            if msvcrt.kbhit():
                char = msvcrt.getch().decode('utf-8', errors='ignore')
                if char == '\r':  # Enter key
                    return True
                elif char == '\b':  # Backspace
                    if self.user_input:
                        self.user_input = self.user_input[:-1]
                        print(f"\r{self.active_user or 'You'}: {self.user_input} ", end="", flush=True)
                elif char.isprintable():  # Only add printable characters
                    self.user_input += char
                    print(char, end="", flush=True)
        except Exception as e:
            print(f"\n[INPUT ERROR] {e}")
        return False
    
    def handle_user_session(self, name, confidence):
        """Manage dedicated user sessions - one user at a time"""
        try:
            current_time = time.time()
            
            if name != "Unknown":
                # Known user detected
                if not self.session_started:
                    # Start new session with this user
                    self.active_user = name
                    self.session_started = True
                    save_memory("name", name, 0.95)
                    if self.send_to_ai(f"I can see {name} is here. Greet them warmly as your dedicated companion."):
                        print(f"\n[SESSION] Started dedicated session with {name}")
                    
                elif self.active_user != name:
                    # Different user detected - check if should switch
                    if (current_time - self.last_face_time) > self.user_switch_threshold:
                        # Switch to new user after threshold
                        old_user = self.active_user
                        self.active_user = name
                        save_memory("name", name, 0.95)
                        if self.send_to_ai(f"I see {name} is now here. Say goodbye to {old_user} and greet {name} as your new companion."):
                            print(f"\n[SESSION] Switched from {old_user} to {name}")
                    # else: ignore brief appearances of other users
                    
                self.last_face_time = current_time
                
            else:
                # Unknown user
                if not self.session_started or (current_time - self.last_face_time) > self.user_switch_threshold:
                    if not self.waiting_for_name:
                        self.waiting_for_name = True
                        # Store unknown face for learning
                        if hasattr(self, 'current_frame_results') and self.current_frame_results:
                            _, _, bbox = self.current_frame_results[0]
                            x, y, w, h = bbox
                            self.unknown_face_img = self.current_frame[y:y+h, x:x+w]
                        
                        if self.send_to_ai("I see someone new. Introduce yourself warmly and ask for their name to start our companion session."):
                            print(f"\n[SESSION] New unknown user detected")
        except Exception as e:
            print(f"\n[SESSION ERROR] {e}")
    
    def ai_worker(self):
        """Background thread for AI processing"""
        while True:
            try:
                request = self.ai_queue.get(timeout=1)
                if request is None:  # Shutdown signal
                    break
                
                response = ask_buddy(request)
                self.response_queue.put(response)
                self.ai_busy = False
            except queue.Empty:
                continue
            except Exception as e:
                self.response_queue.put(f"Error: {e}")
                self.ai_busy = False
    
    def send_to_ai(self, message):
        """Send message to AI without blocking - prevent duplicates"""
        if not self.ai_busy:
            self.ai_busy = True
            self.ai_queue.put(message)
            return True
        return False
    
    def get_ai_response(self):
        """Get AI response if available"""
        try:
            response = self.response_queue.get_nowait()
            # Decode HTML entities properly
            import html
            decoded_response = html.unescape(response)
            # Also handle common HTML entities manually
            decoded_response = decoded_response.replace('&quot;', '"').replace('&#39;', "'").replace('&amp;', '&')
            return decoded_response
        except queue.Empty:
            return None
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Start AI worker thread
        ai_thread = threading.Thread(target=self.ai_worker, daemon=True)
        ai_thread.start()
        
        print("Buddy - Dedicated Companion System")
        print("- One user at a time for focused interaction")
        print("- Buddy will engage with the main person in front")
        print("- Press 'q' in video window to quit")
        print("- Natural conversation as your personal companion\n")
        
        last_recognition_time = 0
        # Removed complex cooldown system for simpler dedicated companion mode
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, results = self.process_frame(frame)
                
                # Handle face recognition - dedicated companion mode
                current_time = time.time()
                if results:
                    for name, confidence, bbox in results:
                        self.handle_user_session(name, confidence)
                        break  # Only process the main detected face
                
                # Check for AI responses and display them
                ai_response = self.get_ai_response()
                if ai_response:
                    # Add face detection info to JSON response
                    try:
                        response_data = json.loads(ai_response)
                        # Add faces information
                        response_data["faces_detected"] = self.current_faces
                        
                        # Format and display enhanced JSON
                        enhanced_response = json.dumps(response_data, indent=2)
                        print(f"\nBuddy: {enhanced_response}")
                        
                        reply = response_data.get('reply', '')
                        
                        # If asking for name, prepare for input
                        if self.waiting_for_name and ("name" in reply.lower() or "who" in reply.lower()):
                            print("You: ", end="", flush=True)
                        elif self.active_user:
                            print(f"{self.active_user}: ", end="", flush=True)
                        else:
                            print("You: ", end="", flush=True)
                    except:
                        # Fallback for non-JSON responses - still add face info
                        fallback_response = {
                            "reply": ai_response,
                            "emotion": "neutral",
                            "intent": "unknown",
                            "faces_detected": self.current_faces
                        }
                        print(f"\nBuddy: {json.dumps(fallback_response, indent=2)}")
                        print("You: ", end="", flush=True)
                
                # Handle user input
                if self.check_keyboard_input():
                    user_text = self.user_input.strip()
                    self.user_input = ""
                    
                    if user_text.lower() in ['exit', 'quit']:
                        break
                    
                    if self.waiting_for_name and self.unknown_face_img is not None:
                        # Learning new face - validate name input
                        if len(user_text) > 1 and user_text.isalpha() and user_text.lower() not in ['hey', 'hi', 'hello', 'yes', 'no', 'ok', 'okay']:
                            result = self.add_face(user_text, self.unknown_face_img)
                            if result is not None:
                                self.waiting_for_name = False
                                self.unknown_face_img = None
                                # Send warm welcome for new companion
                                if self.send_to_ai(f"Perfect! I now know you're {user_text}. Welcome them warmly as your new dedicated companion."):
                                    print(f"\n[SESSION] Started new session with {user_text}")
                            else:
                                print(f"\nFailed to learn face. Please try again: ", end="", flush=True)
                        else:
                            # Invalid name, ask again
                            print(f"\nPlease enter a valid name (not '{user_text}'): ", end="", flush=True)
                    else:
                        # Regular conversation with active user
                        if self.active_user and any(phrase in user_text.lower() for phrase in ["call me", "my name is", "i'm", "change my name"]):
                            # Handle name changes for active user
                            import re
                            patterns = [
                                r"call me ([a-zA-Z]+)",
                                r"my name is ([a-zA-Z]+)",
                                r"i'?m ([a-zA-Z]+)",
                                r"change my name to ([a-zA-Z]+)"
                            ]
                            
                            new_name = None
                            for pattern in patterns:
                                match = re.search(pattern, user_text.lower())
                                if match:
                                    new_name = match.group(1).capitalize()
                                    break
                            
                            if new_name and new_name != self.active_user:
                                # Update face database and session
                                if update_face_name(self.active_user, new_name):
                                    if self.active_user in self.known_faces:
                                        self.known_faces[new_name] = self.known_faces.pop(self.active_user)
                                    
                                    old_name = self.active_user
                                    self.active_user = new_name
                                    save_memory("name", new_name, 0.95)
                                    
                                    self.send_to_ai(f"I've updated your name from {old_name} to {new_name}. Acknowledge this change as your dedicated companion.")
                                    print(f"\n[SESSION] Updated user name from {old_name} to {new_name}")
                                else:
                                    self.send_to_ai(user_text)
                            else:
                                self.send_to_ai(user_text)
                        else:
                            # Regular conversation
                            self.send_to_ai(user_text)
                
                cv2.imshow('Buddy - Face Recognition + AI', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Shutdown AI thread
            self.ai_queue.put(None)
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if model file exists
    model_path = "face-recog/MobileFaceNet.onnx"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        print("Please ensure MobileFaceNet.onnx is in the face-recog directory")
        exit(1)
    
    buddy = IntegratedBuddy(model_path)
    buddy.run()
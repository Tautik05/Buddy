import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine
import json
import os
import time
import msvcrt
import sys
from buddy_brain import ask_buddy
from memory import save_memory, get_memory, save_face, get_all_faces, update_face_name, get_all_memory

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
            
            # Session state
            self.active_user = None
            self.face_recognized = False
            self.unknown_face_img = None
            self.user_input = ""
            self.last_ai_call = 0
            self.ai_processing = False  # Prevent concurrent AI calls
            
            # Face recognition control
            self.last_recognition_time = 0
            self.recognition_interval = 30  # Increased to 30 seconds
            
            self.current_faces = []
            self.current_frame = None
            
            self.load_known_faces()
            
            # Initial face check and greeting
            try:
                memory_context = get_all_memory()
                print(f"[DEBUG] Memory context: {memory_context}")
                
                # Do immediate face recognition at startup
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        faces = self.detect_faces(frame)
                        if faces is not None and len(faces) > 0:
                            largest_face = max(faces, key=lambda f: f[2] * f[3])
                            x, y, w, h = largest_face
                            if w > 100 and h > 100:
                                face_roi = frame[y:y+h, x:x+w]
                                name, confidence = self.recognize_face(face_roi)
                                if name != "Unknown" and confidence > 0.6:
                                    self.active_user = name
                                    save_memory("name", name, 0.95)
                                    response = ask_buddy(f"I can see {name} is here. Greet them warmly.")
                                else:
                                    self.unknown_face_img = face_roi
                                    self.face_recognized = True
                                    response = ask_buddy("I see someone new. Introduce yourself and ask their name.")
                            else:
                                response = ask_buddy("System starting. Greet the user briefly.")
                        else:
                            response = ask_buddy("System starting. Greet the user briefly.")
                    else:
                        response = ask_buddy("System starting. Greet the user briefly.")
                    cap.release()
                else:
                    response = ask_buddy("System starting. Greet the user briefly.")
                    
                self.display_response(response)
            except Exception as e:
                print(f"[INIT WARNING] Initial greeting failed: {e}")
                print("System ready - type to chat")
                
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
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        if min_distance < self.threshold:
            return best_match, 1 - min_distance
        return "Unknown", 0.0
    
    def load_known_faces(self):
        """Load known faces from database"""
        try:
            self.known_faces = get_all_faces()
            # Clean face names from HTML entities
            import html
            cleaned_faces = {}
            for name, embedding in self.known_faces.items():
                clean_name = html.unescape(name)
                cleaned_faces[clean_name] = embedding
            self.known_faces = cleaned_faces
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
            save_memory("name", name, 0.95)
            
            return embedding
        except Exception as e:
            print(f"\n[FACE LEARNING ERROR] Failed to learn {name}: {e}")
            return None
    
    def display_response(self, response):
        """Display AI response with JSON and clean text"""
        try:
            # First decode HTML entities in the raw response
            import html
            decoded_response = html.unescape(response)
            
            response_data = json.loads(decoded_response)
            
            # Show full JSON with decoded entities
            print(f"\nBuddy JSON: {json.dumps(response_data, indent=2)}")
            
            # Then show clean reply
            reply = response_data.get('reply', '')
            print(f"Buddy: {reply}")
        except:
            # Fallback for non-JSON responses
            import html
            clean_response = html.unescape(response)
            print(f"\nBuddy: {clean_response}")
        
        # Show input prompt
        if self.active_user:
            print(f"{self.active_user}: ", end="", flush=True)
        else:
            print("You: ", end="", flush=True)
    
    def process_frame(self, frame):
        """Process frame at intervals for face recognition with error handling"""
        try:
            current_time = time.time()
            if (current_time - self.last_recognition_time) > self.recognition_interval and not self.ai_processing:
                faces = self.detect_faces(frame)
                if faces is not None and len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    if w > 100 and h > 100:
                        try:
                            face_roi = frame[y:y+h, x:x+w]
                            name, confidence = self.recognize_face(face_roi)
                            self.handle_face_recognition(name, confidence, face_roi)
                            self.last_recognition_time = current_time
                        except Exception as e:
                            print(f"\n[FACE ERROR] {e}")
                            self.last_recognition_time = current_time  # Skip this cycle
            
            # Show status on frame
            status_text = f"Chatting with: {self.active_user}" if self.active_user else "Ready to chat"
            color = (0, 255, 0) if self.active_user else (0, 255, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if self.ai_processing:
                cv2.putText(frame, "AI Processing...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            return frame
        except Exception as e:
            print(f"\n[FRAME ERROR] {e}")
            return frame
    
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
    
    def handle_face_recognition(self, name, confidence, face_img):
        """Handle face recognition with robust error handling"""
        try:
            current_time = time.time()
            if (current_time - self.last_ai_call) < 3 or self.ai_processing:  # Increased cooldown
                return
                
            self.ai_processing = True
            
            if name != "Unknown" and confidence > 0.6:
                if not self.active_user or self.active_user != name:
                    self.active_user = name
                    save_memory("name", name, 0.95)
                    try:
                        response = ask_buddy(f"I can see {name} is here. Greet them briefly.")
                        self.display_response(response)
                        print(f"\n[RECOGNITION] Recognized {name}")
                    except Exception as e:
                        print(f"\n[AI ERROR] Failed to process recognition: {e}")
                    self.last_ai_call = current_time
            else:
                if not self.face_recognized:
                    self.face_recognized = True
                    self.unknown_face_img = face_img
                    try:
                        response = ask_buddy("I see someone new. Introduce yourself and ask their name.")
                        self.display_response(response)
                        print(f"\n[RECOGNITION] Unknown person detected")
                    except Exception as e:
                        print(f"\n[AI ERROR] Failed to process unknown user: {e}")
                    self.last_ai_call = current_time
                    
        except Exception as e:
            print(f"\n[RECOGNITION ERROR] {e}")
        finally:
            self.ai_processing = False
    
    def run(self):
        print("Buddy - AI Companion System")
        print("- Robust operation with error handling")
        print("- Face recognition every 30 seconds")
        print("- Press 'q' in video window or type 'exit' to stop\n")
        
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return
            
            # Set camera properties for stability
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print("\n[WARNING] Failed to read frame, retrying...")
                        time.sleep(0.1)
                        continue
                    
                    # Process frame (face recognition at intervals)
                    processed_frame = self.process_frame(frame)
                    
                    # Handle user input (non-blocking)
                    if self.check_keyboard_input():
                        user_text = self.user_input.strip()
                        self.user_input = ""
                        
                        if user_text.lower() in ['exit', 'quit']:
                            break
                        
                        if user_text:  # Only process non-empty input
                            try:
                                # Check for name learning
                                if not self.active_user and self.unknown_face_img is not None:
                                    import re
                                    name_patterns = [
                                        r"i'?m\s+([a-zA-Z]+)",
                                        r"my name is\s+([a-zA-Z]+)",
                                        r"call me\s+([a-zA-Z]+)",
                                        r"i am\s+([a-zA-Z]+)"
                                    ]
                                    
                                    learned_name = None
                                    for pattern in name_patterns:
                                        match = re.search(pattern, user_text.lower())
                                        if match:
                                            potential_name = match.group(1).capitalize()
                                            if potential_name.lower() not in ['not', 'am', 'is', 'here', 'good', 'fine']:
                                                learned_name = potential_name
                                                break
                                    
                                    if learned_name:
                                        result = self.add_face(learned_name, self.unknown_face_img)
                                        if result is not None:
                                            self.unknown_face_img = None
                                            print(f"\n[LEARNED] New user: {learned_name}")
                                
                                # Send to AI with error handling
                                if not self.ai_processing:
                                    self.ai_processing = True
                                    try:
                                        response = ask_buddy(user_text)
                                        self.display_response(response)
                                    except Exception as e:
                                        print(f"\n[AI ERROR] {e}")
                                        print("You: ", end="", flush=True)
                                    finally:
                                        self.ai_processing = False
                                else:
                                    print("\n[BUSY] AI is processing, please wait...")
                                    print("You: ", end="", flush=True)
                                    
                            except Exception as e:
                                print(f"\n[INPUT ERROR] {e}")
                                print("You: ", end="", flush=True)
                    
                    # Show video with error handling
                    try:
                        cv2.imshow('Buddy - AI Companion', processed_frame)
                    except Exception as e:
                        print(f"\n[DISPLAY ERROR] {e}")
                    
                    # Check for 'q' key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    
                    time.sleep(0.01)  # Small delay for stability
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\n[LOOP ERROR] {e}")
                    time.sleep(0.1)  # Brief pause before continuing
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
        except Exception as e:
            print(f"\n[SYSTEM ERROR] {e}")
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            print("\n[CLEANUP] System shutdown complete")

if __name__ == "__main__":
    # Check if model file exists
    model_path = "face-recog/MobileFaceNet.onnx"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        print("Please ensure MobileFaceNet.onnx is in the face-recog directory")
        exit(1)
    
    buddy = IntegratedBuddy(model_path)
    buddy.run()
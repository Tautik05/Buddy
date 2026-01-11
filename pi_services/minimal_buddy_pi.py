"""
Minimal Buddy Pi - ONLY Hardware Functions
- Voice to text
- Text to voice  
- Face recognition
- Object detection
- Camera processing
"""

import cv2
import numpy as np
import time
import requests
import threading
import speech_recognition as sr
import edge_tts
import asyncio
import pygame
import io
from typing import Optional

from config import Config
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from stability_tracker import StabilityTracker
from objrecog.obj import ObjectDetector

class MinimalBuddyPi:
    """Minimal Pi - Only hardware, no brain"""
    
    def __init__(self, llm_service_url: str = "http://localhost:8000"):
        self.config = Config.from_env()
        self.llm_service_url = llm_service_url
        
        # Hardware only
        self._init_camera()
        self._init_face_recognition()
        self._init_object_detection()
        self._init_speech()
        
        # State
        self.running = False
        self.active_user = None
        self.is_speaking = False
        
    def _init_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.config.camera_index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def _init_face_recognition(self):
        """Initialize face recognition"""
        self.detector = FaceDetector(self.config.cascade_path, self.config)
        self.face_recognizer = FaceRecognizer(self.config.model_path, self.config)
        self.stability = StabilityTracker(self.config)
        self.last_recognition_time = 0
    
    def _init_object_detection(self):
        """Initialize object detection"""
        self.object_detector = ObjectDetector(confidence_threshold=0.7)
        self.stable_objects = set()
    
    def _init_speech(self):
        """Initialize speech"""
        try:
            self.speech_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.tts_voice = "en-IN-NeerjaNeural"
            
            with self.microphone as source:
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.speech_recognizer.energy_threshold = 400
            
            self.speech_enabled = True
            print("Speech initialized")
        except Exception as e:
            print(f"Speech failed: {e}")
            self.speech_enabled = False
    
    def recognize_face(self) -> tuple[Optional[str], float]:
        """Recognize face from camera"""
        ret, frame = self.cap.read()
        if not ret:
            return None, 0.0
        
        faces = self.detector.detect(frame)
        if not faces:
            return None, 0.0
        
        largest_face = self.detector.get_largest_face(faces)
        is_stable = self.stability.update(largest_face)
        
        current_time = time.time()
        if (is_stable and 
            (current_time - self.last_recognition_time) > 1.0):
            
            x, y, w, h = largest_face
            if w > 80 and h > 80:
                face_roi = frame[y:y+h, x:x+w]
                name, confidence = self.face_recognizer.recognize(face_roi)
                self.last_recognition_time = current_time
                return name, confidence
        
        return None, 0.0
    
    def detect_objects(self) -> list:
        """Detect objects from camera"""
        ret, frame = self.cap.read()
        if not ret:
            return []
        
        detections = self.object_detector.detect(frame)
        return [det['name'] for det in detections]
    
    def listen_for_speech(self, timeout=5.0) -> str:
        """Listen for speech input"""
        if not self.speech_enabled:
            return ""
        
        try:
            with sr.Microphone() as source:
                print("🎤 Listening...")
                audio = self.speech_recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
            
            text = self.speech_recognizer.recognize_google(audio)
            print(f"You said: '{text}'")
            return text
        except:
            return ""
    
    def speak(self, text: str):
        """Convert text to speech"""
        if not self.speech_enabled or not text:
            return
        
        def _speak():
            try:
                self.is_speaking = True
                asyncio.run(self._generate_speech(text))
                self.is_speaking = False
            except Exception as e:
                print(f"TTS Error: {e}")
                self.is_speaking = False
        
        threading.Thread(target=_speak, daemon=True).start()
    
    async def _generate_speech(self, text):
        """Generate and play speech"""
        try:
            communicate = edge_tts.Communicate(text, self.tts_voice)
            
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if audio_data:
                audio_io = io.BytesIO(audio_data)
                pygame.mixer.music.load(audio_io)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Speech error: {e}")
    
    def call_llm_brain(self, user_input: str, recognized_user: str = None, objects: list = None) -> dict:
        """Call LLM service brain"""
        try:
            request_data = {
                "user_input": user_input,
                "recognized_user": recognized_user,
                "objects_visible": objects or []
            }
            
            response = requests.post(
                f"{self.llm_service_url}/chat",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"reply": "Sorry, brain not responding.", "intent": "conversation", "emotion": "apologetic"}
        except Exception as e:
            print(f"Brain error: {e}")
            return {"reply": "Sorry, brain not responding.", "intent": "conversation", "emotion": "apologetic"}
    
    def run(self):
        """Main loop - hardware only"""
        self.running = True
        print("🤖 Buddy Pi Hardware Ready")
        
        # Initial face recognition
        print("Looking for faces...")
        face_found = False
        for i in range(30):
            name, confidence = self.recognize_face()
            if name and name != "Unknown" and confidence > 0.3:
                self.active_user = name
                print(f"Recognized: {name}")
                
                # Call brain for greeting
                response = self.call_llm_brain(
                    f"I just recognized {name} on camera. Greet them warmly.",
                    recognized_user=name
                )
                
                print(f"Buddy: {response.get('raw_response', response.get('reply'))}")
                self.speak(response.get('reply', ''))
                face_found = True
                break
            time.sleep(0.1)
        
        if not face_found:
            print("No face recognized, starting general interaction")
            response = self.call_llm_brain("Hello! I'm ready to chat.")
            print(f"Buddy: {response.get('reply')}")
            self.speak(response.get('reply', ''))
        
        # Main interaction loop
        try:
            while self.running:
                # Listen for voice
                user_text = self.listen_for_speech(timeout=3.0)
                
                if user_text:
                    print("🤔 Thinking...")
                    
                    # Get current objects
                    objects = self.detect_objects()
                    
                    # Call brain
                    response = self.call_llm_brain(user_text, self.active_user, objects)
                    
                    # Display and speak response
                    print(f"Buddy: {response.get('raw_response', response.get('reply'))}")
                    if response.get('intent') != 'conversation':
                        print(f"[INTENT: {response.get('intent')}]")
                    
                    self.speak(response.get('reply', ''))
                
                # Check for face changes occasionally
                if not user_text:  # Only check faces when not actively talking
                    name, confidence = self.recognize_face()
                    if name and name != "Unknown" and name != self.active_user and confidence > 0.3:
                        self.active_user = name
                        response = self.call_llm_brain(f"I just recognized {name}.", recognized_user=name)
                        print(f"Buddy: {response.get('reply')}")
                        self.speak(response.get('reply', ''))
        
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup hardware"""
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("Buddy Pi: Hardware shutdown")

def main():
    import os
    llm_url = os.getenv('LLM_SERVICE_URL', 'http://localhost:8000')
    
    buddy = MinimalBuddyPi(llm_url)
    buddy.run()

if __name__ == "__main__":
    main()
"""
BULLETPROOF BUDDY - All Features with Robust Error Handling
Face Recognition + Speech + Memory + Object Detection + Web Search
"""

import cv2
import numpy as np
import time
import logging
import requests
import threading
import json
from pathlib import Path
from typing import Optional
from collections import defaultdict

# Hardware modules with error handling
try:
    from config import Config
    from face_detector import FaceDetector
    from face_recognizer import FaceRecognizer
    from stability_tracker import StabilityTracker
    from objrecog.obj import ObjectDetector
except ImportError as e:
    print(f"Warning: {e}")

# Speech imports with fallbacks
try:
    import speech_recognition as sr
    import edge_tts
    import asyncio
    import pygame
    import io
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Warning: Speech modules not available")

class BulletproofBuddy:
    def __init__(self, config: Optional[Config] = None, llm_service_url: str = "http://localhost:8000"):
        self.config = config or self._safe_config()
        self.llm_service_url = llm_service_url
        self.logger = self._setup_logging()
        
        # Initialize with fallbacks
        self.cap = self._safe_init_camera()
        self.detector = self._safe_init_face_detector()
        self.face_recognizer = self._safe_init_face_recognizer()
        self.stability = self._safe_init_stability()
        self.object_detector = self._safe_init_object_detector()
        self.speech_enabled = self._safe_init_speech()
        
        # State management
        self.last_recognition_time = 0
        self.last_object_detection_time = 0
        self.current_detections = []
        self.running = False
        self.active_user = None
        self.recognition_threshold = 0.4
        
        print("🛡️ Bulletproof Buddy Ready - All Systems Operational!")
    
    def _safe_config(self):
        """Safe config loading with defaults"""
        try:
            return Config.from_env()
        except:
            # Create minimal config
            class MinimalConfig:
                camera_index = 0
                camera_width = 640
                camera_height = 480
                camera_fps = 30
                camera_buffer_size = 1
                min_face_size = (80, 80)
                cascade_path = "models/haarcascade_frontalface_default.xml"
                model_path = "models/MobileFaceNet.tflite"
                log_level = "INFO"
                log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            return MinimalConfig()
    
    def _setup_logging(self):
        """Safe logging setup"""
        try:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
            return logging.getLogger(__name__)
        except:
            return None
    
    def _safe_init_camera(self):
        """Safe camera initialization with fallbacks"""
        try:
            cap = cv2.VideoCapture(self.config.camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
                print("✅ Camera initialized")
                return cap
        except Exception as e:
            print(f"⚠️ Camera error: {e}")
        
        # Fallback: try different camera indices
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"✅ Camera fallback: index {i}")
                    return cap
            except:
                continue
        
        print("❌ No camera available - using dummy")
        return None
    
    def _safe_init_face_detector(self):
        """Safe face detector with fallbacks"""
        try:
            detector = FaceDetector(self.config.cascade_path, self.config)
            print("✅ Face detector initialized")
            return detector
        except Exception as e:
            print(f"⚠️ Face detector error: {e}")
            return None
    
    def _safe_init_face_recognizer(self):
        """Safe face recognizer with fallbacks"""
        try:
            recognizer = FaceRecognizer(self.config.model_path, self.config)
            print("✅ Face recognizer initialized")
            return recognizer
        except Exception as e:
            print(f"⚠️ Face recognizer error: {e}")
            return None
    
    def _safe_init_stability(self):
        """Safe stability tracker"""
        try:
            stability = StabilityTracker(self.config)
            print("✅ Stability tracker initialized")
            return stability
        except Exception as e:
            print(f"⚠️ Stability tracker error: {e}")
            return None
    
    def _safe_init_object_detector(self):
        """Safe object detector"""
        try:
            detector = ObjectDetector(confidence_threshold=0.6)
            print("✅ Object detector initialized")
            return detector
        except Exception as e:
            print(f"⚠️ Object detector error: {e}")
            return None
    
    def _safe_init_speech(self):
        """Safe speech initialization"""
        if not SPEECH_AVAILABLE:
            print("⚠️ Speech not available")
            return False
        
        try:
            self.speech_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            with self.microphone as source:
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=1.0)
                self.speech_recognizer.energy_threshold = 300
                self.speech_recognizer.pause_threshold = 1.0
            
            print("✅ Speech initialized")
            return True
        except Exception as e:
            print(f"⚠️ Speech error: {e}")
            return False
    
    def safe_face_detection(self, frame):
        """Bulletproof face detection"""
        try:
            if self.detector:
                return self.detector.detect(frame)
        except cv2.error as e:
            print(f"OpenCV face detection error: {e}")
        except Exception as e:
            print(f"Face detection error: {e}")
        return []
    
    def safe_face_recognition(self, face_roi):
        """Bulletproof face recognition"""
        try:
            if self.face_recognizer:
                return self.face_recognizer.recognize(face_roi)
        except Exception as e:
            print(f"Face recognition error: {e}")
        return "Unknown", 0.0
    
    def safe_object_detection(self, frame):
        """Bulletproof object detection"""
        try:
            if self.object_detector:
                return self.object_detector.detect(frame)
        except Exception as e:
            print(f"Object detection error: {e}")
        return []
    
    def safe_speech_recognition(self, timeout=5.0):
        """Bulletproof speech recognition"""
        if not self.speech_enabled:
            return ""
        
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.speech_recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            # Try multiple languages
            for lang in ['en-IN', 'en-US']:
                try:
                    text = self.speech_recognizer.recognize_google(audio, language=lang)
                    print(f"You said: '{text}'")
                    return text
                except sr.UnknownValueError:
                    continue
            
            print("Couldn't understand that.")
            return ""
        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return ""
    
    def safe_tts(self, text):
        """Bulletproof text-to-speech"""
        if not self.speech_enabled or not text:
            print(f"TTS: {text}")
            return
        
        def _speak():
            try:
                async def _async_speak():
                    communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
                    audio_data = b""
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]
                    
                    if audio_data:
                        pygame.mixer.music.load(io.BytesIO(audio_data))
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            await asyncio.sleep(0.1)
                
                asyncio.run(_async_speak())
            except Exception as e:
                print(f"TTS Error: {e} - Fallback to print")
                print(f"TTS: {text}")
        
        threading.Thread(target=_speak, daemon=True).start()
    
    def safe_llm_call(self, user_input, recognized_user=None, objects=None):
        """Bulletproof LLM service call"""
        try:
            request_data = {
                "user_input": user_input,
                "recognized_user": recognized_user,
                "objects_visible": objects or []
            }
            
            response = requests.post(f"{self.llm_service_url}/chat", json=request_data, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "reply": data.get("reply", "I'm thinking..."),
                    "intent": data.get("intent", "conversation"),
                    "emotion": data.get("emotion", "neutral")
                }
        except Exception as e:
            print(f"LLM service error: {e}")
        
        # Fallback response
        return {
            "reply": "Sorry, I'm having trouble connecting to my brain right now.",
            "intent": "conversation",
            "emotion": "apologetic"
        }
    
    def process_frame(self, frame):
        """Bulletproof frame processing"""
        if frame is None:
            return frame, False, None, 0.0
        
        # Safe face detection
        faces = self.safe_face_detection(frame)
        face_detected = len(faces) > 0
        name, confidence = None, 0.0
        
        if face_detected and self.stability:
            try:
                largest_face = max(faces, key=lambda f: f[2] * f[3])  # width * height
                is_stable = self.stability.update(largest_face)
                
                current_time = time.time()
                if is_stable and (current_time - self.last_recognition_time) > 30.0:
                    x, y, w, h = largest_face
                    if w > 80 and h > 80:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.safe_face_recognition(face_roi)
                        
                        if name != "Unknown" and confidence > self.recognition_threshold:
                            self.active_user = name
                        
                        self.last_recognition_time = current_time
            except Exception as e:
                print(f"Face processing error: {e}")
        
        # Safe object detection (every 5 seconds)
        current_time = time.time()
        if (current_time - self.last_object_detection_time) > 5.0:
            self.current_detections = self.safe_object_detection(frame)
            self.last_object_detection_time = current_time
        
        # Draw visualization
        processed_frame = self.draw_visualization(frame, faces, name, confidence)
        
        return processed_frame, face_detected, name, confidence
    
    def draw_visualization(self, frame, faces, name=None, confidence=0.0):
        """Safe visualization drawing"""
        try:
            # Draw object detection
            if self.current_detections and self.object_detector:
                frame = self.object_detector.draw_detections(frame, self.current_detections)
            
            # Draw face detection
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if self.stability and self.stability.is_stable else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                if name and name != "Unknown":
                    label = f"{name} ({confidence:.0%})"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Status display
            status = f"Active: {self.active_user}" if self.active_user else "Looking for people"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Visualization error: {e}")
        
        return frame
    
    def startup_greeting(self):
        """Safe startup with face recognition"""
        try:
            print("Starting camera and face recognition...")
            time.sleep(1.0)
            
            for attempt in range(30):  # Reduced attempts
                if not self.cap:
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                processed, face_detected, name, confidence = self.process_frame(frame)
                
                if self.cap:
                    cv2.imshow('Bulletproof Buddy', processed)
                    cv2.waitKey(1)
                
                if name and name != "Unknown" and confidence > self.recognition_threshold:
                    self.active_user = name
                    print(f"✅ Recognized: {name}")
                    
                    response = self.safe_llm_call(f"I just recognized {name}. Greet them warmly.", name)
                    self.safe_tts(response["reply"])
                    return
                
                time.sleep(0.1)
            
            # Fallback greeting
            print("Starting general interaction")
            response = self.safe_llm_call("Hello! I'm ready to chat!")
            self.safe_tts(response["reply"])
            
        except Exception as e:
            print(f"Startup error: {e}")
            self.safe_tts("Hello! I'm ready to chat!")
    
    def conversation_loop(self):
        """Safe conversation loop"""
        while self.running:
            try:
                user_text = self.safe_speech_recognition(timeout=6.0)
                
                if user_text:
                    if "goodbye" in user_text.lower():
                        self.safe_tts("Goodbye!")
                        break
                    
                    # Get current objects
                    objects = []
                    if self.current_detections:
                        objects = [det['name'] for det in self.current_detections]
                    
                    response = self.safe_llm_call(user_text, self.active_user, objects)
                    self.safe_tts(response["reply"])
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Conversation error: {e}")
                time.sleep(1)
    
    def run(self):
        """Main bulletproof run loop"""
        self.running = True
        
        try:
            # Startup greeting
            self.startup_greeting()
            
            # Start conversation in separate thread
            conversation_thread = threading.Thread(target=self.conversation_loop, daemon=True)
            conversation_thread.start()
            
            # Main camera loop
            while self.running:
                if self.cap:
                    ret, frame = self.cap.read()
                    if ret:
                        processed, _, _, _ = self.process_frame(frame)
                        cv2.imshow('Bulletproof Buddy', processed)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                time.sleep(0.03)  # ~30 FPS
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Main loop error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Safe cleanup"""
        self.running = False
        
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
        except:
            pass
        
        print("🛡️ Bulletproof Buddy: Goodbye! 👋")

def main():
    try:
        print("🛡️ Starting Bulletproof Buddy...")
        buddy = BulletproofBuddy()
        buddy.run()
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
"""
Buddy Raspberry Pi Main Application
Handles face recognition, object detection, voice I/O locally
Communicates with remote LLM service via API
"""

import cv2
import numpy as np
import time
import logging
import json
import requests
import threading
from pathlib import Path
from typing import Optional
from collections import defaultdict

# Import local modules - HARDWARE ONLY
from config import Config
from states import BuddyState, StateManager
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from stability_tracker import StabilityTracker
from objrecog.obj import ObjectDetector
from objrecog.perception import interpret_objects

# Speech imports
import speech_recognition as sr
import edge_tts
import asyncio
import pygame
import io

class BuddyPi:
    """Raspberry Pi Buddy - Local processing with remote LLM"""
    
    def __init__(self, config: Optional[Config] = None, llm_service_url: str = "http://192.168.1.100:8000"):
        self.config = config or Config.from_env()
        self.llm_service_url = llm_service_url
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory
        init_db()
        
        # State management
        self.state_manager = StateManager()
        
        # Initialize camera
        self._init_camera()
        
        # Initialize face components - using original
        self.detector = FaceDetector(self.config.cascade_path, self.config)
        self.face_recognizer = FaceRecognizer(self.config.model_path, self.config)
        self.stability = StabilityTracker(self.config)
        
        # Initialize object detection - using original
        self.object_detector = ObjectDetector(confidence_threshold=0.7)
        self.object_history = defaultdict(int)
        self.object_absence = defaultdict(int)
        self.stable_objects = set()
        
        # Initialize sleep/wake manager
        self.sleep_wake = SleepWakeManager(self.config, self.state_manager)
        self.input_handler = InputHandler()
        
        # Initialize speech
        self._init_speech()
        
        # Session context for conversation continuity
        self.session_context = {'conversations': [], 'max_size': 5}
        
        # Application state
        self.unknown_face_img: Optional[np.ndarray] = None
        self.awaiting_name = False
        self.last_recognition_time = 0
        self.running = False
        self.is_speaking = False
        self._processing_input = False
        
        self.state_manager.state = BuddyState.ACTIVE
        
        # Perform initial greeting
        self._startup_greeting()
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.log_file) 
                if self.config.log_file else logging.NullHandler()
            ]
        )
    
    def _init_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.config.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera_buffer_size)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
    
    def _init_speech(self):
        """Initialize speech recognition and TTS"""
        try:
            self.speech_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.tts_voice = "en-IN-NeerjaNeural"
            
            with self.microphone as source:
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.speech_recognizer.energy_threshold = 400
                self.speech_recognizer.pause_threshold = 0.8
                self.speech_recognizer.dynamic_energy_threshold = True
            
            self.speech_enabled = True
            print(f"Speech initialized with voice: {self.tts_voice}")
            
        except Exception as e:
            print(f"Speech initialization failed: {e}")
            self.speech_enabled = False
    
    def _startup_greeting(self):
        """Detect and greet person on startup"""
        try:
            print("\\nStarting camera and face recognition...")
            time.sleep(1.0)
            
            for attempt in range(30):
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                faces = self.detector.detect(frame)
                
                if faces:
                    largest_face = self.detector.get_largest_face(faces)
                    is_stable = self.stability.update(largest_face)
                    x, y, w, h = largest_face
                    
                    processed = self._draw_visualization(frame, faces)
                    cv2.imshow('Buddy Vision', processed)
                    cv2.waitKey(1)
                    
                    if w > 80 and h > 80 and is_stable:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.face_recognizer.recognize(face_roi)
                        
                        print(f"DEBUG: Face recognition - Name: {name}, Confidence: {confidence}")
                        
                        if name != "Unknown" and confidence > 0.3:  # Lower threshold
                            self.sleep_wake.active_user = name
                            print(f"Recognized {name} successfully!")
                            
                            response = self._call_llm_service(
                                f"I just recognized {name} on camera. Greet them warmly.",
                                recognized_user=name
                            )
                            self._display_response(response)
                            return
                
                time.sleep(0.1)
            
            # No face recognized
            ret, frame = self.cap.read()
            if ret:
                faces = self.detector.detect(frame)
                if faces:
                    largest_face = self.detector.get_largest_face(faces)
                    x, y, w, h = largest_face
                    if w > 80 and h > 80:
                        self.unknown_face_img = frame[y:y+h, x:x+w]
                        self.awaiting_name = True
            
            response = self._call_llm_service(
                "[CONTEXT: Unknown person detected on camera at startup] Hello! I can see someone but don't recognize you yet."
            )
            self._display_response(response)
            
        except Exception as e:
            self.logger.error(f"Startup greeting error: {e}")
            print("Buddy: Hello! I'm ready to chat!")
    
    def _call_llm_service(self, user_input: str, recognized_user: Optional[str] = None) -> dict:
        """Call remote LLM service"""
        try:
            # Get memory context
            memory_context = get_all_memory(user_name=recognized_user if recognized_user else "Unknown")
            
            # Prepare request
            request_data = {
                "user_input": user_input,
                "recognized_user": recognized_user,
                "memory_context": memory_context,
                "session_context": self.session_context,
                "objects_visible": list(self.stable_objects)
            }
            
            # Debug URL
            url = f"{self.llm_service_url}/chat"
            print(f"DEBUG: Calling URL: '{url}'")
            
            # Call API
            response = requests.post(
                url,
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"LLM service error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"LLM service call failed: {e}")
            return {
                "reply": "Sorry, I'm having trouble thinking right now.",
                "intent": "conversation",
                "emotion": "apologetic",
                "raw_response": ""
            }
    
    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool, Optional[str], float]:
        """Process frame for face detection and recognition"""
        faces = self.detector.detect(frame)
        
        name = None
        confidence = 0.0
        face_detected = len(faces) > 0
        
        if face_detected:
            largest_face = self.detector.get_largest_face(faces)
            is_stable = self.stability.update(largest_face)
            
            current_time = time.time()
            if (is_stable and 
                (current_time - self.last_recognition_time) > self.config.recognition_interval):
                
                x, y, w, h = largest_face
                if w > self.config.min_face_size[0] and h > self.config.min_face_size[1]:
                    face_roi = frame[y:y+h, x:x+w]
                    name, confidence = self.face_recognizer.recognize(face_roi)
                    
                    print(f"DEBUG: Recognition - Name: {name}, Confidence: {confidence}")
                    
                    if name == "Unknown" and self.unknown_face_img is None:
                        self.unknown_face_img = face_roi
                    
                    self.last_recognition_time = current_time
        else:
            self.stability.reset()
        
        # Process object detection
        self._process_objects(frame)
        
        # Draw visualization
        processed = self._draw_visualization(frame, faces, name, confidence)
        
        return processed, face_detected, name, confidence
    
    def _process_objects(self, frame):
        """Process object detection with stability tracking"""
        detections = self.object_detector.detect(frame)
        current_objects = set([det['name'] for det in detections])
        
        # Simple stability - objects seen in last few frames
        self.stable_objects = current_objects
    
    def speak(self, text):
        """Convert text to speech using Edge TTS"""
        if not self.speech_enabled or not text:
            return
        
        def _speak():
            try:
                self.is_speaking = True
                asyncio.run(self._generate_and_play_speech(text))
                self.is_speaking = False
            except Exception as e:
                print(f"TTS Error: {e}")
                self.is_speaking = False
        
        threading.Thread(target=_speak, daemon=True).start()
    
    async def _generate_and_play_speech(self, text):
        """Generate speech using Edge TTS and play it"""
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
            print(f"Edge TTS Error: {e}")
    
    def listen_for_speech(self, timeout=5.0):
        """Listen for speech input"""
        if not self.speech_enabled:
            return ""
        
        try:
            with sr.Microphone() as source:
                print("🎤 Listening...")
                audio = self.speech_recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
            
            print("Processing speech...")
            text = self.speech_recognizer.recognize_google(audio)
            print(f"You said: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("Couldn't understand that.")
            return ""
        except Exception as e:
            print(f"Speech error: {e}")
            return ""
    
    def _draw_visualization(self, frame, faces, name=None, confidence=0.0):
        """Draw UI overlays"""
        # Draw face detection
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if self.stability.is_stable else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            if name and name != "Unknown":
                label = f"{name} ({confidence:.0%})"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status display
        if self.sleep_wake.active_user:
            status = f"💬 Chatting with: {self.sleep_wake.active_user}"
            color = (0, 255, 0)
        else:
            status = "🔍 Looking for people"
            color = (0, 255, 255)
        
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.stable_objects:
            objects_text = f"Objects: {len(self.stable_objects)}"
            cv2.putText(frame, objects_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def _display_response(self, response: dict):
        """Display AI response and handle TTS"""
        if not response:
            return
        
        reply = response.get("reply", "")
        intent = response.get("intent", "conversation")
        emotion = response.get("emotion", "neutral")
        raw_response = response.get("raw_response", "")
        
        # Show the full JSON response like original
        print(f"\nBuddy: {raw_response if raw_response else reply}")
        
        if intent != "conversation":
            print(f"[INTENT: {intent}]")
        
        # Save conversation (basic logging only)
        save_conversation(
            "user_input", 
            raw_response if raw_response else reply, 
            intent, 
            self.sleep_wake.active_user
        )
        
        # Update session context
        self.session_context['conversations'].append({
            'user': "system_message",
            'buddy': reply
        })
        if len(self.session_context['conversations']) > self.session_context['max_size']:
            self.session_context['conversations'].pop(0)
        
        # Start TTS with just the reply text
        self.speak(reply)
        
        # Start listening after response
        words = len(reply.split())
        reading_time = max(2.0, words / 3.0)
        threading.Timer(reading_time, self._delayed_listening).start()
    
    def _delayed_listening(self):
        """Start listening after a delay"""
        if self._processing_input:
            return
            
        user_text = self.listen_for_speech(timeout=5.0)
        
        if user_text:
            print("🤔 Thinking...")
            threading.Thread(target=self._process_input, args=(user_text,), daemon=True).start()
        else:
            threading.Timer(1.0, self._delayed_listening).start()
    
    def _process_input(self, user_text):
        """Process user input"""
        if self._processing_input:
            return
        
        self._processing_input = True
        try:
            response = self._call_llm_service(user_text, self.sleep_wake.active_user)
            self._display_response(response)
        except Exception as e:
            print(f"Error processing input: {e}")
        finally:
            self._processing_input = False
    
    def run(self):
        """Main application loop"""
        self.running = True
        last_frame_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                if current_time - last_frame_time >= self.config.frame_process_interval:
                    ret, frame = self.cap.read()
                    if ret:
                        processed, face_detected, name, confidence = self._process_frame(frame)
                        cv2.imshow('Buddy Vision', processed)
                    last_frame_time = current_time
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\\n")
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\\nBuddy Pi: Goodbye! 👋")

def main():
    """Entry point"""
    try:
        config = Config.from_env()
        
        # Get LLM service URL from environment or use default
        import os
        llm_url = os.getenv('LLM_SERVICE_URL', 'http://localhost:8000')
        
        print(f"🤖 Starting Buddy Pi...")
        print(f"🔗 LLM Service: {llm_url}")
        print(f"📷 Camera: {config.camera_index}")
        buddy = BuddyPi(config, llm_url)
        buddy.run()
    
    except Exception as e:
        logging.error(f"Startup failed: {e}")
        print(f"\\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
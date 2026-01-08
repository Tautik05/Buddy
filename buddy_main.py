"""
Buddy Main Application - Modular Chat + Face Recognition with Sleep/Wake
Version: 3.0.0
Features:
- Modular architecture
- Sleep when no person detected
- Wake and greet when person appears
- Learn new faces
- Context-aware conversations
"""

import cv2
import numpy as np
import time
import logging
import json
import html
import re
from pathlib import Path
from typing import Optional

# Import our modules
from config import Config
from states import BuddyState, StateManager
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from stability_tracker import StabilityTracker
from input_handler import InputHandler
from sleep_wake_manager import SleepWakeManager
from buddy_brain import ask_buddy
from objrecog.obj import ObjectDetector
from objrecog.perception import interpret_objects
from collections import defaultdict

# Speech imports
import speech_recognition as sr
import edge_tts
import asyncio
import pygame
import io
import threading


class IntegratedBuddy:
    """Main application - modular chat with face recognition"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.state_manager = StateManager()
        
        # Initialize camera
        self._init_camera()
        
        # Initialize face components
        self.detector = FaceDetector(self.config.cascade_path, self.config)
        self.face_recognizer = FaceRecognizer(self.config.model_path, self.config)  # Renamed
        self.stability = StabilityTracker(self.config)
        
        # Initialize object detection
        self.object_detector = ObjectDetector(confidence_threshold=0.7)  # Higher confidence
        self.object_history = defaultdict(int)
        self.object_absence = defaultdict(int)
        self.stable_objects = set()
        self.last_object_detection = 0
        
        # Stricter thresholds for reliability
        self.APPEAR_THRESHOLD = 5  # Object must be seen 5 times
        self.DISAPPEAR_THRESHOLD = 8  # Object must be missing 8 times
        
        # Initialize managers
        self.sleep_wake = SleepWakeManager(self.config, self.state_manager)
        self.input_handler = InputHandler()
        
        # Initialize speech components
        self._init_speech()
        
        # Application state
        self.unknown_face_img: Optional[np.ndarray] = None
        self.awaiting_name = False
        self.last_recognition_time = 0
        self.running = False
        self._listening = False  # Initialize listening state
        self._processing_input = False  # Initialize processing state
        
        self.state_manager.state = BuddyState.ACTIVE
        
        # Perform initial face detection and greeting
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
    
    def _startup_greeting(self):
        """Detect and greet person on startup"""
        try:
            print("\nStarting camera and face recognition...")
            print("OpenCV window will open - position yourself in front of camera")
            print("Waiting for face recognition to stabilize...\n")
            
            time.sleep(1.0)  # Give camera more time
            
            stable_recognition_count = 0
            required_stable_frames = 5
            
            for attempt in range(50):
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                faces = self.detector.detect(frame)
                
                if faces:
                    largest_face = self.detector.get_largest_face(faces)
                    is_stable = self.stability.update(largest_face)
                    x, y, w, h = largest_face
                    
                    # Show video feed
                    processed = self._draw_visualization(frame, faces)
                    cv2.imshow('Buddy Vision', processed)
                    cv2.waitKey(1)
                    
                    if w > 80 and h > 80 and is_stable:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.face_recognizer.recognize(face_roi)
                        
                        print(f"DEBUG: Attempt {attempt+1}: '{name}' confidence {confidence:.3f}")
                        
                        if name != "Unknown" and confidence > self.config.confidence_threshold:
                            stable_recognition_count += 1
                            if stable_recognition_count >= required_stable_frames:
                                self.sleep_wake.active_user = name
                                print(f"Recognized {name} successfully!\n")
                                
                                prompt = f"I just recognized {name} on camera. Greet them warmly and ask how they're doing."
                                print(f"DEBUG: Sending to AI: '{prompt}'")
                                response = ask_buddy(prompt, recognized_user=name)
                                print(f"DEBUG: AI raw response: '{response}'")
                                self._display_response(response)
                                return
                        else:
                            stable_recognition_count = 0
                    else:
                        stable_recognition_count = 0
                        processed = self._draw_visualization(frame, faces)
                        cv2.imshow('Buddy Vision', processed)
                        cv2.waitKey(1)
                else:
                    self.stability.reset()
                    stable_recognition_count = 0
                    cv2.imshow('Buddy Vision', frame)
                    cv2.waitKey(1)
                
                time.sleep(0.1)
            
            print("Could not recognize face after startup attempts\n")
            # Set up for name learning if we have a face
            ret, frame = self.cap.read()
            if ret:
                faces = self.detector.detect(frame)
                if faces:
                    largest_face = self.detector.get_largest_face(faces)
                    x, y, w, h = largest_face
                    if w > 80 and h > 80:
                        self.unknown_face_img = frame[y:y+h, x:x+w]
                        self.awaiting_name = True
                        print("DEBUG: Stored unknown face for learning")
            
            context = "[CONTEXT: Unknown person detected on camera at startup]"
            prompt = f"{context} Hello! I don't think we've met before. What's your name?"
            response = ask_buddy(prompt)
            self._display_response(response)
            
        except Exception as e:
            self.logger.error(f"Startup greeting error: {e}", exc_info=True)
            print("Buddy: Hello! I'm ready to chat!")
            self._show_prompt()
    
    
    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool, Optional[str], float]:
        """
        Process frame for face detection and recognition
        Returns: (processed_frame, face_detected, name, confidence)
        """
        faces = self.detector.detect(frame)
        
        name = None
        confidence = 0.0
        face_detected = len(faces) > 0
        
        if face_detected:
            largest_face = self.detector.get_largest_face(faces)
            is_stable = self.stability.update(largest_face)
            
            # Recognize if stable and interval passed
            current_time = time.time()
            if (is_stable and 
                (current_time - self.last_recognition_time) > self.config.recognition_interval):
                
                x, y, w, h = largest_face
                if w > self.config.min_face_size[0] and h > self.config.min_face_size[1]:
                    face_roi = frame[y:y+h, x:x+w]
                    name, confidence = self.face_recognizer.recognize(face_roi)
                    
                    # Store unknown face for learning
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
        current_time = time.time()
        if current_time - self.last_object_detection < 1.0:  # Check every 1 second
            return
        
        detections = self.object_detector.detect(frame)
        current_objects = set([det['name'] for det in detections])
        
        # Update object history
        for obj in current_objects:
            self.object_history[obj] += 1
            self.object_absence[obj] = 0
        
        # Update absence counters
        all_known_objects = set(self.object_history.keys())
        missing_objects = all_known_objects - current_objects
        for obj in missing_objects:
            self.object_absence[obj] += 1
            self.object_history[obj] = 0
        
        # Determine stable objects (stricter threshold)
        for obj, count in self.object_history.items():
            if count >= self.APPEAR_THRESHOLD and obj not in self.stable_objects:
                self.stable_objects.add(obj)
        
        # Remove disappeared objects (stricter threshold)
        objects_to_remove = []
        for obj, absence_count in list(self.object_absence.items()):
            if absence_count >= self.DISAPPEAR_THRESHOLD and obj in self.stable_objects:
                self.stable_objects.remove(obj)
                objects_to_remove.append(obj)
        
        for obj in objects_to_remove:
            del self.object_history[obj]
            del self.object_absence[obj]
        
        self.last_object_detection = current_time
    
    def _init_speech(self):
        """Initialize speech recognition and Edge TTS"""
        try:
            # Initialize speech recognition
            self.speech_recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Initialize pygame for audio playback
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Edge TTS voice - Indian female voices
            self.tts_voice = "en-IN-NeerjaNeural"  # Indian female voice
            # Alternative: "en-IN-PrabhatNeural" (male), "hi-IN-SwaraNeural" (Hindi female)
            
            self.tts_lock = threading.Lock()
            self.is_speaking = False
            
            # Calibrate microphone
            with self.microphone as source:
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
                self.speech_recognizer.energy_threshold = 300
                self.speech_recognizer.pause_threshold = 1.0
                self.speech_recognizer.dynamic_energy_threshold = True
            
            self.speech_enabled = True
            print(f"Speech recognition initialized with voice: {self.tts_voice}")
            
        except Exception as e:
            print(f"Speech initialization failed: {e}")
            self.speech_enabled = False
    
    def speak(self, text):
        """Convert text to speech using Edge TTS"""
        if not self.speech_enabled or not text:
            return
        
        def _speak():
            with self.tts_lock:
                try:
                    self.is_speaking = True
                    # Generate speech with Edge TTS
                    asyncio.run(self._generate_and_play_speech(text))
                    self.is_speaking = False
                except Exception as e:
                    print(f"TTS Error: {e}")
                    self.is_speaking = False
        
        threading.Thread(target=_speak, daemon=True).start()
    
    async def _generate_and_play_speech(self, text):
        """Generate speech using Edge TTS and play it"""
        try:
            # Create TTS communication
            communicate = edge_tts.Communicate(text, self.tts_voice)
            
            # Generate audio data
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            # Play audio using pygame
            if audio_data:
                audio_io = io.BytesIO(audio_data)
                pygame.mixer.music.load(audio_io)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"Edge TTS Error: {e}")
    
    def _start_listening(self):
        """Start listening for speech in background"""
        if self.is_speaking or not self.speech_enabled or self._listening:
            return
        
        # Wait for TTS to complete before listening
        while self.is_speaking:
            time.sleep(0.1)
        
        # Add a small delay after speaking to avoid immediate re-listening
        time.sleep(1.0)
        
        user_text = self.listen_for_speech(timeout=3)
        if user_text:
            print("🤔 Thinking...")
            threading.Thread(target=self._process_input, args=(user_text,), daemon=True).start()
    
    def _process_input(self, user_text):
        """Process user input in background with duplicate prevention"""
        if self._processing_input:
            print("DEBUG: Already processing input, skipping")
            return  # Already processing
        
        self._processing_input = True
        try:
            response = ask_buddy(user_text, recognized_user=self.sleep_wake.active_user)
            if response:  # Only process if not None (not skipped)
                self._display_response(response)
        except Exception as e:
            print(f"Error processing input: {e}")
        finally:
            self._processing_input = False
    
    def listen_for_speech(self, timeout=2):
        """Listen for speech input"""
        if not self.speech_enabled:
            return ""
        
        try:
            # Use a fresh microphone instance each time to avoid context conflicts
            with sr.Microphone() as source:
                print("🎤 Listening...")
                audio = self.speech_recognizer.listen(source, timeout=timeout, phrase_time_limit=6)
            
            print("Processing speech...")
            text = self.speech_recognizer.recognize_google(audio)
            print(f"You said: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("Didn't catch that")
            return ""
        except sr.RequestError as e:
            print(f"Speech error: {e}")
            return ""
        except Exception as e:
            print(f"Unexpected speech error: {e}")
            return ""
    
    def _draw_visualization(self, frame, faces, name=None, confidence=0.0):
        """Draw UI overlays"""
        # Draw face detection
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if self.stability.is_stable else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Show name if recognized
            if name and name != "Unknown":
                label = f"{name} ({confidence:.0%})"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw object detection boxes (less frequent for Pi)
        current_time = time.time()
        if current_time - self.last_object_detection < 2.0:  # Reduced frequency
            detections = self.object_detector.detect(frame)
            frame = self.object_detector.draw_detections(frame, detections)
        
        # Status display
        if self.state_manager.is_sleeping():
            status = "💤 Sleeping (no one around)"
            color = (128, 128, 128)
        elif self.state_manager.is_waking():
            status = "👀 Waking up..."
            color = (255, 255, 0)
        elif self.sleep_wake.active_user:
            status = f"💬 Chatting with: {self.sleep_wake.active_user}"
            color = (0, 255, 0)
        else:
            status = "🔍 Looking for people"
            color = (0, 255, 255)
        
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show stable objects count
        if self.stable_objects:
            objects_text = f"Objects: {len(self.stable_objects)}"
            cv2.putText(frame, objects_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show speech status
        if hasattr(self, 'speech_enabled') and self.speech_enabled:
            cv2.putText(frame, "Voice Active - Speak Anytime", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _show_prompt(self):
        """Show input prompt"""
        if self.state_manager.is_sleeping():
            return  # No prompt when sleeping
        
        # No text prompt needed - voice is default
        pass
    
    def _display_response(self, response: str):
        """Display AI response"""
        if not response:  # Skip if None or empty
            return
            
        # The response from ask_buddy is now just the reply text
        clean_response = html.unescape(response)
        print(f"\nBuddy: {clean_response}")
        self.speak(clean_response)
        self._show_prompt()
    
    def _handle_user_input(self, user_text: str) -> bool:
        """Process user input"""
        if user_text.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("\nBuddy: Goodbye! It was nice talking with you! 👋")
            return False
        
        # Learning new name
        if self.awaiting_name and self.unknown_face_img is not None:
            # Enhanced name detection patterns
            name_patterns = [
                r"(?:i'?m|my name is|call me|i am|name'?s|this is)\s+([a-zA-Z]+)",
                r"^([a-zA-Z]+)$",  # Just a single name
                r"it'?s\s+([a-zA-Z]+)",
                r"([a-zA-Z]+)\s*(?:here|speaking)"
            ]
            
            name = None
            for pattern in name_patterns:
                match = re.search(pattern, user_text.lower())
                if match:
                    name = match.group(1).capitalize()
                    break
            
            if name:
                print(f"DEBUG: Detected name '{name}' from input '{user_text}'")
                if self.face_recognizer.add_face(name, self.unknown_face_img):
                    print(f"DEBUG: Successfully added face for {name}")
                    self.sleep_wake.active_user = name
                    self.unknown_face_img = None
                    self.awaiting_name = False
                    
                    context = f"[CONTEXT: User introduced themselves as {name}, face learned] {user_text}"
                    response = ask_buddy(context, recognized_user=name)
                    if response:
                        self._display_response(response)
                    return True
                else:
                    print(f"DEBUG: Failed to add face for {name}")
        
        # Regular conversation - simplified to avoid duplicate processing
        try:
            response = ask_buddy(user_text, recognized_user=self.sleep_wake.active_user)
            if response:  # Only display if not None (not skipped)
                self._display_response(response)
        except Exception as e:
            self.logger.error(f"AI error: {e}", exc_info=True)
            print(f"\nBuddy: Sorry, I had trouble with that. {str(e)}")
            self._show_prompt()
        
        return True
    
    def run(self):
        """Main application loop"""
        self.running = True
        
        last_frame_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Process video frames
                if current_time - last_frame_time >= self.config.frame_process_interval:
                    ret, frame = self.cap.read()
                    if ret:
                        processed, face_detected, name, confidence = self._process_frame(frame)
                        
                        # Update sleep/wake state
                        state_changed, greeting = self.sleep_wake.update(
                            face_detected, name, confidence
                        )
                        
                        # Display greeting if state changed
                        if state_changed and greeting:
                            if self.state_manager.is_sleeping():
                                print(f"\n{greeting}")
                            else:
                                print()
                                self._display_response(greeting)
                        
                        # Show video
                        cv2.imshow('Buddy Vision', processed)
                    
                    last_frame_time = current_time
                
                # Handle voice input (default - no keyboard needed)
                if not self.state_manager.is_sleeping():
                    # Simple continuous listening - no complex state checks
                    if not self.is_speaking:
                        try:
                            speech_text = self.listen_for_speech(timeout=2)
                            if speech_text and not self._processing_input:
                                print("🤔 Thinking...")
                                threading.Thread(target=self._process_input, args=(speech_text,), daemon=True).start()
                        except Exception as e:
                            print(f"Voice input error: {e}")
                            time.sleep(0.1)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n")
                    break
                
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n")
        except Exception as e:
            self.logger.error(f"Main loop error: {e}", exc_info=True)
            print(f"\nError: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.state_manager.state = BuddyState.SHUTDOWN
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\nBuddy: Goodbye! 👋")


def main():
    """Entry point"""
    try:
        config = Config.from_env()
        
        # Validate model
        model_path = Path(config.model_path)
        if not model_path.exists():
            if not model_path.is_absolute():
                model_path = Path(__file__).parent / config.model_path
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {config.model_path}")
        
        config.model_path = str(model_path)
        
        # Run
        buddy = IntegratedBuddy(config)
        buddy.run()
        
        return 0
    
    except Exception as e:
        logging.error(f"Startup failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
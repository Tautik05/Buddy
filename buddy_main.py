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
        self.recognizer = FaceRecognizer(self.config.model_path, self.config)
        self.stability = StabilityTracker(self.config)
        
        # Initialize managers
        self.sleep_wake = SleepWakeManager(self.config, self.state_manager)
        self.input_handler = InputHandler()
        
        # Application state
        self.unknown_face_img: Optional[np.ndarray] = None
        self.awaiting_name = False
        self.last_recognition_time = 0
        self.running = False
        
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
                        name, confidence = self.recognizer.recognize(face_roi)
                        
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
                    name, confidence = self.recognizer.recognize(face_roi)
                    
                    # Store unknown face for learning
                    if name == "Unknown" and self.unknown_face_img is None:
                        self.unknown_face_img = face_roi
                    
                    self.last_recognition_time = current_time
        else:
            self.stability.reset()
        
        # Draw visualization
        processed = self._draw_visualization(frame, faces, name, confidence)
        
        return processed, face_detected, name, confidence
    
    def _draw_visualization(self, frame, faces, name=None, confidence=0.0):
        """Draw UI overlays"""
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if self.stability.is_stable else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Show name if recognized
            if name and name != "Unknown":
                label = f"{name} ({confidence:.0%})"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
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
        
        return frame
    
    def _show_prompt(self):
        """Show input prompt"""
        if self.state_manager.is_sleeping():
            return  # No prompt when sleeping
        
        user = self.sleep_wake.active_user or "You"
        print(f"{user}: ", end="", flush=True)
    
    def _display_response(self, response: str):
        """Display AI response"""
        try:
            decoded = html.unescape(response)
            data = json.loads(decoded)
            reply = data.get('reply', '')
            if reply:
                print(f"\nBuddy: {reply}")
            else:
                print(f"\nBuddy: {decoded}")
        except (json.JSONDecodeError, KeyError):
            clean = html.unescape(response)
            clean = re.sub(r'\[CONTEXT:.*?\]\s*', '', clean)
            print(f"\nBuddy: {clean}")
        
        self._show_prompt()
    
    def _handle_user_input(self, user_text: str) -> bool:
        """Process user input"""
        if user_text.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("\nBuddy: Goodbye! It was nice talking with you! 👋")
            return False
        
        # Learning new name
        if self.awaiting_name and self.unknown_face_img is not None:
            name_match = re.search(
                r"(?:i'?m|my name is|call me|i am|name'?s|this is)\s+([a-zA-Z]+)",
                user_text.lower()
            )
            
            if name_match:
                name = name_match.group(1).capitalize()
                if self.recognizer.add_face(name, self.unknown_face_img):
                    self.sleep_wake.active_user = name
                    self.unknown_face_img = None
                    self.awaiting_name = False
                    
                    context = f"[CONTEXT: User introduced themselves as {name}, face learned] {user_text}"
                    response = ask_buddy(context, recognized_user=name)
                    self._display_response(response)
                    return True
        
        # Regular conversation
        context_parts = []
        if self.sleep_wake.active_user:
            context_parts.append(f"User {self.sleep_wake.active_user} says")
        
        full_message = f"[CONTEXT: {', '.join(context_parts)}] {user_text}" if context_parts else user_text
        
        try:
            response = ask_buddy(full_message, recognized_user=self.sleep_wake.active_user)
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
                
                # Handle keyboard input (only when not sleeping)
                if not self.state_manager.is_sleeping():
                    enter_pressed, user_text = self.input_handler.check_input()
                    
                    if enter_pressed and user_text.strip():
                        if not self._handle_user_input(user_text.strip()):
                            break
                
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
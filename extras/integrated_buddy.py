"""
Integrated Buddy - Production-Ready Face Recognition and AI Chat System
Version: 2.0.0
Focus: Chat-first with face recognition as context enhancement
"""

import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine
import json
import os
import time
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import threading
import queue
from enum import Enum
import html
import re
import sys

# Import custom modules with error handling
try:
    from buddy_brain import ask_buddy
    from memory import (
        save_memory, get_memory, save_face, 
        get_all_faces, update_face_name, get_all_memory
    )
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise

# Platform-specific input handling
if sys.platform == 'win32':
    import msvcrt
    USE_MSVCRT = True
    select = None
else:
    import select
    USE_MSVCRT = False


class BuddyState(Enum):
    """System states for the Buddy application"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SLEEPING = "sleeping"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class Config:
    """Configuration management for the Buddy system"""
    # Model paths
    model_path: str = "face-recog/MobileFaceNet.onnx"
    cascade_path: str = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    # Camera settings
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_buffer_size: int = 1
    
    # Recognition thresholds
    recognition_threshold: float = 0.3
    confidence_threshold: float = 0.5
    min_face_size: Tuple[int, int] = (50, 50)
    max_face_size: Tuple[int, int] = (600, 600)
    
    # Timing settings
    recognition_interval: float = 2.0  # Check faces every 2 seconds
    sleep_threshold: int = 150  # More frames before sleeping
    frame_process_interval: float = 0.1  # Process frames less frequently
    
    # Motion detection
    motion_threshold: int = 20
    stability_frames: int = 5
    face_history_size: int = 10
    
    # Chat settings
    chat_priority: bool = True  # Chat always takes priority over face recognition
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "buddy.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        config = cls()
        config.model_path = os.getenv('BUDDY_MODEL_PATH', config.model_path)
        config.camera_index = int(os.getenv('BUDDY_CAMERA_INDEX', config.camera_index))
        config.log_level = os.getenv('BUDDY_LOG_LEVEL', config.log_level)
        return config


class FaceDetector:
    """Handles face detection with multiple strategies"""
    
    def __init__(self, cascade_path: str, config: Config):
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")
        self.logger = logging.getLogger(__name__ + ".FaceDetector")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better face detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        return gray
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image using multiple strategies"""
        gray = self.preprocess_image(image)
        
        # Multi-scale detection
        faces1 = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=self.config.min_face_size,
            maxSize=self.config.max_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces2 = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            maxSize=(500, 500)
        )
        
        # Merge and deduplicate
        all_faces = list(faces1) + list(faces2)
        return self._remove_duplicates(all_faces)
    
    def _remove_duplicates(
        self, 
        faces: List[Tuple[int, int, int, int]], 
        overlap_threshold: int = 30
    ) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping face detections"""
        if len(faces) == 0:
            return []
        
        filtered = []
        for face in faces:
            x, y, w, h = face
            is_duplicate = False
            
            for ex, ey, ew, eh in filtered:
                if (abs(x - ex) < overlap_threshold and 
                    abs(y - ey) < overlap_threshold and
                    abs(w - ew) < overlap_threshold and 
                    abs(h - eh) < overlap_threshold):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(face)
        
        return filtered


class FaceRecognizer:
    """Handles face recognition using ONNX model"""
    
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".FaceRecognizer")
        
        # Load ONNX model
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, session_options)
        self.input_name = self.session.get_inputs()[0].name
        
        self.known_faces: Dict[str, np.ndarray] = {}
        self.load_known_faces()
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for recognition model"""
        face_resized = cv2.resize(face_img, (112, 112))
        face_normalized = (face_resized - 127.5) / 128.0
        return np.expand_dims(
            face_normalized.transpose(2, 0, 1), 
            axis=0
        ).astype(np.float32)
    
    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extract face embedding from image"""
        preprocessed = self.preprocess_face(face_img)
        embedding = self.session.run(None, {self.input_name: preprocessed})[0]
        return embedding.flatten()
    
    def recognize(self, face_img: np.ndarray) -> Tuple[str, float]:
        """Recognize a face from image"""
        try:
            # Enhance image
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_enhanced = cv2.equalizeHist(face_gray)
            face_enhanced = cv2.cvtColor(face_enhanced, cv2.COLOR_GRAY2BGR)
            
            embedding = self.get_embedding(face_enhanced)
            
            if not self.known_faces:
                return "Unknown", 0.0
            
            best_match = None
            min_distance = float('inf')
            
            for name, known_embedding in self.known_faces.items():
                distance = cosine(embedding, known_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = name
            
            confidence = 1 - min_distance
            
            if min_distance < self.config.recognition_threshold:
                return best_match, confidence
            
            return "Unknown", 0.0
            
        except Exception as e:
            self.logger.error(f"Recognition error: {e}", exc_info=True)
            return "Unknown", 0.0
    
    def add_face(self, name: str, face_img: np.ndarray) -> bool:
        """Add a new face to known faces"""
        try:
            embedding = self.get_embedding(face_img)
            self.known_faces[name] = embedding
            save_face(name, embedding, 0.95)
            self.logger.info(f"Added face for user: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add face for {name}: {e}", exc_info=True)
            return False
    
    def load_known_faces(self) -> None:
        """Load known faces from database"""
        try:
            faces = get_all_faces()
            self.known_faces = {
                html.unescape(name): embedding 
                for name, embedding in faces.items()
            }
            self.logger.info(f"Loaded {len(self.known_faces)} known faces")
        except Exception as e:
            self.logger.error(f"Failed to load known faces: {e}", exc_info=True)
            self.known_faces = {}


class StabilityTracker:
    """Tracks face stability for reliable recognition"""
    
    def __init__(self, config: Config):
        self.config = config
        self.face_history: List[Tuple[int, int]] = []
        self.stable_count = 0
    
    def update(self, face: Tuple[int, int, int, int]) -> bool:
        """Update with new face position and check stability"""
        x, y, w, h = face
        center = (x + w // 2, y + h // 2)
        
        self.face_history.append(center)
        if len(self.face_history) > self.config.face_history_size:
            self.face_history.pop(0)
        
        if len(self.face_history) < 3:
            self.stable_count = 0
            return False
        
        if self._is_stable():
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        return self.stable_count >= self.config.stability_frames
    
    def _is_stable(self) -> bool:
        """Check if recent positions indicate stability"""
        recent = self.face_history[-3:]
        max_distance = 0
        
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                dist = np.sqrt(
                    (recent[i][0] - recent[j][0]) ** 2 + 
                    (recent[i][1] - recent[j][1]) ** 2
                )
                max_distance = max(max_distance, dist)
        
        return max_distance < self.config.motion_threshold
    
    def reset(self) -> None:
        """Reset tracking state"""
        self.face_history.clear()
        self.stable_count = 0
    
    @property
    def is_stable(self) -> bool:
        """Check if currently stable"""
        return self.stable_count >= self.config.stability_frames


class IntegratedBuddy:
    """Main application class - Chat-first with face recognition context"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing Integrated Buddy System")
        self.state = BuddyState.INITIALIZING
        
        # Initialize components
        self._init_camera()
        self._init_face_components()
        
        # State management - Chat focused
        self.active_user: Optional[str] = None
        self.unknown_face_img: Optional[np.ndarray] = None
        self.last_recognition_time: float = 0
        self.no_face_count: int = 0
        
        # Input handling
        self.user_input = ""
        self.awaiting_name = False  # Flag for name learning mode
        
        self.running = False
        self.state = BuddyState.ACTIVE
        self.logger.info("Initialization complete")
        
        # Initial greeting
        self._startup_greeting()
    
    def _setup_logging(self) -> None:
        """Configure logging system"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.log_file) 
                if self.config.log_file else logging.NullHandler()
            ]
        )
    
    def _init_camera(self) -> None:
        """Initialize camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(self.config.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")
            
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera_buffer_size)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            
            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}", exc_info=True)
            raise
    
    def _init_face_components(self) -> None:
        """Initialize face detection and recognition components"""
        try:
            self.detector = FaceDetector(self.config.cascade_path, self.config)
            self.recognizer = FaceRecognizer(self.config.model_path, self.config)
            self.stability = StabilityTracker(self.config)
            self.logger.info("Face components initialized")
        except Exception as e:
            self.logger.error(f"Face component initialization failed: {e}", exc_info=True)
            raise
    
    def _startup_greeting(self) -> None:
        """Perform startup greeting with face detection"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Buddy: Hello! I'm ready to chat!")
                self._show_prompt()
                return
            
            faces = self.detector.detect(frame)
            
            if faces:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                if w > 80 and h > 80:
                    face_roi = frame[y:y+h, x:x+w]
                    name, confidence = self.recognizer.recognize(face_roi)
                    
                    if name != "Unknown" and confidence > self.config.confidence_threshold:
                        self.active_user = name
                        # Provide context to AI about who is talking
                        response = ask_buddy(f"[CONTEXT: User {name} just appeared on camera] Hello {name}! How are you doing today?")
                    else:
                        self.unknown_face_img = face_roi
                        self.awaiting_name = True
                        response = ask_buddy("[CONTEXT: Unknown person detected on camera] Hello there! I don't think we've met. What's your name?")
                    
                    self._display_response(response)
                    return
            
            print("Buddy: Hello! I'm ready to chat!")
            self._show_prompt()
            
        except Exception as e:
            self.logger.error(f"Startup greeting error: {e}", exc_info=True)
            print(f"Buddy: Hello! I'm ready to chat.")
            self._show_prompt()
    
    def _show_prompt(self) -> None:
        """Show input prompt"""
        prompt = f"{self.active_user}: " if self.active_user else "You: "
        print(prompt, end="", flush=True)
    
    def _display_response(self, response: str) -> None:
        """Display AI response with proper formatting"""
        try:
            # Try to parse as JSON first
            decoded = html.unescape(response)
            data = json.loads(decoded)
            reply = data.get('reply', '')
            print(f"\nBuddy: {reply}")
        except (json.JSONDecodeError, KeyError):
            # If not JSON, just clean and display
            clean = html.unescape(response)
            # Remove [CONTEXT: ...] tags if they appear in response
            clean = re.sub(r'\[CONTEXT:.*?\]\s*', '', clean)
            print(f"\nBuddy: {clean}")
        
        self._show_prompt()
    
    def _check_keyboard_input(self) -> bool:
        """Check for keyboard input (returns True if Enter was pressed)"""
        if USE_MSVCRT:
            # Windows
            try:
                if msvcrt.kbhit():
                    char = msvcrt.getch().decode('utf-8', errors='ignore')
                    if char == '\r':
                        return True
                    elif char == '\b':
                        if self.user_input:
                            self.user_input = self.user_input[:-1]
                            prompt = f"\r{self.active_user or 'You'}: {self.user_input} "
                            print(prompt, end="", flush=True)
                    elif char.isprintable():
                        self.user_input += char
                        print(char, end="", flush=True)
            except:
                pass
        else:
            # Unix/Linux - non-blocking input check
            if not USE_MSVCRT and select is not None and select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                if char == '\n':
                    return True
                elif char == '\x7f':  # backspace
                    if self.user_input:
                        self.user_input = self.user_input[:-1]
                        print('\b \b', end="", flush=True)
                elif char.isprintable():
                    self.user_input += char
                    print(char, end="", flush=True)
        
        return False
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for face recognition (background operation)"""
        current_time = time.time()
        faces = self.detector.detect(frame)
        
        # Handle no faces detected
        if not faces:
            self.no_face_count += 1
            self.stability.reset()
            
            if (self.no_face_count > self.config.sleep_threshold and 
                self.state != BuddyState.SLEEPING):
                self.state = BuddyState.SLEEPING
                self.active_user = None
                self.logger.info("Entering sleep mode - no faces detected")
        else:
            # Handle faces detected
            self.no_face_count = 0
            
            if self.state == BuddyState.SLEEPING:
                self.state = BuddyState.ACTIVE
                self.logger.info("Waking from sleep mode")
            
            # Get largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            
            # Check stability
            is_stable = self.stability.update(largest_face)
            
            # Perform recognition on stable faces (in background, doesn't interrupt chat)
            if (is_stable and 
                (current_time - self.last_recognition_time) > self.config.recognition_interval):
                
                x, y, w, h = largest_face
                if w > self.config.min_face_size[0] and h > self.config.min_face_size[1]:
                    face_roi = frame[y:y+h, x:x+w]
                    self._handle_face_recognition_background(face_roi)
                    self.last_recognition_time = current_time
        
        # Draw visualization
        return self._draw_visualization(frame, faces)
    
    def _draw_visualization(
        self, 
        frame: np.ndarray, 
        faces: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Draw visualization overlays on frame"""
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if self.stability.is_stable else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Status display
        if self.state == BuddyState.SLEEPING:
            status = "Sleeping (no faces)"
            color = (128, 128, 128)
        elif self.active_user:
            status = f"Chatting with: {self.active_user}"
            color = (0, 255, 0)
        else:
            status = f"Ready to chat (detecting faces)"
            color = (0, 255, 255)
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def _handle_face_recognition_background(self, face_img: np.ndarray) -> None:
        """Handle face recognition in background (doesn't interrupt conversation)"""
        try:
            name, confidence = self.recognizer.recognize(face_img)
            
            if name != "Unknown" and confidence > self.config.confidence_threshold:
                if not self.active_user or self.active_user != name:
                    # Silently update active user - don't interrupt conversation
                    self.active_user = name
                    self.logger.info(f"User changed to: {name} (confidence: {confidence:.2f})")
            else:
                if self.unknown_face_img is None and not self.awaiting_name:
                    # New unknown face detected
                    self.unknown_face_img = face_img
                    self.awaiting_name = True
                    # Ask for name without interrupting if user is typing
                    if not self.user_input:
                        print("\n")
                        response = ask_buddy("[CONTEXT: Unknown person just appeared on camera] Hello! I see someone new. What's your name?")
                        self._display_response(response)
                        
        except Exception as e:
            self.logger.error(f"Background face recognition error: {e}", exc_info=True)
    
    def _handle_user_input(self, user_text: str) -> bool:
        """Handle user text input - THIS IS THE MAIN FUNCTION"""
        if user_text.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("\nBuddy: Goodbye! It was nice talking with you!")
            return False
        
        # Check if we're learning a new name
        if self.awaiting_name and self.unknown_face_img is not None:
            # Look for name patterns
            name_match = re.search(
                r"(?:i'?m|my name is|call me|i am|name'?s|this is)\s+([a-zA-Z]+)",
                user_text.lower()
            )
            
            if name_match:
                name = name_match.group(1).capitalize()
                if self.recognizer.add_face(name, self.unknown_face_img):
                    self.active_user = name
                    self.unknown_face_img = None
                    self.awaiting_name = False
                    
                    # Let AI know we learned the name
                    context_msg = f"[CONTEXT: User introduced themselves as {name}, and I've now learned to recognize their face] {user_text}"
                    response = ask_buddy(context_msg)
                    self._display_response(response)
                    return True
        
        # Regular conversation - Build context for AI
        context_parts = []
        
        # Add user identity context if known
        if self.active_user:
            context_parts.append(f"User {self.active_user} says")
        
        # Build full message with context
        if context_parts:
            full_message = f"[CONTEXT: {', '.join(context_parts)}] {user_text}"
        else:
            full_message = user_text
        
        # Send to AI and get response
        try:
            response = ask_buddy(full_message)
            self._display_response(response)
        except Exception as e:
            self.logger.error(f"AI response error: {e}", exc_info=True)
            print(f"\nBuddy: Sorry, I had trouble processing that. {str(e)}")
            self._show_prompt()
        
        return True
    
    def run(self) -> None:
        """Main application loop - CHAT FOCUSED"""
        self.running = True
        self.logger.info("Starting main loop - Chat priority mode")
        print("\n🤖 Buddy is ready to chat! Face recognition running in background.")
        print("💬 Type your message and press Enter. Type 'quit' to exit.\n")
        
        last_frame_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Process video frames in background (low priority)
                if current_time - last_frame_time >= self.config.frame_process_interval:
                    ret, frame = self.cap.read()
                    if ret:
                        processed_frame = self._process_frame(frame)
                        cv2.imshow('Buddy Vision', processed_frame)
                    
                    last_frame_time = current_time
                
                # Check for keyboard input (HIGH PRIORITY)
                if self._check_keyboard_input():
                    user_text = self.user_input.strip()
                    self.user_input = ""
                    
                    if user_text:
                        if not self._handle_user_input(user_text):
                            break
                
                # Check for 'q' key in video window
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n")
                    break
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                    
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            print("\n")
        except Exception as e:
            self.logger.error(f"Main loop error: {e}", exc_info=True)
            self.state = BuddyState.ERROR
            print(f"\nError: {e}")
        finally:
            self.running = False
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.logger.info("Cleaning up resources")
        self.state = BuddyState.SHUTDOWN
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        print("\nGoodbye! 👋")
        self.logger.info("Shutdown complete")


def main():
    """Application entry point"""
    try:
        # Load configuration
        config = Config.from_env()
        
        # Validate model path
        model_path = Path(config.model_path)
        if not model_path.exists():
            if not model_path.is_absolute():
                # Try relative to script directory
                script_dir = Path(__file__).parent
                model_path = script_dir / config.model_path
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {config.model_path}")
        
        config.model_path = str(model_path)
        
        # Initialize and run
        buddy = IntegratedBuddy(config)
        buddy.run()
        
        return 0
        
    except Exception as e:
        logging.error(f"Startup failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
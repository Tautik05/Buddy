"""
Simple Sleep/Wake Manager for Pi
"""

import time
import logging
from states import BuddyState

class SleepWakeManager:
    """Manages sleep/wake cycles"""
    
    def __init__(self, config, state_manager):
        self.config = config
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        self.no_face_count = 0
        self.last_wake_time = 0
        self.wake_greeting_sent = False
        self.active_user = None
        self.failed_voice_attempts = 0
    
    def update(self, faces_detected: bool, recognized_name: str = None, confidence: float = 0.0):
        """Update sleep/wake state based on face detection"""
        if faces_detected:
            return self._handle_face_present(recognized_name, confidence)
        else:
            return self._handle_no_face()
    
    def _handle_face_present(self, name: str, confidence: float):
        """Handle when a face is detected"""
        self.no_face_count = 0
        
        if self.state_manager.is_sleeping():
            self.state_manager.state = BuddyState.WAKING
            self.last_wake_time = time.time()
            self.wake_greeting_sent = False
            return True, None
        
        if self.state_manager.is_waking():
            if time.time() - self.last_wake_time >= self.config.wake_delay:
                self.state_manager.state = BuddyState.ACTIVE
                
                if not self.wake_greeting_sent:
                    self.wake_greeting_sent = True
                    return True, "Hello! Nice to see you!"
        
        if name and name != "Unknown" and confidence > self.config.confidence_threshold:
            if self.active_user != name:
                self.active_user = name
        
        return False, None
    
    def _handle_no_face(self):
        """Handle when no face is detected"""
        self.no_face_count += 1
        
        if (self.no_face_count >= self.config.sleep_threshold and 
            not self.state_manager.is_sleeping()):
            
            self.state_manager.state = BuddyState.SLEEPING
            self.active_user = None
            self.wake_greeting_sent = False
            
            return True, self.config.sleep_message
        
        return False, None
    
    def reset(self):
        """Reset sleep/wake state"""
        self.no_face_count = 0
        self.wake_greeting_sent = False
        self.active_user = None
        self.failed_voice_attempts = 0
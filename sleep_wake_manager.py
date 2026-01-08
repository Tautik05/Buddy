"""
Sleep/Wake Manager Module
Handles sleep and wake transitions with greetings
"""

import time
import logging
from buddy_brain import ask_buddy
from states import BuddyState


class SleepWakeManager:
    """Manages sleep/wake cycles and greetings"""
    
    def __init__(self, config, state_manager):
        self.config = config
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        self.no_face_count = 0
        self.last_wake_time = 0
        self.wake_greeting_sent = False
        self.active_user = None
    
    def update(self, faces_detected: bool, recognized_name: str = None, confidence: float = 0.0):
        """
        Update sleep/wake state based on face detection
        Returns: (state_changed, greeting_message)
        """
        if faces_detected:
            return self._handle_face_present(recognized_name, confidence)
        else:
            return self._handle_no_face()
    
    def _handle_face_present(self, name: str, confidence: float):
        """Handle when a face is detected"""
        self.no_face_count = 0
        
        # Wake up if sleeping
        if self.state_manager.is_sleeping():
            self.state_manager.state = BuddyState.WAKING
            self.last_wake_time = time.time()
            self.wake_greeting_sent = False
            return True, None
        
        # Send greeting after wake delay
        if self.state_manager.is_waking():
            if time.time() - self.last_wake_time >= self.config.wake_delay:
                self.state_manager.state = BuddyState.ACTIVE
                
                if not self.wake_greeting_sent:
                    self.wake_greeting_sent = True
                    greeting = self._generate_greeting(name, confidence)
                    return True, greeting
        
        # Update active user if recognized
        if name and name != "Unknown" and confidence > self.config.confidence_threshold:
            if self.active_user != name:
                self.active_user = name
        
        return False, None
    
    def _handle_no_face(self):
        """Handle when no face is detected"""
        self.no_face_count += 1
        
        # Go to sleep after threshold
        if (self.no_face_count >= self.config.sleep_threshold and 
            not self.state_manager.is_sleeping()):
            
            self.state_manager.state = BuddyState.SLEEPING
            self.active_user = None
            self.wake_greeting_sent = False
            
            return True, self.config.sleep_message
        
        return False, None
    
    def _generate_greeting(self, name: str, confidence: float):
        """Generate appropriate greeting based on recognition"""
        if name and name != "Unknown" and confidence > self.config.confidence_threshold:
            # Known person
            context = f"[CONTEXT: User {name} just appeared after I was sleeping] "
            prompt = f"{context}Hello {name}! Nice to see you again! How are you doing?"
            return ask_buddy(prompt, recognized_user=name)
        else:
            # Unknown person
            context = "[CONTEXT: Unknown person appeared after I was sleeping] "
            prompt = f"{context}Hello there! I just woke up. I don't think we've met - what's your name?"
            return ask_buddy(prompt)
    
    def reset(self):
        """Reset sleep/wake state"""
        self.no_face_count = 0
        self.wake_greeting_sent = False
        self.active_user = None
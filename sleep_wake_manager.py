"""
Sleep/Wake Manager Module
Handles sleep and wake transitions with greetings and voice-based sleep
"""

import time
import logging
import speech_recognition as sr
from buddy_brain import ask_buddy
from states import BuddyState


class SleepWakeManager:
    """Manages sleep/wake cycles and greetings with voice-based sleep"""
    
    def __init__(self, config, state_manager):
        self.config = config
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        # Face-based sleep/wake
        self.no_face_count = 0
        self.last_wake_time = 0
        self.wake_greeting_sent = False
        self.active_user = None
        
        # Voice-based sleep/wake
        self.failed_voice_attempts = 0
        
        # Initialize speech recognizer for wake word detection
        self.wake_recognizer = sr.Recognizer()
        self.wake_recognizer.energy_threshold = 300
        self.wake_recognizer.pause_threshold = 0.8
        self.wake_recognizer.dynamic_energy_threshold = True
    
    def update(self, faces_detected: bool, recognized_name: str = None, confidence: float = 0.0):
        """
        Update sleep/wake state based on face detection
        Returns: (state_changed, greeting_message)
        """
        # Don't process face detection if in voice sleep mode - completely disabled
        if self.state_manager.is_voice_sleeping():
            return False, None
            
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
        self.failed_voice_attempts = 0
    
    def check_voice_sleep_condition(self, speech_input_received: bool):
        """Check if system should go to voice sleep based on failed attempts"""
        if speech_input_received:
            self.failed_voice_attempts = 0
            return False
        
        self.failed_voice_attempts += 1
        
        if (self.failed_voice_attempts >= self.config.max_listening_attempts and 
            not self.state_manager.is_sleeping()):
            
            self.logger.info(f"Going to voice sleep after {self.failed_voice_attempts} failed attempts")
            self.state_manager.state = BuddyState.VOICE_SLEEPING
            self.failed_voice_attempts = 0
            return True
        
        return False
    
    def enter_voice_sleep(self):
        """Enter voice sleep mode and reset counters"""
        # Prevent multiple entries
        if self.state_manager.is_voice_sleeping():
            return
            
        self.state_manager.state = BuddyState.VOICE_SLEEPING
        self.failed_listening_count = 0
        self.failed_voice_attempts = 0
        self.logger.info("Entered voice sleep mode")
        print("💤 Now listening for 'Hey Buddy' to wake up...")
        self._test_microphone_access()
    
    def _test_microphone_access(self):
        """Test if microphone is accessible"""
        try:
            import speech_recognition as sr
            test_mic = sr.Microphone()
            with test_mic as source:
                print("[DEBUG: Microphone access test - OK]")
                return True
        except Exception as e:
            print(f"[DEBUG: Microphone access test - FAILED: {e}]")
            print("[DEBUG: Please check:")
            print("  1. Microphone is connected and working")
            print("  2. No other applications are using the microphone")
            print("  3. Windows microphone permissions are enabled]")
            return False
    
    def listen_for_wake_word(self, microphone, timeout=4.0):  # Optimized timeout
        """Listen specifically for wake word when in voice sleep mode"""
        if not self.state_manager.is_voice_sleeping():
            return False
        
        try:
            print("🎤 Listening for 'Hey Buddy'...")
            
            # Test microphone access first
            try:
                with microphone as source:
                    # Optimized settings for wake word detection
                    self.wake_recognizer.energy_threshold = 300  # Slightly higher for better detection
                    self.wake_recognizer.pause_threshold = 0.8   # Longer pause for clearer detection
                    self.wake_recognizer.dynamic_energy_threshold = True
                    
                    # Quick ambient noise adjustment
                    print("[DEBUG: Adjusting for ambient noise...]")
                    self.wake_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    
                    print("[DEBUG: Microphone ready, listening...]")
                    audio = self.wake_recognizer.listen(source, timeout=timeout, phrase_time_limit=6)
            except Exception as mic_error:
                print(f"❌ Microphone access error: {mic_error}")
                print("[DEBUG: Check if microphone is being used by another application]")
                return False
            
            print("🤔 Processing wake word...")
            text = self.wake_recognizer.recognize_google(audio).lower().strip()
            print(f"[DEBUG: Wake word detection heard: '{text}']")
            
            # Strict wake word matching - only "hey buddy"
            wake_detected = "hey buddy" in text
            
            if wake_detected:
                print(f"✅ Wake word 'hey buddy' detected! Waking up...")
                self.logger.info(f"Wake word detected: '{text}'")
                self.state_manager.state = BuddyState.ACTIVE
                self.failed_voice_attempts = 0
                return True
            else:
                print(f"❌ Wake word 'hey buddy' not found. Still sleeping...")
            
            return False
            
        except sr.WaitTimeoutError:
            print("[DEBUG: No speech detected, continuing to listen...]")
            return False
        except sr.UnknownValueError:
            print("❌ Could not understand audio. Still sleeping...")
            return False
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            print("[DEBUG: Check internet connection for Google Speech Recognition]")
            return False
        except Exception as e:
            print(f"❌ Wake word error: {e}. Still sleeping...")
            self.logger.error(f"Wake word error: {e}")
            return False
    
    def get_voice_sleep_message(self):
        """Get voice sleep message"""
        return self.config.voice_sleep_message
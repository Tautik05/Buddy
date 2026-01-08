import subprocess
import os
import json
import speech_recognition as sr
import pyttsx3
import time
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_memory import init_db, save_memory, get_memory, get_all_memory
from conversation_db import init_conversation_db, save_conversation, get_recent_conversations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(BASE_DIR, "buddy_prompt.txt")

# Initialize memory and conversation systems
init_db()
init_conversation_db()

# Load system prompt
try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except:
    SYSTEM_PROMPT = "You are Buddy, a helpful AI assistant."

class DynamicVoiceBuddy:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize TTS
        print("Initializing TTS...")
        try:
            self.tts = pyttsx3.init()
            self.setup_tts()
            
            # Test TTS immediately
            self.tts.say("Ready")
            self.tts.runAndWait()
            print("TTS working!")
        except Exception as e:
            print(f"TTS failed: {e}")
            self.tts = None
        
        # Calibrate microphone
        self.calibrate_microphone()
        
        print("Dynamic Voice Buddy initialized!")
    
    def setup_tts(self):
        """Setup TTS with optimal settings"""
        voices = self.tts.getProperty('voices')
        if voices:
            self.tts.setProperty('voice', voices[0].id)
        
        self.tts.setProperty('rate', 160)
        self.tts.setProperty('volume', 1.0)
    
    def calibrate_microphone(self):
        """Smart microphone calibration"""
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            self.recognizer.energy_threshold = 100
            self.recognizer.pause_threshold = 1.2
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.non_speaking_duration = 0.8
        print("Microphone ready!")
    
    def dynamic_speak(self, text, emotion="neutral", intent="unknown"):
        """Dynamic TTS with emotion"""
        if not text or not text.strip():
            return

        print(f"Buddy: {text}")
        
        # Use Windows SAPI with better voice selection
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            
            # Get available voices
            voices = speaker.GetVoices()
            
            # Try to find a more natural voice
            for i in range(voices.Count):
                voice = voices.Item(i)
                voice_name = voice.GetDescription().lower()
                if any(name in voice_name for name in ['zira', 'hazel', 'eva', 'aria']):
                    speaker.Voice = voice
                    break
            
            speaker.Rate = 1
            speaker.Speak(text)
            print("Speech completed (SAPI)")
            
        except ImportError:
            # Fallback to pyttsx3 with better voice
            try:
                tts = pyttsx3.init()
                voices = tts.getProperty('voices')
                
                for voice in voices:
                    if 'zira' in voice.name.lower() or 'hazel' in voice.name.lower():
                        tts.setProperty('voice', voice.id)
                        break
                
                tts.setProperty('rate', 180)
                tts.setProperty('volume', 1.0)
                tts.say(text)
                tts.runAndWait()
                print("Speech completed (pyttsx3)")
            except Exception as e:
                print(f"TTS failed: {e}")
    
    def smart_listen(self, timeout=30):
        """Enhanced listening with better accuracy"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=20
                )
            
            print("Processing speech...")
            
            for attempt in range(2):
                try:
                    text = self.recognizer.recognize_google(
                        audio, 
                        language='en-US',
                        show_all=False
                    )
                    
                    if text and len(text.strip()) > 0:
                        print(f"You: {text}")
                        return text.strip()
                        
                except sr.UnknownValueError:
                    if attempt == 0:
                        print("Trying again...")
                        continue
                    else:
                        print("Didn't catch that")
                        return ""
            
            return ""
            
        except sr.WaitTimeoutError:
            print("No speech detected")
            return ""
        except sr.RequestError as e:
            print(f"Speech service error: {e}")
            return ""
    
    def ask_buddy(self, user_input):
        """Get response from Buddy AI"""
        try:
            result = subprocess.run(
                [r"C:\Users\Debanshu\AppData\Local\Programs\Ollama\ollama.exe", "run", "llama3.2:3b"],
                input=user_input,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="ignore"
            )

            response = result.stdout.strip()
            
            if result.stderr:
                print(f"Ollama error: {result.stderr}")

            return response or "I'm not sure how to respond.", "neutral", "unknown"
                
        except Exception as e:
            print(f"Error: {e}")
            return "I'm having trouble right now.", "neutral", "unknown"
    
    def extract_and_store_info(self, text):
        """Extract and store personal information from conversation"""
        import re
        text_lower = text.lower()
        
        # Extract friends - more precise patterns
        friend_patterns = [
            r"my friend is (\w+)",
            r"my friend (\w+)",
            r"friend.*name.*is (\w+)",
            r"i have a friend called (\w+)",
            r"her name is (\w+)",
            r"his name is (\w+)"
        ]
        for pattern in friend_patterns:
            match = re.search(pattern, text_lower)
            if match:
                friend_name = match.group(1).strip().title()
                existing_friends = get_memory("friends") or ""
                if friend_name not in existing_friends and len(friend_name) > 1:
                    new_friends = f"{existing_friends}, {friend_name}" if existing_friends else friend_name
                    save_memory("friends", new_friends)
                    print(f"Stored friend: {friend_name}")
                    break  # Only capture first match
        
        # Extract favorite color - more precise
        color_patterns = [
            r"my favorite color is (\w+)",
            r"i like (\w+) color",
            r"my favourite color is (\w+)"
        ]
        for pattern in color_patterns:
            match = re.search(pattern, text_lower)
            if match:
                color = match.group(1).strip().lower()
                save_memory("favorite_color", color)
                print(f"Stored favorite color: {color}")
                break
        
        # Extract age
        age_patterns = [
            r"i am (\d+) years old",
            r"i'm (\d+) years old", 
            r"i am (\d+)",
            r"i'm (\d+)",
            r"my age is (\d+)"
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = match.group(1)
                if 10 <= int(age) <= 120:
                    save_memory("age", age)
                    print(f"Stored age: {age}")
        
        # Extract location/city - more precise
        location_patterns = [
            r"i live in (\w+)",
            r"i'm from (\w+)",
            r"i am from (\w+)",
            r"my city is (\w+)"
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group(1).strip().title()
                save_memory("location", location)
                print(f"Stored location: {location}")
                break
        
        # Extract occupation/job - more precise
        job_patterns = [
            r"i work as a (\w+)",
            r"i am a (\w+)",
            r"i'm a (\w+)",
            r"my job is (\w+)",
            r"i work in (\w+)"
        ]
        for pattern in job_patterns:
            match = re.search(pattern, text_lower)
            if match:
                job = match.group(1).strip().title()
                save_memory("occupation", job)
                print(f"Stored occupation: {job}")
                break
        
        # Extract hobbies/interests - more precise
        hobby_patterns = [
            r"i like (\w+)",
            r"i love (\w+)",
            r"i enjoy (\w+)",
            r"my hobby is (\w+)"
        ]
        for pattern in hobby_patterns:
            match = re.search(pattern, text_lower)
            if match:
                hobby = match.group(1).strip().lower()
                existing_hobbies = get_memory("hobbies") or ""
                if hobby not in existing_hobbies and len(hobby) > 2:
                    new_hobbies = f"{existing_hobbies}, {hobby}" if existing_hobbies else hobby
                    save_memory("hobbies", new_hobbies)
                    print(f"Stored hobby: {hobby}")
                    break
    
    def run_dynamic_buddy(self):
        """Main conversation loop"""
        self.dynamic_speak("Hello! I'm Buddy. Ready to chat!", "friendly", "greeting")
        
        try:
            while True:
                text = self.smart_listen()
                
                if not text:
                    continue
                
                # Handle exit
                if any(word in text.lower() for word in ['bye', 'goodbye', 'exit', 'quit']):
                    self.dynamic_speak("Goodbye! Talk to you later!", "friendly", "farewell")
                    break
                
                # Get AI response
                response, emotion, intent = self.ask_buddy(text)
                
                self.dynamic_speak(response, emotion, intent)
                
        except KeyboardInterrupt:
            self.dynamic_speak("Goodbye!", "friendly", "farewell")
        except Exception as e:
            print(f"Error: {e}")

def main():
    buddy = DynamicVoiceBuddy()
    buddy.run_dynamic_buddy()

if __name__ == "__main__":
    main()
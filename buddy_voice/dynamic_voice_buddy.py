import subprocess
import os
import json
import pyaudio
import numpy as np
import pyttsx3
import time
import speech_recognition as sr
import threading
from queue import Queue
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import init_db, save_memory, get_memory, get_all_memory
from smart_memory import intelligent_memory_save

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(BASE_DIR, "buddy_prompt.txt")

# Initialize memory system
init_db()

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

class DynamicVoiceBuddy:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize TTS with dynamic settings
        self.tts = pyttsx3.init()
        self.setup_tts()
        
        # Voice activity detection
        self.is_listening = False
        self.speech_queue = Queue()
        
        # Calibrate microphone
        self.calibrate_microphone()
        
        print("Dynamic Voice BUDDY initialized!")
    
    def setup_tts(self):
        """Setup TTS with optimal settings"""
        voices = self.tts.getProperty('voices')
        if voices:
            # Try to find a female voice for warmer interaction
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts.setProperty('voice', voice.id)
                    break
            else:
                self.tts.setProperty('voice', voices[0].id)
        
        self.tts.setProperty('rate', 160)
        self.tts.setProperty('volume', 0.9)
    
    def calibrate_microphone(self):
        """Smart microphone calibration"""
        print("Calibrating microphone... Please stay quiet for 2 seconds.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            # Dynamic settings for better detection
            self.recognizer.energy_threshold = max(300, self.recognizer.energy_threshold)
            self.recognizer.pause_threshold = 1.5
            self.recognizer.dynamic_energy_threshold = True
        print("Microphone calibrated!")
    
    def dynamic_speak(self, text, emotion="neutral", intent="unknown"):
        """Dynamic TTS with emotion and context awareness"""
        if not text or text.strip() == "":
            return
        
        # Adjust speech parameters based on emotion and intent
        rate = 160
        volume = 0.9
        
        if emotion == "happy" or emotion == "excited":
            rate = 180
            volume = 1.0
        elif emotion == "sad" or emotion == "tired":
            rate = 140
            volume = 0.8
        elif emotion == "curious" or intent == "ask_question":
            rate = 170
        elif intent == "greeting":
            rate = 155
            volume = 0.95
        elif emotion == "anxious" or emotion == "frustrated":
            rate = 150
        
        # Apply settings
        self.tts.setProperty('rate', rate)
        self.tts.setProperty('volume', volume)
        
        # Add natural pauses for questions
        if "?" in text:
            text = text.replace("?", "? ")
        
        print(f"BUDDY: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
        
        # Small pause for natural conversation flow
        time.sleep(0.2)
    
    def smart_listen(self, timeout=30, phrase_limit=None):
        """Smart listening with voice activity detection"""
        try:
            with self.microphone as source:
                print("🎤 Listening... (speak naturally)")
                
                # Listen with smart timeout
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit
                )
            
            print("🔄 Processing...")
            
            # Use Google's speech recognition for best accuracy
            text = self.recognizer.recognize_google(audio, language='en-US')
            
            if text:
                print(f"👤 You: {text}")
                return text.strip()
            
            return ""
            
        except sr.WaitTimeoutError:
            print("⏰ No speech detected")
            return ""
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return ""
    
    def continuous_listen(self):
        """Continuous listening mode with wake word detection"""
        print("🔊 Continuous listening mode activated!")
        print("Say 'Hey Buddy' or 'Buddy' to start conversation")
        
        while True:
            try:
                text = self.smart_listen(timeout=5, phrase_limit=3)
                
                if text:
                    # Check for wake words
                    wake_words = ['hey buddy', 'buddy', 'hi buddy', 'hello buddy']
                    if any(wake in text.lower() for wake in wake_words):
                        self.dynamic_speak("Yes? I'm listening!", "friendly", "greeting")
                        return self.conversation_mode()
                    
                    # Direct conversation if already talking
                    elif len(text) > 3:
                        return text
                        
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error in continuous listening: {e}")
                time.sleep(1)
    
    def conversation_mode(self):
        """Active conversation mode"""
        while True:
            text = self.smart_listen(timeout=15)
            
            if not text:
                self.dynamic_speak("I'm still here if you need me!", "friendly")
                continue
            
            # Check for exit commands
            if any(word in text.lower() for word in ['bye', 'goodbye', 'exit', 'quit', 'stop']):
                self.dynamic_speak("Goodbye! Talk to you later!", "friendly", "farewell")
                break
            
            return text
    
    def ask_buddy(self, user_input):
        """Enhanced buddy interaction with memory"""
        # Get current memory context
        memory_context = get_all_memory()
        name_known = "name" in memory_context or "user_name" in memory_context
        user_name = memory_context.get("name") or memory_context.get("user_name", "")

        # Build context-aware prompt
        greeting_instruction = f"\nIMPORTANT: The user's name is '{user_name}'. Use their name in your replies when appropriate.\n" if user_name else ""
        context = f"Current memory: {json.dumps(memory_context)}\nname_known: {name_known}\n{greeting_instruction}"
        prompt = SYSTEM_PROMPT + "\n" + context + "\nUser: " + user_input + "\nAssistant:"

        result = subprocess.run(
            ["ollama", "run", "llama3.2:3b"],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="ignore"
        )

        response = result.stdout.strip()

        # Process memory operations if JSON response
        try:
            data = json.loads(response)
            
            # Normalize response structure
            reply = data.get("reply", "")
            memory = data.get("memory", {})
            emotion = data.get("emotion", "neutral")
            intent = data.get("intent", "unknown")
            
            # Generate fallback reply if empty
            if not reply and memory.get("store"):
                reply = f"Got it{', ' + user_name if user_name else ''}! I'll remember that."
            elif not reply:
                reply = f"I understand{', ' + user_name if user_name else ''}."
                
            normalized = {
                "reply": reply,
                "emotion": emotion,
                "intent": intent,
                "memory": {
                    "store": memory.get("store", False),
                    "key": memory.get("key", ""),
                    "value": memory.get("value", ""),
                    "confidence": memory.get("confidence", 0.0)
                }
            }
            
            # Store memory if needed
            if memory.get("store") and memory.get("key") and memory.get("value"):
                save_memory(memory["key"], memory["value"], memory.get("confidence", 0.7))
            
            # Intelligent memory extraction
            smart_memories = intelligent_memory_save(user_input, intent, memory_context)
            for mem_op in smart_memories:
                save_memory(mem_op["key"], mem_op["value"], mem_op["confidence"])
            
            return reply, emotion, intent
            
        except Exception:
            return response, "neutral", "unknown"
    
    def run(self):
        """Main conversation loop"""
        # Get user's name if not known
        memory_context = get_all_memory()
        user_name = memory_context.get("name")
        
        if not user_name:
            self.dynamic_speak("Hello! I'm BUDDY, your AI companion. What's your name?", "friendly", "greeting")
            name_input = self.smart_listen(timeout=10)
            if name_input:
                # Extract name and save
                smart_memories = intelligent_memory_save(f"I'm {name_input}", "provide_name", {})
                for mem_op in smart_memories:
                    save_memory(mem_op["key"], mem_op["value"], mem_op["confidence"])
                
                # Update memory context
                memory_context = get_all_memory()
                user_name = memory_context.get("name", "friend")
                self.dynamic_speak(f"Nice to meet you, {user_name}!", "happy", "greeting")
        else:
            self.dynamic_speak(f"Hello {user_name}! I'm ready to chat!", "friendly", "greeting")
        
        print("\n🤖 BUDDY Voice Assistant is ready!")
        print("💡 Tips:")
        print("   - Speak naturally and clearly")
        print("   - Say 'bye' to exit")
        print("   - I'll remember our conversations!")
        print("\n" + "="*50)
        
        try:
            while True:
                # Listen for user input
                user_input = self.continuous_listen()
                
                if not user_input:
                    continue
                
                # Get response from BUDDY
                reply, emotion, intent = self.ask_buddy(user_input)
                
                # Speak the response with dynamic voice
                self.dynamic_speak(reply, emotion, intent)
                
        except KeyboardInterrupt:
            self.dynamic_speak("Goodbye! It was nice talking with you!", "friendly", "farewell")
        except Exception as e:
            print(f"Error: {e}")
            self.dynamic_speak("I'm having some technical difficulties. Let's try again later!", "neutral")

if __name__ == "__main__":
    buddy = DynamicVoiceBuddy()
    buddy.run()
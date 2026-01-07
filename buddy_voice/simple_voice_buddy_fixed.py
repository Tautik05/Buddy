import subprocess
import os
import json
import pyaudio
import numpy as np
import pyttsx3
import time
import speech_recognition as sr
from simple_memory import init_db, save_memory, get_memory, get_all_memory
from speaker_recognition import SpeakerRecognition
from enroll_speaker import record_audio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(BASE_DIR, "buddy_prompt.txt")

# Initialize memory system
init_db()

# Check if prompt file exists and read it
if os.path.exists(PROMPT_PATH):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    print(f"Warning: {PROMPT_PATH} not found, using default prompt")
    SYSTEM_PROMPT = "You are Buddy, a helpful AI assistant."

class VoiceBuddy:
    def __init__(self):
        # Initialize Google Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize speaker recognition for voice identification
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        us_model = os.path.join(parent_dir, "vosk-model-small-en-us-0.15")
        self.speaker_sr = SpeakerRecognition(us_model)
        self.speaker_sr.load_speakers()
        
        # Initialize TTS
        self.tts = pyttsx3.init()
        voices = self.tts.getProperty('voices')
        if voices:
            self.tts.setProperty('voice', voices[0].id)
        self.tts.setProperty('rate', 150)
        self.tts.setProperty('volume', 1.0)
        
        # Calibrate microphone for better sensitivity
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            # Adjust energy threshold for better detection
            self.recognizer.energy_threshold = 300
            self.recognizer.pause_threshold = 2.0  # Wait 2 seconds of silence before stopping
            self.recognizer.dynamic_energy_threshold = True  # Automatically adjust to background noise
        print("Google Speech Recognition initialized")
        
    def speak(self, text, emotion="neutral", intent="unknown"):
        """Convert text to speech with emotion and intent"""
        if not text or text.strip() == "":
            return
            
        try:
            # Try Windows SAPI first with emotion and intent
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            
            # Adjust voice settings based on emotion and intent
            if intent == "ask_question" or "?" in text:
                speaker.Rate = 1  # Slightly faster for questions
                # Add slight pause before questions
                time.sleep(0.2)
            elif emotion == "happy" or emotion == "excited":
                speaker.Rate = 2  # Faster and energetic
            elif emotion == "sad" or emotion == "tired":
                speaker.Rate = -2  # Slower and subdued
            elif emotion == "curious":
                speaker.Rate = 1  # Slightly faster for curiosity
            elif intent == "greeting":
                speaker.Rate = 0  # Normal, warm greeting
            else:
                speaker.Rate = 0  # Normal
            
            speaker.Speak(text)
        except Exception as e:
            try:
                # Fallback to pyttsx3 with emotion and intent
                if intent == "ask_question" or "?" in text:
                    self.tts.setProperty('rate', 160)
                elif emotion == "happy" or emotion == "excited":
                    self.tts.setProperty('rate', 180)
                elif emotion == "sad" or emotion == "tired":
                    self.tts.setProperty('rate', 120)
                elif emotion == "curious":
                    self.tts.setProperty('rate', 160)
                else:
                    self.tts.setProperty('rate', 150)
                    
                self.tts.say(text)
                self.tts.runAndWait()
            except Exception as e2:
                pass
        time.sleep(0.3)
    
    def listen_for_speech_dynamic(self, timeout=30):
        """Dynamic listening that detects when you stop speaking"""
        try:
            with self.microphone as source:
                print("Listening... (speak naturally, I'll wait for you to finish)")
                
                # Start listening without time limit on phrase
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=None  # No limit - wait for natural pause
                )
                
            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            
            if text:
                print(f"Complete speech: '{text}'")
                speaker, confidence = self.identify_speaker_from_audio(audio)
                return text, speaker, confidence
            
            return "", "Unknown", 0.0
            
        except sr.WaitTimeoutError:
            print("No speech detected")
            return "", "Unknown", 0.0
        except sr.UnknownValueError:
            print("Could not understand audio")
            return "", "Unknown", 0.0
        except sr.RequestError as e:
            print(f"Error with speech recognition: {e}")
            return "", "Unknown", 0.0
    
    def identify_speaker_from_audio(self, audio):
        """Identify speaker from Google SR audio data"""
        try:
            # Convert audio to numpy array for speaker recognition
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            audio_array = audio_data.astype(np.float32) / 32768.0
            
            if len(audio_array) > 8000:  # Need enough samples
                speaker_info = self.speaker_sr.identify_speaker(audio_array)
                if isinstance(speaker_info, tuple):
                    return speaker_info
                else:
                    return speaker_info, 0.0
        except Exception as e:
            print(f"Speaker identification error: {e}")
        
        return "Unknown", 0.0
    
    def get_name_from_speech(self):
        """Get name with confirmation"""
        for attempt in range(3):
            self.speak("Please tell me your name clearly.")
            text, _, _ = self.listen_for_speech(timeout=8)
            
            print(f"Name attempt {attempt+1}: '{text}'")  # Debug
            
            if text:
                words = text.lower().split()
                skip_words = ['my', 'name', 'is', 'i', 'am', 'call', 'me']
                name = None
                
                for word in words:
                    if word not in skip_words and len(word) > 1:
                        name = word.capitalize()
                        break
                
                if name:
                    self.speak(f"Did you say your name is {name}? Say yes or no.")
                    response, _, _ = self.listen_for_speech(timeout=5)
                    
                    print(f"Confirmation: '{response}'")  # Debug
                    
                    if response and 'yes' in response.lower():
                        return name
                    
            self.speak("I didn't understand. Let me try again.")
        
        return None
    
    def enroll_new_user(self):
        """Enroll new speaker with voice recording"""
        self.speak("Hello! I don't recognize your voice yet.")
        
        name = self.get_name_from_speech()
        if not name:
            self.speak("I'm having trouble understanding your name. Let's try later.")
            return None
            
        self.speak(f"Nice to meet you, {name}! I'll record your voice for 10 seconds. Please speak naturally about anything.")
        
        audio_file = f"{name}_sample.wav"
        record_audio(audio_file, duration=10)
        
        self.sr.enroll_speaker(name, audio_file)
        self.sr.save_speakers()
        
        save_memory("name", name)
        
        self.speak(f"Perfect! I've learned your voice, {name}!")
        return name

def ask_buddy(user_input):
    # Get current memory context
    memory_context = get_all_memory()
    name_known = "name" in memory_context or "user_name" in memory_context

    # Build context-aware prompt
    context = f"Current memory: {json.dumps(memory_context)}\\nname_known: {name_known}\\n"
    prompt = SYSTEM_PROMPT + "\\n" + context + "\\nUser: " + user_input + "\\nAssistant:"

    print(f"Calling Ollama with: {user_input}")
    
    result = subprocess.run(
        [r"C:\Users\Debanshu\AppData\Local\Programs\Ollama\ollama.exe", "run", "llama3.2:3b"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )

    response = result.stdout.strip()
    print(f"Ollama response: '{response}'")
    
    if result.stderr:
        print(f"Ollama error: {result.stderr}")

    # Process memory operations if JSON response
    try:
        data = json.loads(response)
        
        normalized = {
            "reply": data.get("reply", ""),
            "emotion": data.get("emotion", "neutral"),
            "intent": data.get("intent", "unknown"),
            "memory": data.get("memory", {
                "store": False,
                "key": "",
                "value": "",
                "confidence": 0.0
            })
        }
        
        # Store memory if needed
        memory = normalized["memory"]
        if memory.get("store") and memory.get("key") and memory.get("value"):
            save_memory(memory["key"], memory["value"], memory.get("confidence", 0.7))
        
        return normalized["reply"]
        
    except Exception as e:
        print(f"JSON parse error: {e}")
        return response if response else "I'm having trouble thinking right now."

def run_voice_buddy():
    """Main voice interaction loop"""
    buddy = VoiceBuddy()
    buddy.speak("Hello! I'm Buddy, your personal AI assistant. How are you today?!")
    
    try:
        while True:
            text, speaker, confidence = buddy.listen_for_speech_dynamic()
            
            if not text:
                buddy.speak("I didn't hear anything. Try speaking again.")
                continue
                
            print(f"Heard: '{text}' from {speaker} (confidence: {confidence:.2f})")
            
            # Check if it's a name introduction
            if "my name is" in text.lower():
                # Extract name and enroll
                words = text.lower().replace("my name is", "").strip().split()
                if words:
                    name = words[0].capitalize()
                    buddy.speak(f"Nice to meet you, {name}! Let me record your voice for 10 seconds.")
                    
                    # Record and enroll
                    audio_file = f"{name}_sample.wav"
                    record_audio(audio_file, duration=10)
                    buddy.speaker_sr.enroll_speaker(name, audio_file)
                    buddy.speaker_sr.save_speakers()
                    save_memory("name", name)
                    save_memory("user_name", name)
                    
                    buddy.speak(f"Perfect! I've learned your voice, {name}!")
                    continue
            
            # Check if speaker is unknown (only if we have enrolled speakers)
            if speaker == "Unknown" and buddy.speaker_sr.speaker_embeddings:
                buddy.speak("I don't recognize your voice. Please say 'My name is' followed by your name.")
                continue
            
            # Process with Buddy AI (pass speaker name for personalization)
            response = ask_buddy(f"Speaker: {speaker}. {text}")
            
            # Extract emotion, intent and reply from AI response
            try:
                response_data = json.loads(response)
                reply = response_data.get("reply", response)
                emotion = response_data.get("emotion", "neutral")
                intent = response_data.get("intent", "unknown")
            except:
                reply = response
                emotion = "neutral"
                intent = "unknown"
            
            buddy.speak(reply, emotion, intent)
            
            # Exit condition
            if any(word in text.lower() for word in ["goodbye", "bye", "exit", "quit"]):
                break
            
    except KeyboardInterrupt:
        buddy.speak("Goodbye!")

if __name__ == "__main__":
    run_voice_buddy()
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
            self.recognizer.energy_threshold = 300
            self.recognizer.pause_threshold = 1.0  # Reduce to 1 second for faster response
            self.recognizer.dynamic_energy_threshold = True
        print("Google Speech Recognition initialized")
        
    def speak(self, text, emotion="neutral", intent="unknown"):
        """Convert text to speech with fallback options"""
        if not text or text.strip() == "":
            return
            
        print(f"Buddy: {text}")  # Always print what we're trying to say
        
        try:
            # Simple pyttsx3 approach first
            self.tts.say(text)
            self.tts.runAndWait()
            return
        except Exception as e:
            print(f"TTS Error: {e}")
            
        try:
            # Try Windows SAPI as backup
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        except Exception as e2:
            print(f"SAPI Error: {e2}")
            print("Speech synthesis failed - check audio drivers")
    
    def listen_for_speech_dynamic(self, timeout=15):
        """Fast dynamic listening"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
                
            print("Processing...")
            text = self.recognizer.recognize_google(audio)
            
            if text:
                print(f"You said: '{text}'")
                speaker, confidence = self.identify_speaker_from_audio(audio)
                return text, speaker, confidence
            
            return "", "Unknown", 0.0
            
        except sr.WaitTimeoutError:
            return "", "Unknown", 0.0
        except sr.UnknownValueError:
            print("Didn't catch that")
            return "", "Unknown", 0.0
        except sr.RequestError as e:
            print(f"Speech error: {e}")
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
                words = text.lower().replace("my name is", "").strip().split()
                if words:
                    name = words[0].capitalize()
                    buddy.speak(f"Nice to meet you, {name}! Let me record your voice for 10 seconds.")
                    
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
            
            # Process with Buddy AI
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
    run_voice_buddy()ction
            if "my name is" in text.lower():
                words = text.lower().replace("my name is", "").strip().split()
                if words:
                    name = words[0].capitalize()
                    buddy.speak(f"Nice to meet you, {name}! Let me record your voice for 10 seconds.")
                    
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
            
            # Process with Buddy AI
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
import subprocess
import os
import json
import pyaudio
import numpy as np
import pyttsx3
import time
from simple_memory import init_db, save_memory, get_memory, get_all_memory
from speaker_recognition import SpeakerRecognition
from enroll_speaker import record_audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "buddy_prompt.txt")

# Initialize memory system
init_db()

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

class VoiceBuddy:
    def __init__(self):
        self.sr = SpeakerRecognition("vosk-model-small-en-us-0.15")
        self.sr.load_speakers()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 140)
        
        # Audio settings
        self.sample_rate = 16000
        self.p = pyaudio.PyAudio()
        
    def speak(self, text):
        """Convert text to speech"""
    self.tts.say(text)
    self.tts.runAndWait()
    time.sleep(0.3)
    
    def listen_for_speech(self, timeout=15):
    """Listen for speech with timeout"""
    stream = self.p.open(format=pyaudio.paInt16, channels=1,
    rate=self.sample_rate, input=True, frames_per_buffer=4000)
        
    audio_buffer = []
    start_time = time.time()
        
    try:
    while time.time() - start_time < timeout:
    data = stream.read(4000, exception_on_overflow=False)
    audio_buffer.extend(np.frombuffer(data, dtype=np.int16))
                
    if self.sr.rec.AcceptWaveform(data):
    result = json.loads(self.sr.rec.Result())
    text = result.get('text', '').strip()
                    
    if text and len(audio_buffer) > self.sample_rate:
    audio_array = np.array(audio_buffer[-self.sample_rate:], dtype=np.float32) / 32768.0
    speaker_info = self.sr.identify_speaker(audio_array)
                        
    if isinstance(speaker_info, tuple):
    speaker, confidence = speaker_info
    else:
    speaker, confidence = speaker_info, 0.0
                            
    return text, speaker, confidence
            
    return "", "Unknown", 0.0
                        
    finally:
    stream.stop_stream()
    stream.close()
    
    def get_name_from_speech(self):
    """Get name with confirmation"""
    for attempt in range(3):
    self.speak("Please tell me your name clearly.")
    text, _, _ = self.listen_for_speech(timeout=8)
            
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
                    
    if 'yes' in response.lower():
    return name
                    
    self.speak("I didn't understand. Let me try again.")
        
    return None
    
    def enroll_new_user(self):
    """Enroll new speaker"""
    self.speak("Hello! I don't recognize your voice yet.")
        
    name = self.get_name_from_speech()
    if not name:
    self.speak("I'm having trouble understanding your name. Let's try later.")
    return None
            
    self.speak(f"Nice to meet you, {name}! I'll record your voice for 10 seconds.")
        
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

    result = subprocess.run(
    [r"C:\Users\Debanshu\AppData\Local\Programs\Ollama\ollama.exe", "run", "llama3.2:3b"],
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
        
    except Exception:
    return response

    def run_voice_buddy():
    """Main voice interaction loop"""
    buddy = VoiceBuddy()
    buddy.speak("Hello! I'm Buddy, your AI assistant. Say something to start!")
    
    try:
    while True:
    text, speaker, confidence = buddy.listen_for_speech()
            
    if not text:
     buddy.speak("I didn't hear anything. Try speaking again.")
     continue
                
            # Check if speaker is unknown
    if speaker == "Unknown" or confidence < 0.3:
    name = buddy.enroll_new_user()
    if not name:
    continue
    speaker = name
            
            # Process with Buddy AI
            response = ask_buddy(text)
            buddy.speak(response)
            
            # Exit condition
            if any(word in text.lower() for word in ["goodbye", "bye", "exit", "quit"]):
                break
            
    except KeyboardInterrupt:
        buddy.speak("Goodbye!")
    finally:
        buddy.p.terminate()

if __name__ == "__main__":
    run_voice_buddy()
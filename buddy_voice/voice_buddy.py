import pyaudio
import json
import numpy as np
import pyttsx3
import wave
from speaker_recognition import SpeakerRecognition
from enroll_speaker import record_audio
from buddy_brain import ask_buddy
from memory import save_memory, get_memory

class VoiceBuddy:
    def __init__(self):
        self.sr = SpeakerRecognition("vosk-model-small-en-us-0.15")
        self.sr.load_speakers()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        
        # Audio settings
        self.sample_rate = 16000
        self.p = pyaudio.PyAudio()
        
    def speak(self, text):
        """Convert text to speech"""
        print(f"BUDDY: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
    
    def listen_for_speech(self):
        """Listen for speech and return text + speaker info"""
        stream = self.p.open(format=pyaudio.paInt16, channels=1, 
                           rate=self.sample_rate, input=True, frames_per_buffer=4000)
        
        print("Listening...")
        audio_buffer = []
        
        try:
            while True:
                data = stream.read(4000, exception_on_overflow=False)
                audio_buffer.extend(np.frombuffer(data, dtype=np.int16))
                
                if self.sr.rec.AcceptWaveform(data):
                    result = json.loads(self.sr.rec.Result())
                    text = result.get('text', '')
                    
                    if text and len(audio_buffer) > self.sample_rate:
                        audio_array = np.array(audio_buffer[-self.sample_rate:], dtype=np.float32) / 32768.0
                        speaker_info = self.sr.identify_speaker(audio_array)
                        
                        if isinstance(speaker_info, tuple):
                            speaker, confidence = speaker_info
                        else:
                            speaker, confidence = speaker_info, 0.0
                            
                        return text, speaker, confidence
                        
        finally:
            stream.stop_stream()
            stream.close()
    
    def enroll_new_user(self):
        """Enroll a new speaker"""
        self.speak("I don't recognize your voice. What's your name?")
        
        # Get name via speech
        text, _, _ = self.listen_for_speech()
        name = text.strip()
        
        if not name:
            self.speak("I didn't catch that. Please try again.")
            return None
            
        self.speak(f"Nice to meet you {name}! I'll record your voice for 10 seconds. Please speak clearly.")
        
        # Record enrollment audio
        audio_file = f"{name}_sample.wav"
        record_audio(audio_file, duration=10)
        
        # Enroll speaker
        self.sr.enroll_speaker(name, audio_file)
        self.sr.save_speakers()
        
        # Save to memory
        save_memory("user_name", name)
        
        self.speak(f"Great! I've registered your voice, {name}. Now I'll remember you!")
        return name
    
    def run(self):
        """Main voice interaction loop"""
        self.speak("Hello! I'm Buddy, your AI assistant. Please say something.")
        
        try:
            while True:
                text, speaker, confidence = self.listen_for_speech()
                
                if not text:
                    continue
                    
                print(f"Heard: '{text}' from {speaker} (confidence: {confidence:.2f})")
                
                # Check if speaker is unknown
                if speaker == "Unknown" or confidence < 0.5:
                    name = self.enroll_new_user()
                    if not name:
                        continue
                    speaker = name
                
                # Update current user in memory
                save_memory("current_user", speaker)
                
                # Process with Buddy AI
                response = ask_buddy(text)
                
                # Extract reply from JSON response if needed
                try:
                    response_data = json.loads(response)
                    reply = response_data.get("reply", response)
                except:
                    reply = response
                
                self.speak(reply)
                
        except KeyboardInterrupt:
            self.speak("Goodbye!")
        finally:
            self.p.terminate()

if __name__ == "__main__":
    buddy = VoiceBuddy()
    buddy.run()
"""
SIMPLE BUDDY - Stable Version
Removes complex features, focuses on core functionality
"""

import cv2
import time
import requests
import speech_recognition as sr
import edge_tts
import asyncio
import pygame
import threading
from config import Config

class SimpleBuddy:
    def __init__(self):
        self.config = Config.from_env()
        self.llm_url = "http://localhost:8000"
        
        # Simple camera
        self.cap = cv2.VideoCapture(0)
        
        # Simple speech
        self.speech = sr.Recognizer()
        self.mic = sr.Microphone()
        pygame.mixer.init()
        
        print("🤖 Simple Buddy Ready!")
    
    def speak(self, text):
        """Simple TTS"""
        try:
            async def _speak():
                communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                
                if audio_data:
                    import io
                    pygame.mixer.music.load(io.BytesIO(audio_data))
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)
            
            asyncio.run(_speak())
        except:
            print(f"TTS: {text}")
    
    def listen(self):
        """Simple speech recognition"""
        try:
            with self.mic as source:
                print("🎤 Listening...")
                audio = self.speech.listen(source, timeout=5)
            
            text = self.speech.recognize_google(audio, language='en-IN')
            print(f"You said: '{text}'")
            return text
        except:
            return ""
    
    def chat(self, text):
        """Simple chat with LLM"""
        try:
            response = requests.post(f"{self.llm_url}/chat", 
                json={"user_input": text}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("reply", "Sorry, I didn't understand.")
            else:
                return "I'm having trouble thinking."
        except:
            return "Connection error."
    
    def run(self):
        """Simple main loop"""
        self.speak("Hello! I'm Simple Buddy!")
        
        while True:
            try:
                # Show camera
                ret, frame = self.cap.read()
                if ret:
                    cv2.imshow('Simple Buddy', frame)
                
                # Listen and respond
                user_text = self.listen()
                if user_text:
                    if "goodbye" in user_text.lower():
                        self.speak("Goodbye!")
                        break
                    
                    reply = self.chat(user_text)
                    self.speak(reply)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Simple Buddy: Goodbye! 👋")

if __name__ == "__main__":
    buddy = SimpleBuddy()
    buddy.run()
import pyaudio
import wave
import numpy as np
from speaker_recognition import SpeakerRecognition

def record_audio(filename, duration=10, sample_rate=16000):
    """Record audio for speaker enrollment"""
    p = pyaudio.PyAudio()
    
    print(f"Recording {duration} seconds for {filename}...")
    print("Speak clearly and naturally...")
    
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                   input=True, frames_per_buffer=1024)
    
    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save audio file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Audio saved as {filename}")

def enroll_new_speaker():
    """Enroll a new speaker"""
    speaker_name = input("Enter speaker name: ")
    audio_file = f"{speaker_name}_sample.wav"
    
    # Record audio
    record_audio(audio_file)
    
    # Initialize speaker recognition system
    sr = SpeakerRecognition("vosk-model-small-en-us-0.15")
    sr.load_speakers()
    
    # Enroll speaker
    sr.enroll_speaker(speaker_name, audio_file)
    sr.save_speakers()
    
    print(f"Speaker {speaker_name} enrolled successfully!")

if __name__ == "__main__":
    enroll_new_speaker()
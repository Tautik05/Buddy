import vosk
import json
import pyaudio
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class SpeakerRecognition:
    def __init__(self, vosk_model_path, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model = vosk.Model(vosk_model_path)
        self.rec = vosk.KaldiRecognizer(self.model, sample_rate)
        self.speaker_embeddings = {}
        self.threshold = 0.5
        
    def extract_features(self, audio_data):
        """Extract MFCC features from audio"""
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    
    def enroll_speaker(self, speaker_name, audio_file_path):
        """Enroll a new speaker with reference audio"""
        audio, _ = librosa.load(audio_file_path, sr=self.sample_rate)
        features = self.extract_features(audio)
        self.speaker_embeddings[speaker_name] = features
        print(f"Speaker {speaker_name} enrolled successfully")
    
    def identify_speaker(self, audio_data):
        """Identify speaker from audio data"""
        if not self.speaker_embeddings:
            return "Unknown"
            
        features = self.extract_features(audio_data)
        best_match = "Unknown"
        best_score = 0
        
        for speaker, ref_features in self.speaker_embeddings.items():
            similarity = cosine_similarity([features], [ref_features])[0][0]
            if similarity > best_score and similarity > self.threshold:
                best_score = similarity
                best_match = speaker
                
        return best_match, best_score
    
    def save_speakers(self, filepath="speakers.pkl"):
        """Save enrolled speakers"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.speaker_embeddings, f)
    
    def load_speakers(self, filepath="speakers.pkl"):
        """Load enrolled speakers"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.speaker_embeddings = pickle.load(f)
    
    def real_time_recognition(self):
        """Real-time speech and speaker recognition"""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
                       input=True, frames_per_buffer=4000)
        
        print("Listening... Press Ctrl+C to stop")
        audio_buffer = []
        
        try:
            while True:
                data = stream.read(4000, exception_on_overflow=False)
                audio_buffer.extend(np.frombuffer(data, dtype=np.int16))
                
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get('text', '')
                    
                    if text and len(audio_buffer) > self.sample_rate:  # 1 second of audio
                        audio_array = np.array(audio_buffer[-self.sample_rate:], dtype=np.float32) / 32768.0
                        speaker, confidence = self.identify_speaker(audio_array)
                        print(f"Speaker: {speaker} ({confidence:.2f}) | Text: {text}")
                        audio_buffer = []
                        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    # Initialize system
    sr = SpeakerRecognition("vosk-model-en-us-0.22")  # Use larger model
    
    # Load existing speakers
    sr.load_speakers()
    
    # Enroll speakers (run once per speaker)
    # sr.enroll_speaker("John", "john_sample.wav")
    # sr.enroll_speaker("Jane", "jane_sample.wav")
    # sr.save_speakers()
    
    # Start real-time recognition
    sr.real_time_recognition()
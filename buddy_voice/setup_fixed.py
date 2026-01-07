import subprocess
import sys
import urllib.request
import zipfile
import os

def install_requirements():
    packages = ["vosk", "pyaudio", "librosa", "scikit-learn", "numpy"]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"OK {package} installed")

def download_vosk_model():
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    model_zip = "vosk-model.zip"
    
    if not os.path.exists("vosk-model-small-en-us-0.15"):
        print("Downloading Vosk model...")
        urllib.request.urlretrieve(model_url, model_zip)
        
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall()
        
        os.remove(model_zip)
        print("OK Vosk model downloaded")
    else:
        print("OK Vosk model already exists")

if __name__ == "__main__":
    install_requirements()
    download_vosk_model()
    print("\nSetup complete! Run: python enroll_speaker.py")
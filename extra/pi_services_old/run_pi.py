"""Run Buddy Pi Service"""

import os
import sys

def check_dependencies():
    required = ['cv2', 'numpy', 'requests', 'speech_recognition', 'edge_tts', 'pygame']
    missing = []
    
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install opencv-python numpy requests SpeechRecognition edge-tts pygame pyaudio")
        return False
    return True

def check_models():
    if not os.path.exists('models/MobileFaceNet.onnx'):
        print("Missing models/MobileFaceNet.onnx")
        return False
    if not os.path.exists('models/yolov8n.pt'):
        print("Missing models/yolov8n.pt")
        return False
    return True

def check_files():
    required_files = ['config.py', 'states.py', 'face_detector.py', 'memory.py']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f"Missing files: {', '.join(missing)}")
        print("Copy these from main Buddy folder")
        return False
    return True

if __name__ == "__main__":
    print("Starting Buddy Pi...")
    
    if not check_dependencies():
        sys.exit(1)
    
    if not check_models():
        sys.exit(1)
    
    if not check_files():
        sys.exit(1)
    
    # Set default LLM service URL if not set
    if 'LLM_SERVICE_URL' not in os.environ:
        os.environ['LLM_SERVICE_URL'] = 'http://localhost:8000'
        print(f"Using default LLM service: {os.environ['LLM_SERVICE_URL']}")
    
    try:
        from buddy_pi_main import main
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)
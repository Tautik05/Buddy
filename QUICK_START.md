# Buddy AI - Quick Start

## 1. LLM Service (Powerful Machine)

```bash
cd llm_service/
pip install fastapi uvicorn requests
ollama pull llama3.2:3b
python main.py
```

## 2. Raspberry Pi Setup

```bash
cd pi_services/
pip install opencv-python numpy requests SpeechRecognition edge-tts pygame pyaudio onnxruntime ultralytics
mkdir models
# Copy MobileFaceNet.onnx and yolov8n.pt to models/
# Copy config.py, states.py, face_detector.py, etc. from main Buddy folder
export LLM_SERVICE_URL="http://YOUR-SERVER-IP:8000"
python buddy_pi_main.py
```

## Files to Copy to Pi:
- config.py
- states.py  
- face_detector.py
- stability_tracker.py
- input_handler.py
- sleep_wake_manager.py
- memory.py
- objrecog/ folder (if using original object detection)

## Network:
- Both devices on same network
- LLM service accessible on port 8000
- Update LLM_SERVICE_URL environment variable
# Buddy AI - Distributed Architecture

A distributed AI companion system with face recognition, voice interaction, and object detection.

## 🏗️ Architecture

**Two-Service Design:**
- **LLM Service** (Server): Handles all AI processing, memory, and conversations
- **Pi Service** (Client): Handles hardware - camera, face recognition, voice I/O, object detection

## 📁 Project Structure

```
Buddy/
├── llm_service/          # AI Brain Service (FastAPI)
│   ├── main.py          # Main FastAPI application
│   ├── memory.py        # Database and memory management
│   ├── smart_memory.py  # Intelligent memory extraction
│   ├── run_llm.py       # Service launcher
│   └── requirements.txt # Python dependencies
│
├── pi_services/         # Hardware Service (Raspberry Pi)
│   ├── models/          # AI models (face recognition, YOLO)
│   ├── objrecog/        # Object detection module
│   ├── clean_buddy_pi.py # Main Pi service
│   ├── face_detector.py # Face detection
│   ├── face_recognizer.py # Face recognition
│   ├── pi_memory.py     # Database access for Pi
│   ├── config.py        # Configuration
│   ├── states.py        # State management
│   ├── stability_tracker.py # Face tracking stability
│   └── requirements.txt # Python dependencies
│
└── extra/               # Archive of old/unused files
    ├── face-recog/      # Old face recognition system
    ├── objrecog/        # Old object detection
    ├── pi_services_old/ # Old Pi service files
    └── *.py             # Old monolithic system files
```

## 🚀 Quick Start

### 1. Start LLM Service (Server)
```bash
cd llm_service
python run_llm.py
```

### 2. Start Pi Service (Client)
```bash
cd pi_services
python clean_buddy_pi.py
```

## 🔧 Configuration

- **Database**: Uses Neon DB (cloud PostgreSQL) for face recognition data
- **LLM**: Uses Ollama with llama3.2:3b model
- **Face Recognition**: TensorFlow Lite MobileFaceNet (192-dim embeddings)
- **Object Detection**: YOLO TensorFlow Lite
- **Speech**: Azure Edge TTS with Indian English support

## 📋 Requirements

- Python 3.8+
- Ollama (for LLM service)
- PostgreSQL database (Neon DB)
- Camera (for Pi service)
- Microphone and speakers (for voice interaction)

## 🎯 Features

- **Face Recognition**: Automatic face detection and recognition with database storage
- **Voice Interaction**: Speech-to-text and text-to-speech with natural conversation
- **Object Detection**: Real-time object detection and description
- **Memory System**: Intelligent memory extraction and storage
- **Distributed**: Separate AI brain and hardware components

## 📝 Notes

- The `extra/` folder contains archived files from the old monolithic system
- All active development should use the distributed architecture
- Face recognition uses 192-dimensional embeddings (TFLite model)
- Object detection is optimized for common household items
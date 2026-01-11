# Buddy AI - Distributed Architecture

This is the distributed version of Buddy AI, split into separate components for Raspberry Pi deployment.

## Architecture

```
┌─────────────────┐    HTTP API    ┌──────────────────┐
│  Raspberry Pi   │◄──────────────►│   LLM Server     │
│                 │                │                  │
│ • Face Recognition              │ • FastAPI Service │
│ • Object Detection              │ • Ollama LLM      │
│ • Voice I/O                     │ • Memory Context  │
│ • Camera/Sensors                │ • Prompt Processing│
└─────────────────┘                └──────────────────┘
```

## Components

### 1. LLM Service (`llm_service/`)
- **Purpose**: Handles all AI processing
- **Runs on**: Powerful machine (desktop/server)
- **Components**:
  - FastAPI web service
  - Ollama integration
  - Memory context management
  - Response parsing

### 2. Pi Services (`pi_services/`)
- **Purpose**: Local hardware interaction
- **Runs on**: Raspberry Pi
- **Components**:
  - Face recognition (ONNX optimized)
  - Object detection (YOLOv8 nano)
  - Voice input/output (Speech Recognition + Edge TTS)
  - Camera processing

## Deployment

### Step 1: Set up LLM Service (on powerful machine)

```bash
cd llm_service/
chmod +x setup_llm.sh
./setup_llm.sh
./start_llm_service.sh
```

The service will be available at `http://localhost:8000`

### Step 2: Set up Raspberry Pi

```bash
# Copy pi_services folder to your Raspberry Pi
scp -r pi_services/ pi@your-pi-ip:~/buddy_pi/

# SSH into Pi and run setup
ssh pi@your-pi-ip
cd buddy_pi/
chmod +x setup_pi.sh
./setup_pi.sh

# Update the LLM service URL
nano start_buddy.sh
# Change: export LLM_SERVICE_URL="http://YOUR-SERVER-IP:8000"

# Start Buddy
./start_buddy.sh
```

## Configuration

### Environment Variables

**LLM Service:**
- `OLLAMA_HOST`: Ollama server host (default: localhost:11434)

**Pi Service:**
- `LLM_SERVICE_URL`: URL of LLM service (e.g., http://192.168.1.100:8000)
- `BUDDY_CAMERA_INDEX`: Camera index (default: 0)
- `BUDDY_LOG_LEVEL`: Logging level (default: WARNING)

### Network Requirements

- Both devices must be on the same network
- LLM service port 8000 must be accessible from Pi
- Stable network connection for real-time communication

## API Endpoints

### POST /chat
Process chat request with context

**Request:**
```json
{
  "user_input": "Hello there!",
  "recognized_user": "John",
  "memory_context": {"favorite_color": "blue"},
  "session_context": {"conversations": []},
  "objects_visible": ["chair", "laptop"]
}
```

**Response:**
```json
{
  "reply": "Hey John! How's it going?",
  "intent": "greeting",
  "emotion": "cheerful",
  "raw_response": "{\"reply\": \"Hey John! How's it going?\", \"intent\": \"greeting\", \"emotion\": \"cheerful\"}"
}
```

### GET /health
Health check endpoint

## Performance Optimizations

### Raspberry Pi Optimizations:
- ONNX Runtime CPU-only execution
- YOLOv8 nano model (smallest)
- Reduced frame processing rate
- Optimized object detection frequency
- Compressed image processing

### LLM Service Optimizations:
- Async FastAPI for concurrent requests
- Connection pooling
- Response caching (can be added)
- Timeout handling

## Troubleshooting

### Common Issues:

1. **Pi can't connect to LLM service**
   - Check network connectivity: `ping YOUR-SERVER-IP`
   - Verify LLM service is running: `curl http://YOUR-SERVER-IP:8000/health`
   - Check firewall settings

2. **Face recognition not working**
   - Ensure MobileFaceNet.onnx is in models/ directory
   - Check camera permissions
   - Verify ONNX Runtime installation

3. **Object detection slow**
   - Reduce camera resolution in config.py
   - Increase frame processing interval
   - Use YOLOv8n (nano) model only

4. **Speech recognition issues**
   - Check microphone permissions
   - Install portaudio: `sudo apt install portaudio19-dev`
   - Test with: `python -c "import speech_recognition; print('OK')"`

### Performance Monitoring:

```bash
# Check Pi resources
htop

# Monitor network traffic
iftop

# Check service logs
journalctl -u buddy -f
```

## Development

### Adding New Features:

1. **New intents**: Update buddy_prompt.txt and intent handling
2. **New objects**: Modify target_classes in pi_object_detector.py
3. **New voice commands**: Add patterns in LLM service
4. **Hardware integration**: Add to Pi service main loop

### Testing:

```bash
# Test LLM service
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello!"}'

# Test Pi components individually
python pi_face_recognizer.py
python pi_object_detector.py
```

## Security Notes

- LLM service runs on all interfaces (0.0.0.0) - use firewall
- No authentication implemented - add if needed
- Consider HTTPS for production deployment
- Secure your network and change default passwords

## License

Same as original Buddy project.
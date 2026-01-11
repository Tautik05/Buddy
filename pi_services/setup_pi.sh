#!/bin/bash

# Buddy AI Deployment Script for Raspberry Pi

echo "🤖 Setting up Buddy AI on Raspberry Pi..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3-pip python3-venv git cmake build-essential
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y espeak espeak-data libespeak1 libespeak-dev

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv buddy_env
source buddy_env/bin/activate

# Install Python packages
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install ONNX Runtime for ARM
echo "🧠 Installing ONNX Runtime for ARM..."
pip install onnxruntime

# Copy model files
echo "📁 Setting up model files..."
mkdir -p models
cp ../face-recog/MobileFaceNet.onnx models/
cp ../yolov8n.pt models/

# Set up configuration
echo "⚙️ Setting up configuration..."
cp ../config.py .
cp ../states.py .
cp ../face_detector.py .
cp ../stability_tracker.py .
cp ../input_handler.py .
cp ../sleep_wake_manager.py .
cp ../memory.py .

# Copy object recognition modules
mkdir -p objrecog
cp ../objrecog/__init__.py objrecog/
cp ../objrecog/obj.py objrecog/
cp ../objrecog/perception.py objrecog/

# Create startup script
cat > start_buddy.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source buddy_env/bin/activate
export LLM_SERVICE_URL="http://192.168.1.100:8000"  # Change this to your LLM server IP
python buddy_pi_main.py
EOF

chmod +x start_buddy.sh

# Create systemd service (optional)
cat > buddy.service << 'EOF'
[Unit]
Description=Buddy AI Assistant
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/buddy_pi
ExecStart=/home/pi/buddy_pi/start_buddy.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Raspberry Pi setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update LLM_SERVICE_URL in start_buddy.sh with your server IP"
echo "2. Run: ./start_buddy.sh"
echo "3. Optional: sudo cp buddy.service /etc/systemd/system/ && sudo systemctl enable buddy"
echo ""
echo "🎯 Make sure your LLM service is running on the server!"
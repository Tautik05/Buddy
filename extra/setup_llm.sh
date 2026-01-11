#!/bin/bash

# Buddy AI LLM Service Deployment Script

echo "🧠 Setting up Buddy AI LLM Service..."

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv llm_env
source llm_env/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "🦙 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Pull the model
echo "📥 Pulling Llama 3.2 3B model..."
ollama pull llama3.2:3b

# Create startup script
cat > start_llm_service.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source llm_env/bin/activate

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Start FastAPI service
echo "Starting LLM Service on port 8000..."
python main.py
EOF

chmod +x start_llm_service.sh

# Create systemd service
cat > buddy-llm.service << 'EOF'
[Unit]
Description=Buddy AI LLM Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
ExecStart=$PWD/start_llm_service.sh
Restart=always
RestartSec=10
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
EOF

echo "✅ LLM Service setup complete!"
echo ""
echo "📋 To start the service:"
echo "1. Run: ./start_llm_service.sh"
echo "2. Service will be available at http://localhost:8000"
echo "3. Optional: Install as system service with:"
echo "   sudo cp buddy-llm.service /etc/systemd/system/"
echo "   sudo systemctl enable buddy-llm"
echo "   sudo systemctl start buddy-llm"
echo ""
echo "🔗 API endpoints:"
echo "   POST /chat - Main chat endpoint"
echo "   GET /health - Health check"
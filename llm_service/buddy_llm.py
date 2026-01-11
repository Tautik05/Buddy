"""Run Buddy LLM Service"""

import subprocess
import sys
import time

def check_ollama():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'llama3.2:3b' not in result.stdout:
            print("📥 Pulling llama3.2:3b model...")
            subprocess.run(['ollama', 'pull', 'llama3.2:3b'])
        return True
    except FileNotFoundError:
        print("❌ Ollama not found. Install from https://ollama.ai")
        return False

def start_ollama():
    try:
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        print("✅ Ollama server started")
    except:
        print("⚠️ Could not start Ollama server")

if __name__ == "__main__":
    print("🧠 Starting Buddy LLM Service...")
    
    if not check_ollama():
        sys.exit(1)
    
    start_ollama()
    
    try:
        from main import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("❌ Missing dependencies. Run: pip install fastapi uvicorn requests")
        sys.exit(1)
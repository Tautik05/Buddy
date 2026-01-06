# Buddy AI Assistant

A simple AI assistant that uses Ollama and Llama for conversations, with PostgreSQL memory storage via Neon DB.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Neon Database:**
   - Go to [Neon Console](https://console.neon.tech)
   - Create a new project
   - Copy the connection string

3. **Configure Environment:**
   - Copy `.env.example` to `.env`
   - Update `DATABASE_URL` with your Neon connection string
   - Format: `postgresql://username:password@hostname/database`

4. **Run the Assistant:**
   ```bash
   python buddy_brain.py
   ```

## Memory System

The memory system stores key-value pairs in PostgreSQL:

- `save_memory(key, value)` - Store a value
- `get_memory(key)` - Retrieve a value
- `get_all_memory()` - Get all stored memories

## Files

- `buddy_brain.py` - Main assistant logic
- `memory.py` - PostgreSQL memory storage
- `buddy_prompt.txt` - System prompt for the AI
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (configure your database here)
- `.env.example` - Template for environment variables
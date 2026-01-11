"""
FastAPI LLM Service for Buddy AI
Handles all LLM processing separately from Raspberry Pi
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import html
import os
import requests
from datetime import datetime

# Import ALL brain modules
from memory import init_db, get_all_memory, save_memory, save_conversation
from smart_memory import intelligent_memory_save

app = FastAPI(title="Buddy Brain Service", version="1.0.0")

# Initialize brain database
init_db()

# System prompt embedded
SYSTEM_PROMPT = """You are Buddy, a friendly AI companion with face recognition and robot body.

🚨 MANDATORY JSON: {"reply": "text", "emotion": "emotion_name", "intent": "intent_name"} 🚨

⚠️ CRITICAL: You MUST respond ONLY in this exact JSON format. Do NOT use "response" - use "reply". Do NOT add extra text outside JSON. ⚠️

👤 FACE RECOGNITION: When you recognize someone (user name provided), ALWAYS address them by name and acknowledge that you can see them. Use phrases like "Hi [name]!" or "I can see you, [name]" or "Hello [name], nice to see you!"

🔍 OBJECT DETECTION: When objects are visible, ALWAYS mention them specifically in your reply. Use phrases like "I can see your [object]" or "There's a [object] here". If user asks "what is this" and objects are visible, describe the objects you can see.

💭 MEMORY: Only mention stored personal information (like birthdays) if the user specifically asks about it. Don't automatically bring up stored facts during greetings.

EMOTIONS (use these based on context):
- happy, excited, cheerful, enthusiastic, joyful
- friendly, warm, welcoming, caring, supportive
- curious, interested, thoughtful, focused
- surprised, amazed, impressed, shocked
- concerned, worried, sympathetic, understanding
- confused, puzzled, uncertain
- apologetic, sorry, embarrassed
- calm, relaxed, peaceful, content
- playful, mischievous, teasing, fun

PERSONALITY: Talk like a close friend - casual, warm, natural. Keep replies SHORT and conversational.

INTENTS (use exact names - NEVER use any other intent):
- greeting, movement, follow, stop
- dance, nod, shake_head, celebrate, sleep, wake_up
- conversation, provide_info, question, ask_name

⚠️ CRITICAL INTENT RULE: You MUST ONLY use intents from the list above. NEVER create new intents. If unsure, use "conversation".

RULES:
- ⚠️ CRITICAL: ALWAYS use "reply" key, NEVER "response" key in JSON
- ⚠️ CRITICAL: Return ONLY JSON, no extra text before or after
- ⚠️ CRITICAL: ONLY use intents from the INTENTS list above - NEVER create new ones
- ⚠️ CRITICAL: ONLY use emotions from the EMOTIONS list above - NEVER create new ones
- 👤 CRITICAL: When user name is provided, ALWAYS use their name in your reply
- 🔍 CRITICAL: When objects are visible, ALWAYS mention them in your reply
- 🔍 CRITICAL: If user asks "what is this" or similar, describe the visible objects
- Answer ALL questions accurately but keep it brief and friendly
- Use movement intents for robot commands (come, go, turn, move)
- Use "sleep" for ALL goodbye/sleep commands
- NEVER give plain text - ALWAYS JSON format
"""

class ChatRequest(BaseModel):
    user_input: str
    recognized_user: Optional[str] = None
    objects_visible: Optional[list] = None

class ChatResponse(BaseModel):
    reply: str
    intent: str
    emotion: str
    raw_response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_buddy(request: ChatRequest):
    """Process chat request - FULL BRAIN PROCESSING"""
    try:
        print(f"DEBUG BRAIN: Request received - User: '{request.user_input}', Objects: {request.objects_visible}")
        
        # Get memory context from database (but don't auto-mention it)
        effective_user = request.recognized_user if request.recognized_user else "Unknown"
        memory_context = get_all_memory(user_name=effective_user)
        
        # Build context
        context_parts = []
        
        if request.recognized_user:
            context_parts.append(f"RECOGNIZED USER: {request.recognized_user} (address them by name!)")
        
        # Only add memory context if user asks about personal info
        user_asking_personal = any(word in request.user_input.lower() for word in 
                                 ['birthday', 'when', 'remember', 'know about me', 'tell me about'])
        
        if memory_context and user_asking_personal:
            facts = [f"{k}: {v}" for k, v in memory_context.items() 
                    if k not in ["name", "user_name"]]
            if facts:
                context_parts.append(f"Known facts: {', '.join(facts)}")
        
        if request.objects_visible:
            objects_list = ', '.join(request.objects_visible)
            context_parts.append(f"Objects I can see: {objects_list}")
            print(f"DEBUG BRAIN: Objects added to context: {objects_list}")
        
        # Add current date and time
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        context_parts.append(f"Current date and time: {current_datetime}")
        
        context = "\n".join(context_parts) if context_parts else ""
        prompt = SYSTEM_PROMPT + "\n" + context + "\nUser: " + request.user_input + "\nBuddy:"
        
        # Call LLM
        response_text = await _call_llm(prompt)
        
        print(f"DEBUG BRAIN: Raw response: '{response_text}'")
        
        # Parse response
        reply, intent, emotion = _parse_response(response_text)
        
        print(f"DEBUG BRAIN: Parsed - Reply: '{reply}', Intent: '{intent}', Emotion: '{emotion}'")
        
        # Extract and save memory
        memory_saved = _extract_and_save_memory(request.user_input, effective_user)
        if memory_saved:
            reply = f"Got it! I'll remember that {memory_saved}. {reply}"
        
        # Save conversation
        save_conversation(
            request.user_input, 
            response_text, 
            intent, 
            effective_user
        )
        
        return ChatResponse(
            reply=reply,
            intent=intent,
            emotion=emotion,
            raw_response=response_text
        )
        
    except Exception as e:
        print(f"BRAIN ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _call_llm(prompt: str) -> str:
    """Call LLM service (Ollama)"""
    try:
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'llama3.2:3b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'num_predict': 80,
                    'num_ctx': 1024
                }
            },
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            return '{"reply": "Sorry, having trouble thinking.", "emotion": "apologetic", "intent": "conversation"}'
            
    except Exception:
        return '{"reply": "Sorry, having trouble thinking.", "emotion": "apologetic", "intent": "conversation"}'

def _parse_response(response: str) -> tuple[str, str, str]:
    """Parse LLM response and extract reply, intent, emotion"""
    try:
        # Clean response
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end != -1:
                response = response[start:end].strip()
            else:
                response = response[start:].strip()
        
        # Remove verbose parts
        if '\n\nIn this scenario' in response:
            response = response.split('\n\nIn this scenario')[0].strip()
        if '\n\n' in response:
            response = response.split('\n\n')[0].strip()
        
        # Fix incomplete JSON
        if response.startswith('{') and not response.endswith('}'):
            if '"intent":' in response and not response.rstrip().endswith('"'):
                response = response.rstrip().rstrip(',').rstrip('"') + '"conversation"}'
            else:
                response += '}'
        
        # Parse JSON
        response = html.unescape(response)
        data = json.loads(response.strip())
        
        reply = data.get("reply", data.get("response", ""))
        if isinstance(reply, dict):
            reply = reply.get("reply", "Sorry, I had trouble with that.")
        
        emotion = data.get("emotion", "neutral")
        intent = data.get("intent", "conversation")
        
        return reply, intent, emotion
        
    except json.JSONDecodeError:
        # Manual extraction fallback
        if '"reply":' in response:
            try:
                start = response.find('"reply":') + 8
                start = response.find('"', start) + 1
                end = response.find('"', start)
                if end > start:
                    reply = response[start:end]
                    return reply, "conversation", "neutral"
            except:
                pass
        
        # Return as plain text
        return response, "conversation", "neutral"

def _extract_and_save_memory(user_input: str, user_name: str) -> str:
    """Extract and save meaningful information from user input"""
    import re
    from memory import save_memory
    
    text = user_input.lower()
    effective_user = user_name if user_name else "Unknown"
    
    # Birthday patterns
    if any(word in text for word in ['birthday', 'born']):
        date_patterns = [
            r'(?:birthday|born).*?(\d{1,2}).*?(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(\d{1,2}).*?(?:st|nd|rd|th).*?(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(?:on|in)\s+(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if groups[0].isdigit():
                    day, month = groups[0], groups[1]
                else:
                    month, day = groups[0], groups[1]
                birthday = f"{day} {month.capitalize()}"
                save_memory("birthday", birthday, user_name=effective_user)
                return f"your birthday is {birthday}"
    
    # Personal preferences
    preference_patterns = [
        (r'(?:favorite|favourite|love|like).*?color.*?(?:is|are).*?(\w+)', 'favorite_color'),
        (r'(?:favorite|favourite|love|like).*?food.*?(?:is|are).*?([\w\s]+)', 'favorite_food'),
        (r'(?:favorite|favourite|love|like).*?movie.*?(?:is|are).*?([\w\s]+)', 'favorite_movie')
    ]
    
    for pattern, key in preference_patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            save_memory(key, value, user_name=effective_user)
            return f"your {key.replace('_', ' ')} is {value}"
    
    return None
    """Parse LLM response and extract reply, intent, emotion"""
    try:
        # Clean response
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end != -1:
                response = response[start:end].strip()
            else:
                response = response[start:].strip()
        
        # Remove verbose parts
        if '\n\nIn this scenario' in response:
            response = response.split('\n\nIn this scenario')[0].strip()
        if '\n\n' in response:
            response = response.split('\n\n')[0].strip()
        
        # Fix incomplete JSON
        if response.startswith('{') and not response.endswith('}'):
            if '"intent":' in response and not response.rstrip().endswith('"'):
                response = response.rstrip().rstrip(',').rstrip('"') + '"conversation"}'
            else:
                response += '}'
        
        # Parse JSON
        response = html.unescape(response)
        data = json.loads(response.strip())
        
        reply = data.get("reply", data.get("response", ""))
        if isinstance(reply, dict):
            reply = reply.get("reply", "Sorry, I had trouble with that.")
        
        emotion = data.get("emotion", "neutral")
        intent = data.get("intent", "conversation")
        
        return reply, intent, emotion
        
    except json.JSONDecodeError:
        # Manual extraction fallback
        if '"reply":' in response:
            try:
                start = response.find('"reply":') + 8
                start = response.find('"', start) + 1
                end = response.find('"', start)
                if end > start:
                    reply = response[start:end]
                    return reply, "conversation", "neutral"
            except:
                pass
        
        # Return as plain text
        return response, "conversation", "neutral"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Buddy LLM Service"}

@app.post("/clear")
async def clear_conversation():
    """Clear conversation history"""
    try:
        from memory import clear_conversations
        clear_conversations()
        print("DEBUG BRAIN: Conversation history cleared")
        return {"status": "cleared"}
    except Exception as e:
        print(f"DEBUG BRAIN: Error clearing conversations: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🧠 Starting Buddy LLM Service on port 8000...")
    print("📋 Make sure Ollama is running: ollama serve")
    print("📥 Make sure model is available: ollama pull llama3.2:3b")
    uvicorn.run(app, host="0.0.0.0", port=8000)
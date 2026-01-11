import subprocess
import os
import json
import html
import sys
import re
import threading
from memory import init_db, save_memory, get_memory, get_all_memory, save_conversation, search_conversations
from smart_memory import intelligent_memory_save

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "buddy_prompt.txt")

# Initialize memory system
init_db()

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Global state for waiting indicator
processing_state = {'is_processing': False, 'stop_indicator': False}

# Session conversation context (in-memory for immediate context)
session_context = {'conversations': [], 'max_size': 5}

def show_thinking_indicator():
    """Show animated thinking indicator"""
    indicators = ['🤔 Thinking', '🤔 Thinking.', '🤔 Thinking..', '🤔 Thinking...']
    i = 0
    while processing_state['is_processing'] and not processing_state['stop_indicator']:
        print(f'\r{indicators[i % len(indicators)]}', end='', flush=True)
        i += 1
        threading.Event().wait(0.5)
    if not processing_state['stop_indicator']:
        print('\r' + ' ' * 20 + '\r', end='', flush=True)  # Clear indicator

def build_enhanced_context(user_input, memory_context, user_name=None):
    """Build enhanced context with memory and recent conversation history"""
    context_parts = []
    
    # Add memory context
    if memory_context:
        facts = [f"{k}: {v}" for k, v in memory_context.items() if k not in ["name", "user_name"]]
        if facts:
            context_parts.append(f"Known facts: {', '.join(facts)}")
    
    # Add session conversation context (immediate context)
    if session_context['conversations']:
        context_parts.append("Recent conversation:")
        for conv in session_context['conversations'][-3:]:  # Last 3 for immediate context
            user_msg = conv['user'][:80] + "..." if len(conv['user']) > 80 else conv['user']
            buddy_msg = conv['buddy'][:80] + "..." if len(conv['buddy']) > 80 else conv['buddy']
            context_parts.append(f"User: {user_msg} | Buddy: {buddy_msg}")
    
    return "\n".join(context_parts)

def clear_session_context():
    """Clear session conversation context"""
    session_context['conversations'].clear()
    print("DEBUG: Session context cleared")

# Global lock for preventing concurrent processing
_processing_lock = threading.Lock()
_last_processed = {"input": "", "time": 0}

def ask_buddy(user_input, recognized_user=None):
    import time
    current_time = time.time()
    
    # Acquire lock to prevent concurrent processing
    if not _processing_lock.acquire(blocking=False):
        print("DEBUG: Already processing, skipping duplicate request")
        return None  # Return None to indicate skipped
    
    try:
        # Check for duplicate input within 3 seconds
        if (_last_processed["input"] == user_input and 
            current_time - _last_processed["time"] < 3.0):
            print(f"DEBUG: Duplicate input detected, ignoring: '{user_input}'")
            return None
        
        _last_processed["input"] = user_input
        _last_processed["time"] = current_time
        
        # Start processing indicator
        processing_state['is_processing'] = True
        processing_state['stop_indicator'] = False
        
        # Start thinking indicator in background
        indicator_thread = threading.Thread(target=show_thinking_indicator, daemon=True)
        indicator_thread.start()
        # Get current memory context
        memory_context = get_all_memory(user_name=recognized_user if recognized_user else "Unknown")
        
        # Use recognized_user if provided, otherwise check memory
        effective_user_name = recognized_user or memory_context.get("name") or memory_context.get("user_name", "")
        name_known = bool(effective_user_name)
        
        # Build enhanced context with memory and conversation history
        enhanced_context = build_enhanced_context(user_input, memory_context, effective_user_name)
        
        # Build final context
        context_parts = []
        if effective_user_name:
            context_parts.append(f"User name: {effective_user_name}")
        if enhanced_context:
            context_parts.append(enhanced_context)
        
        # Add current date and time
        from datetime import datetime
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        context_parts.append(f"Current date and time: {current_datetime}")
        
        context = "\n".join(context_parts) if context_parts else ""
        prompt = SYSTEM_PROMPT + "\n" + context + "\nUser: " + user_input + "\nBuddy:"
        
        print(f"DEBUG: Prompt length: {len(prompt)} chars")  # Debug output

        try:
            import requests
            import time
            
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    'model': 'llama3.2:3b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,
                        'top_p': 0.8,
                        'num_predict': 80,   # Increased from 50 to avoid cut-off JSON
                        'num_ctx': 1024
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result_text = response.json()['response'].strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as api_error:
            # Fallback to subprocess
            try:
                result = subprocess.run(
                    ["ollama", "run", "llama3.2:3b"],
                    input=prompt,
                    text=True,
                    capture_output=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=25  # Increased from 20 to 25
                )
                result_text = result.stdout.strip()
            except subprocess.TimeoutExpired:
                # Better fallback for system instructions
                if "I just recognized" in user_input and "on camera" in user_input:
                    # Extract name from system instruction
                    import re
                    name_match = re.search(r'I just recognized (\w+) on camera', user_input)
                    if name_match:
                        name = name_match.group(1)
                        return f"Hey {name}! Good to see you!"
                return "Hey! Good to see you!"
            except Exception as e:
                return f"Having trouble connecting. Try: 'ollama pull llama3.2:3b' first."

        response = result_text
        
        # Clean up qwen's markdown and verbose responses
        if '```json' in response:
            # Extract JSON from markdown
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end != -1:
                response = response[start:end].strip()
            else:
                response = response[start:].strip()
        
        if '\n\nIn this scenario' in response:
            response = response.split('\n\nIn this scenario')[0].strip()
        if '\n\n' in response:
            response = response.split('\n\n')[0].strip()
        
        print(f"DEBUG: Cleaned response: '{response[:100]}...'")
        
        print(f"DEBUG: AI raw response: '{response}'")
        
        # Decode HTML entities
        response = html.unescape(response)

        # Process response and enforce rules
        try:
            # Try to parse as JSON
            json_text = response.strip()
            
            # Fix missing closing brace if needed
            if json_text.startswith('{') and not json_text.endswith('}'):
                # Try to find where JSON was cut off and complete it
                if '"intent":' in json_text and not json_text.rstrip().endswith('"'):
                    # JSON was cut off mid-value, try to complete it
                    json_text = json_text.rstrip().rstrip(',').rstrip('"') + '"conversation"}'
                else:
                    json_text += '}'
            
            data = json.loads(json_text)
            reply = data.get("reply", data.get("response", ""))  # Handle both "reply" and "response" keys
            
            # Fix nested JSON in reply field
            if isinstance(reply, dict):
                # If reply is a dict, extract the actual text or use a default
                if "reply" in reply:
                    reply = reply["reply"]
                else:
                    reply = "Sorry, I had trouble with that."
            
            emotion = data.get("emotion", "neutral")
            intent = data.get("intent", "unknown")
            
            # Apply system rules
            reply, intent = _apply_system_rules(user_input, reply, intent, effective_user_name, memory_context)
            
            # Smart memory extraction and acknowledgment
            memory_saved = _extract_and_save_memory(user_input, effective_user_name)
            if memory_saved:
                reply = f"Got it! I'll remember that {memory_saved}. {reply}"
            
            # Save conversation with original JSON response
            save_conversation(user_input, response, intent, effective_user_name)
            
            # Add to session context for immediate continuity
            session_context['conversations'].append({
                'user': user_input,
                'buddy': reply
            })
            # Keep only last N conversations in memory
            if len(session_context['conversations']) > session_context['max_size']:
                session_context['conversations'].pop(0)
            
            print(f"DEBUG: Final reply: '{reply}'")
            return reply
            
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON parse failed: {e}")
            # Manual extraction fallback
            if '"reply":' in response:
                try:
                    start = response.find('"reply":') + 8
                    start = response.find('"', start) + 1
                    end = response.find('"', start)
                    if end > start:
                        reply = response[start:end]
                        # Still apply rules to manually extracted reply
                        reply, _ = _apply_system_rules(user_input, reply, "unknown", effective_user_name, memory_context)
                        # Smart memory extraction and acknowledgment
                        memory_saved = _extract_and_save_memory(user_input, effective_user_name)
                        if memory_saved:
                            reply = f"Got it! I'll remember that {memory_saved}. {reply}"
                        
                        # Save conversation with original response
                        save_conversation(user_input, response, "unknown", effective_user_name)
                        return reply
                except:
                    pass
            
            # If it's a plain text response, wrap it properly
            if response and not response.startswith('{'):
                # Apply rules to plain text response
                reply, intent = _apply_system_rules(user_input, response, "conversation", effective_user_name, memory_context)
                # Smart memory extraction and acknowledgment
                memory_saved = _extract_and_save_memory(user_input, effective_user_name)
                if memory_saved:
                    reply = f"Got it! I'll remember that {memory_saved}. {reply}"
                
                # Save conversation
                save_conversation(user_input, response, intent, effective_user_name)
                
                # Add to session context
                session_context['conversations'].append({
                    'user': user_input,
                    'buddy': reply
                })
                if len(session_context['conversations']) > session_context['max_size']:
                    session_context['conversations'].pop(0)
                
                return reply
            
            # Smart memory extraction and acknowledgment
            memory_saved = _extract_and_save_memory(user_input, effective_user_name)
            if memory_saved:
                response = f"Got it! I'll remember that {memory_saved}. {response}"
            
            # Save even unparseable responses
            save_conversation(user_input, response, "unknown", effective_user_name)
            
            # Add to session context
            session_context['conversations'].append({
                'user': user_input,
                'buddy': response
            })
            if len(session_context['conversations']) > session_context['max_size']:
                session_context['conversations'].pop(0)
            
            return response
    
    except Exception as e:
        print(f"DEBUG: Error in ask_buddy: {e}")
        return "Sorry, I had trouble with that."
    
    finally:
        # Stop processing indicator
        processing_state['is_processing'] = False
        processing_state['stop_indicator'] = True
        # Release the lock
        _processing_lock.release()


def _extract_and_save_memory(user_input, user_name):
    """Extract and save meaningful information from user input with user-specific keys"""
    import re
    text = user_input.lower()
    
    # Use "Unknown" as fallback user_name if None provided
    effective_user = user_name if user_name else "Unknown"
    
    # Any date patterns (birthdays, anniversaries, etc.)
    date_patterns = [
        r'(?:birthday|born|anniversary).*?(\d{1,2}).*?(january|february|march|april|may|june|july|august|september|october|november|december)',
        r'(\d{1,2}).*?(?:st|nd|rd|th).*?(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)',
        r'(?:on|in)\s+(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
        r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})'
    ]
    
    # Check for birthday specifically
    if any(word in text for word in ['birthday', 'born']):
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
    
    # Any personal preferences
    preference_patterns = [
        (r'(?:favorite|favourite|love|like).*?color.*?(?:is|are).*?(\w+)', 'favorite_color'),
        (r'(?:favorite|favourite|love|like).*?food.*?(?:is|are).*?([\w\s]+)', 'favorite_food'),
        (r'(?:favorite|favourite|love|like).*?movie.*?(?:is|are).*?([\w\s]+)', 'favorite_movie'),
        (r'(?:favorite|favourite|love|like).*?song.*?(?:is|are).*?([\w\s]+)', 'favorite_song'),
        (r'(?:favorite|favourite|love|like).*?book.*?(?:is|are).*?([\w\s]+)', 'favorite_book')
    ]
    
    for pattern, key in preference_patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            save_memory(key, value, user_name=effective_user)
            return f"your {key.replace('_', ' ')} is {value}"
    
    # Work/job information
    work_patterns = [
        (r'i work (?:as|at)\s+([\w\s]+)', 'job'),
        (r'my job is\s+([\w\s]+)', 'job'),
        (r'i am (?:a|an)\s+([\w\s]+)', 'job'),
        (r'i study\s+([\w\s]+)', 'study'),
        (r'my major is\s+([\w\s]+)', 'study')
    ]
    
    for pattern, key in work_patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            save_memory(key, value, user_name=effective_user)
            return f"you {key} as {value}" if key == 'job' else f"you {key} {value}"
    
    # Location information
    location_patterns = [
        (r'i live in\s+([\w\s]+)', 'location'),
        (r'i am from\s+([\w\s]+)', 'hometown'),
        (r'my hometown is\s+([\w\s]+)', 'hometown')
    ]
    
    for pattern, key in location_patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            save_memory(key, value, user_name=effective_user)
            return f"you live in {value}" if key == 'location' else f"you are from {value}"
    
    return None


def _apply_system_rules(user_input, reply, intent, user_name, memory_context):
    """Apply system rules that were removed from prompt"""
    
    # Rule: Handle corrections properly - don't give random responses
    correction_phrases = ['i asked', 'not the', 'i said', 'i meant', 'correction', 'wrong answer']
    if any(phrase in user_input.lower() for phrase in correction_phrases):
        # If it's a correction, don't override with random greetings
        if any(greeting in reply.lower() for greeting in ['hey', 'hello', 'good to see', 'how are', 'how\'s']):
            # This is a random greeting response to a correction - fix it
            if 'president' in user_input.lower() and 'pm' in user_input.lower():
                reply = "Oops! Dr. Rajendra Prasad was the first President"
                intent = "provide_info"
            elif 'prime minister' in user_input.lower() and 'president' in user_input.lower():
                reply = "My bad! Jawaharlal Nehru was the first PM"
                intent = "provide_info"
    
    # Rule: Personal info questions - only for sharing, not asking
    if any(indicator in user_input.lower() for indicator in ['my ', 'i am', 'i have', 'i like', 'i work', 'i live']):
        # This is for sharing info, don't override
        pass
    
    # Rule: Questions about any known info - check memory dynamically
    question_patterns = [
        ('birthday', ['my birthday', 'when is my birthday', 'do you know my birthday']),
        ('favorite_color', ['my favorite color', 'what is my favorite color', 'my favourite color']),
        ('favorite_food', ['my favorite food', 'what is my favorite food', 'my favourite food']),
        ('job', ['my job', 'what do i do', 'where do i work', 'what is my job']),
        ('location', ['where do i live', 'my location', 'where am i from']),
        ('hometown', ['my hometown', 'where am i from', 'my home'])
    ]
    
    for memory_key, questions in question_patterns:
        if any(q in user_input.lower() for q in questions):
            # Check user-specific memory first
            effective_user = user_name if user_name else "Unknown"
            user_memory_value = get_memory(memory_key, effective_user)
            if user_memory_value:
                # Let AI response stand if it has the right info
                if str(user_memory_value).lower() in reply.lower():
                    return reply, intent
                # Override if memory exists but AI doesn't use it
                reply = f"Your {memory_key.replace('_', ' ')} is {user_memory_value}!"
                intent = "provide_info"
            elif "don't know" not in reply.lower():
                reply = "Don't know that about you yet - wanna share?"
                intent = "ask_personal_info"
            break
    
    # Rule: Handle existing user identification properly
    if any(phrase in user_input.lower() for phrase in ['i am', 'yes i am', 'my name is']):
        # Extract name from user input
        name_match = re.search(r'(?:i am|yes i am|my name is)\s+([a-zA-Z]+)', user_input.lower())
        if name_match:
            mentioned_name = name_match.group(1).capitalize()
            # If the AI response already mentions this name correctly, don't override
            if mentioned_name.lower() in reply.lower():
                return reply, intent
    
    # Rule: Name handling
    if not user_name and intent == "greeting":
        if "what should i call you" not in reply.lower():
            reply = "Hey! What should I call you?"
            intent = "ask_name"
    
    # Rule: Use name only for initial greetings, not every response
    # Don't add names when user is introducing themselves or in non-greeting contexts
    if (user_name and intent == "greeting" and 
        any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']) and
        user_name.lower() not in reply.lower() and
        not any(intro in user_input.lower() for intro in ['i am', 'my name is', 'call me', 'i\'m'])):
        # Only add name if this seems like an initial greeting and not an introduction
        reply = f"Hey {user_name}! {reply.replace('Hey!', '').replace('Hey', '').strip()}"
    
    # Rule: Remove wrong name prefixes from responses
    if user_name and reply:
        # Remove any name that's not the current user's name
        all_names = ['Sagnik', 'Debanshu', 'John', 'Jane']  # Add more names as needed
        for name in all_names:
            if name != user_name and f"Hey {name}" in reply:
                reply = reply.replace(f"Hey {name}!", "").replace(f"Hey {name}", "").strip()
                if not reply:  # If reply becomes empty, give a generic response
                    reply = "Hey there!"
    
    # Rule: Handle movement commands
    movement_commands = ['come to me', 'come here', 'move forward', 'come towards']
    if any(cmd in user_input.lower() for cmd in movement_commands):
        if intent != "move_forward":
            intent = "move_forward"
        if "coming" not in reply.lower():
            reply = "Coming over!"
    
    # Rule: Remove forced name prefixes from AI responses
    if user_name and reply.startswith(f"Hey {user_name}!"):
        # Check if this is actually a greeting context, if not remove the name
        if not any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
            reply = reply.replace(f"Hey {user_name}! ", "").replace(f"Hey {user_name}!", "")
    
    # Rule: Prevent random birthday mentions and irrelevant responses
    if "birthday" in reply.lower() and "birthday" not in user_input.lower():
        # Remove random birthday mentions unless it's actually relevant
        reply = reply.split(", happy birthday")[0].split(" happy birthday")[0].split("happy belated birthday")[0].split("Happy birthday")[0].strip()
        # Remove trailing punctuation if left hanging
        if reply.endswith(("!", "?", ",")):
            reply = reply[:-1].strip()
    
    # Rule: Remove name prefixes from non-greeting responses and introductions
    if user_name and f"Hey {user_name}" in reply:
        # If not a greeting or if user is introducing themselves, remove the name prefix
        if (not any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']) or
            any(intro in user_input.lower() for intro in ['i am', 'my name is', 'call me', 'i\'m'])):
            reply = reply.replace(f"Hey {user_name}, ", "").replace(f"Hey {user_name}! ", "").replace(f"Hey {user_name}", "")
    
    # Rule: Object questions
    object_questions = ['do you see', 'what objects', 'is there a']
    if any(q in user_input.lower() for q in object_questions):
        # This would integrate with your object detection system
        # For now, generic response
        if "see" not in reply.lower():
            reply = "I can see what's around, what are you looking for?"
            intent = "ask_question"
    
    return reply, intent


if __name__ == "__main__":
    print("BUDDY is awake. Type something.\n")

    while True:
        user = input("You: ")

        if user.lower() in ["exit", "quit"]:
            break

        response = ask_buddy(user)
        print(f"\nBUDDY: {response}\n")

def get_intent_from_response(response_data):
    """Extract intent for hardware control"""
    try:
        if isinstance(response_data, dict):
            return response_data.get('intent', 'conversation')
        elif isinstance(response_data, str):
            # Try to parse JSON string
            import json
            data = json.loads(response_data)
            return data.get('intent', 'conversation')
    except:
        pass
    return 'conversation'
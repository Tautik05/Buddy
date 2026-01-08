import subprocess
import os
import json
import html
import sys
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

def build_enhanced_context(user_input, memory_context):
    """Build enhanced context with memory and conversation history"""
    context_parts = []
    
    # Add memory context
    if memory_context:
        facts = [f"{k}: {v}" for k, v in memory_context.items() if k not in ["name", "user_name"]]
        if facts:
            context_parts.append(f"Known facts: {', '.join(facts)}")
    
    # Search for relevant past conversations from database
    try:
        relevant_convs = search_conversations(user_input, limit=3)
        if relevant_convs:
            context_parts.append("Recent relevant conversations:")
            for conv in relevant_convs:
                user_msg = conv['user_input'][:60] + "..." if len(conv['user_input']) > 60 else conv['user_input']
                buddy_msg = conv['buddy_reply'][:60] + "..." if len(conv['buddy_reply']) > 60 else conv['buddy_reply']
                context_parts.append(f"User: {user_msg} | Buddy: {buddy_msg}")
    except Exception as e:
        pass  # Skip on error
    
    return "\n".join(context_parts)

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
        memory_context = get_all_memory()
        
        # Use recognized_user if provided, otherwise check memory
        user_name = recognized_user or memory_context.get("name") or memory_context.get("user_name", "")
        name_known = bool(user_name)
        
        # Build enhanced context with memory and conversation history
        enhanced_context = build_enhanced_context(user_input, memory_context)
        
        # Build final context
        context_parts = []
        if user_name:
            context_parts.append(f"User name: {user_name}")
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
            print(f"DEBUG: Calling ollama with llama3.2:3b...")
            import time
            ollama_start = time.time()
            result = subprocess.run(
                ["ollama", "run", "llama3.2:3b"],
                input=prompt,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                timeout=30  # Reduced timeout
            )
            print(f"DEBUG: Ollama took {time.time() - ollama_start:.2f}s")
        except subprocess.TimeoutExpired:
            return "Hey! I'm still thinking... give me a moment to respond."
        except FileNotFoundError:
            return "Ollama not found. Please make sure Ollama is installed and running."
        except Exception as e:
            return f"Having trouble connecting. Try: 'ollama pull phi3:mini' first."

        response = result.stdout.strip()
        
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
                json_text += '}'
            
            data = json.loads(json_text)
            reply = data.get("reply", "")
            
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
            reply, intent = _apply_system_rules(user_input, reply, intent, user_name, memory_context)
            
            # Smart memory extraction and acknowledgment
            memory_saved = _extract_and_save_memory(user_input, user_name)
            if memory_saved:
                reply = f"Got it! I'll remember that {memory_saved}. {reply}"
            
            # Save conversation with original JSON response
            save_conversation(user_input, response, intent, user_name)
            
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
                        reply, _ = _apply_system_rules(user_input, reply, "unknown", user_name, memory_context)
                        # Smart memory extraction and acknowledgment
                        memory_saved = _extract_and_save_memory(user_input, user_name)
                        if memory_saved:
                            reply = f"Got it! I'll remember that {memory_saved}. {reply}"
                        
                        # Save conversation with original response
                        save_conversation(user_input, response, "unknown", user_name)
                        return reply
                except:
                    pass
            
            # If it's a plain text response, wrap it properly
            if response and not response.startswith('{'):
                # Apply rules to plain text response
                reply, intent = _apply_system_rules(user_input, response, "conversation", user_name, memory_context)
                # Smart memory extraction and acknowledgment
                memory_saved = _extract_and_save_memory(user_input, user_name)
                if memory_saved:
                    reply = f"Got it! I'll remember that {memory_saved}. {reply}"
                
                # Save conversation
                save_conversation(user_input, response, intent, user_name)
                return reply
            
            # Smart memory extraction and acknowledgment
            memory_saved = _extract_and_save_memory(user_input, user_name)
            if memory_saved:
                response = f"Got it! I'll remember that {memory_saved}. {response}"
            
            # Save even unparseable responses
            save_conversation(user_input, response, "unknown", user_name)
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
    """Extract and save meaningful information from user input"""
    import re
    text = user_input.lower()
    
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
                save_memory("birthday", birthday)
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
            save_memory(key, value)
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
            save_memory(key, value)
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
            save_memory(key, value)
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
            if memory_context and memory_key in memory_context:
                # Let AI response stand if it has the right info
                if str(memory_context[memory_key]).lower() in reply.lower():
                    return reply, intent
                # Override if memory exists but AI doesn't use it
                reply = f"Your {memory_key.replace('_', ' ')} is {memory_context[memory_key]}!"
                intent = "provide_info"
            elif "don't know" not in reply.lower():
                reply = "Don't know that about you yet - wanna share?"
                intent = "ask_personal_info"
            break
    
    # Rule: Name handling
    if not user_name and intent == "greeting":
        if "what should i call you" not in reply.lower():
            reply = "Hey! What should I call you?"
            intent = "ask_name"
    
    # Rule: Use name only for initial greetings, not every response
    if (user_name and intent == "greeting" and 
        any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']) and
        user_name.lower() not in reply.lower()):
        # Only add name if this seems like an initial greeting
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
    
    # Rule: Remove name prefixes from non-greeting responses
    if user_name and f"Hey {user_name}" in reply:
        # If not a greeting, remove the name prefix
        if not any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
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
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
        context_parts.append(f"User Memory: {json.dumps(memory_context)}")
    
    # Search for relevant past conversations from database
    try:
        relevant_convs = search_conversations(user_input, limit=3)
        if relevant_convs:
            context_parts.append("Relevant past conversations:")
            for i, conv in enumerate(relevant_convs, 1):
                context_parts.append(f"{i}. User: {conv['user_input']} | Buddy: {conv['buddy_reply']}")
    except Exception as e:
        import logging
        logging.warning(f"Conversation search failed: {str(e)}")
    
    return "\n".join(context_parts)

def ask_buddy(user_input, recognized_user=None):
    # Start processing indicator
    processing_state['is_processing'] = True
    processing_state['stop_indicator'] = False
    
    # Start thinking indicator in background
    indicator_thread = threading.Thread(target=show_thinking_indicator, daemon=True)
    indicator_thread.start()
    
    try:
        # Get current memory context
        memory_context = get_all_memory()
        
        # Use recognized_user if provided, otherwise check memory
        user_name = recognized_user or memory_context.get("name") or memory_context.get("user_name", "")
        name_known = bool(user_name)
        
        # Build enhanced context with memory and conversation history
        enhanced_context = build_enhanced_context(user_input, memory_context)
        
        # Build context-aware prompt with greeting instruction
        greeting_instruction = f"\nIMPORTANT: The user's name is '{user_name}'. Use their name in your replies when appropriate.\n" if user_name else ""
        context = f"{enhanced_context}\nname_known: {name_known}\n{greeting_instruction}"
        prompt = SYSTEM_PROMPT + "\n" + context + "\nUser: " + user_input + "\nAssistant:"

        try:
            result = subprocess.run(
                ["ollama", "run", "llama3.2:3b"],
                input=prompt,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                timeout=30
            )
        except subprocess.TimeoutExpired:
            return "Sorry, I'm taking too long to respond. Please try again."
        except Exception as e:
            return f"Error communicating with AI: {str(e)}"

        response = result.stdout.strip()
        
        # Decode HTML entities
        response = html.unescape(response)

        # Process memory operations if JSON response
        try:
            data = json.loads(response)
            
            # Normalize response structure
            reply = data.get("reply", "")
            memory = data.get("memory", {})
            
            # Generate fallback reply if empty
            if not reply and memory.get("store"):
                reply = f"Got it{', ' + user_name if user_name else ''}! I'll remember that."
            elif not reply:
                reply = f"I understand{', ' + user_name if user_name else ''}."
                
            normalized = {
                "reply": reply,
                "emotion": data.get("emotion", "neutral"),
                "intent": data.get("intent", "unknown"),
                "memory": {
                    "store": memory.get("store", False),
                    "key": memory.get("key", ""),
                    "value": memory.get("value", ""),
                    "confidence": memory.get("confidence", 0.0)
                }
            }
            
            # Validate and store memory if needed
            memory = normalized["memory"]
            if memory.get("store") and memory.get("key") and memory.get("value"):
                save_memory(memory["key"], memory["value"], memory.get("confidence", 0.7))
            
            # Intelligent memory extraction from conversation
            smart_memories = intelligent_memory_save(user_input, normalized["intent"], memory_context)
            for mem_op in smart_memories:
                save_memory(mem_op["key"], mem_op["value"], mem_op["confidence"])
            
            # Store conversation in database
            try:
                save_conversation(user_input, reply, normalized["intent"], user_name)
            except Exception as e:
                pass  # Don't fail if conversation save fails
            
            return json.dumps(normalized, indent=2)
            
        except Exception:
            # For non-JSON responses, still store in database
            try:
                save_conversation(user_input, response, "unknown", user_name)
            except Exception:
                pass
                
            return response
    
    finally:
        # Stop processing indicator
        processing_state['is_processing'] = False
        processing_state['stop_indicator'] = True


if __name__ == "__main__":
    print("BUDDY is awake. Type something.\n")

    while True:
        user = input("You: ")

        if user.lower() in ["exit", "quit"]:
            break

        response = ask_buddy(user)
        print(f"\nBUDDY: {response}\n")

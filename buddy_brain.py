import subprocess
import os
import json
from memory import init_db, save_memory, get_memory, get_all_memory
from smart_memory import intelligent_memory_save

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "buddy_prompt.txt")

# Initialize memory system
init_db()

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


def ask_buddy(user_input):
    # Get current memory context
    memory_context = get_all_memory()
    name_known = "name" in memory_context or "user_name" in memory_context
    user_name = memory_context.get("name") or memory_context.get("user_name", "")

    # Build context-aware prompt with greeting instruction
    greeting_instruction = f"\nIMPORTANT: The user's name is '{user_name}'. Use their name in your replies when appropriate.\n" if user_name else ""
    context = f"Current memory: {json.dumps(memory_context)}\nname_known: {name_known}\n{greeting_instruction}"
    prompt = SYSTEM_PROMPT + "\n" + context + "\nUser: " + user_input + "\nAssistant:"

    result = subprocess.run(
        ["ollama", "run", "llama3.2:3b"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )

    response = result.stdout.strip()

    # Process memory operations if JSON response
    try:
        data = json.loads(response)
        
        """
        Generalized memory handling:
        - If model wants to store memory
        - Ensure reply exists
        - Ensure memory schema is complete
        """
        
        # Normalize response structure - keep original memory data if present
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
        
        return json.dumps(normalized, indent=2)
        
    except Exception:
        return response


if __name__ == "__main__":
    print("BUDDY is awake. Type something.\n")

    while True:
        user = input("You: ")

        if user.lower() in ["exit", "quit"]:
            break

        response = ask_buddy(user)
        print(f"\nBUDDY: {response}\n")

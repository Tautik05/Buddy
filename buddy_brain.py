import subprocess
import os
import json
import logging
from memory import save_memory, get_all_memory, init_db

# ------------------ BASIC SETUP ------------------

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "buddy_prompt.txt")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Initialize DB ONCE
init_db()

# ------------------ MEMORY HANDLING ------------------

def build_memory_context():
    """
    Inject only relevant, human-usable memory.
    Avoid dumping everything.
    """
    memory = get_all_memory(min_confidence=0.5)

    filtered = {}

    for key in memory:
        if key in ["name", "preference", "likes", "dislikes"]:
            filtered[key] = memory[key]

    if not filtered:
        return ""

    return (
        "Known user information (use naturally, do not repeat unless relevant):\n"
        + json.dumps(filtered, indent=2)
    )

# ------------------ AI CORE ------------------

def ask_buddy(user_input):
    memory_context = build_memory_context()

    prompt = (
        SYSTEM_PROMPT
        + "\n"
        + memory_context
        + "\nUser: "
        + user_input
        + "\nAssistant:"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:3b"],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="ignore",
            timeout=90  # prevents infinite hang
        )
    except Exception as e:
        logging.error(f"[OLLAMA ERROR] {e}")
        return {
            "reply": "Sorry, I felt a bit off just now. Can you say that again?",
            "emotion": "confused",
            "intent": "none"
        }

    output = result.stdout.strip()

    try:
        response = json.loads(output)
    except json.JSONDecodeError:
        logging.warning("[JSON ERROR] Model output was not valid JSON.")
        return {
            "reply": "I didn’t quite catch that properly. Let’s try again.",
            "emotion": "uncertain",
            "intent": "none"
        }

    # ------------------ MEMORY SAVE ------------------

    mem = response.get("memory", {})
    if (
        isinstance(mem, dict)
        and mem.get("store") is True
        and mem.get("key")
        and mem.get("value")
    ):
        try:
            confidence = float(mem.get("confidence", 0.7))
            save_memory(mem["key"], mem["value"], confidence)
        except Exception as e:
            logging.error(f"[MEMORY SAVE ERROR] {e}")

    return response

# ------------------ INTERACTION LOOP ------------------

print("BUDDY is awake. Type something.\n(Type 'exit' to quit)\n")

while True:
    user = input("You: ").strip()

    if user.lower() in ["exit", "quit"]:
        print("BUDDY: See you soon. Take care 🌙")
        break

    response = ask_buddy(user)

    print("\nBUDDY:", response.get("reply", ""), "\n")

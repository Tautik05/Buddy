import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "buddy_prompt.txt")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

def ask_buddy(user_input):
    prompt = SYSTEM_PROMPT + "\nUser: " + user_input + "\nAssistant:"

    result = subprocess.run(
        ["ollama", "run", "llama3.2:3b"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )


    return result.stdout.strip()

print("BUDDY is awake. Type something.\n")

while True:
    user = input("You: ")
    if user.lower() in ["exit", "quit"]:
        break
    print("\nBUDDY:", ask_buddy(user), "\n")

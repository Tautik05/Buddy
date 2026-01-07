import json
import os
import numpy as np

DB_PATH = "face_db.json"

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def find_person(embedding, threshold=0.6):
    db = load_db()
    best_match = None
    best_score = 0.0

    for person in db:
        score = cosine_similarity(embedding, person["embedding"])
        if score > best_score:
            best_score = score
            best_match = person

    if best_score >= threshold:
        return best_match["name"], round(best_score, 2)

    return None, round(best_score, 2)

def add_person(name, embedding):
    db = load_db()
    db.append({
        "name": name,
        "embedding": embedding
    })
    save_db(db)

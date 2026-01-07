from face.face_embedder import extract_embedding
from face.face_db import find_person, add_person

def recognize_face(image_path, spoken_name=None):
    embedding = extract_embedding(image_path)

    if embedding is None:
        return {"status": "no_face"}

    name, confidence = find_person(embedding)

    if name:
        return {
            "status": "known",
            "name": name,
            "confidence": confidence
        }

    if spoken_name:
        add_person(spoken_name, embedding)
        return {
            "status": "learned",
            "name": spoken_name
        }

    return {
        "status": "unknown"
    }

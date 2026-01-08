import os
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

def recognize_person(current_image_path):
    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith((".jpg", ".png")):
            continue

        person_name = os.path.splitext(filename)[0]
        known_image_path = os.path.join(KNOWN_FACES_DIR, filename)

        try:
            result = DeepFace.verify(
                img1_path=current_image_path,
                img2_path=known_image_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=True
            )

            if result["verified"]:
                return person_name

        except Exception as e:
            print("Verification error:", e)
            continue

    return None

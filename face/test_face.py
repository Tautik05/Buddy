import os
from face.face_verify import recognize_person

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "known_faces", "test_face.jpg")

name = recognize_person(IMAGE_PATH)

if name:
    print(f"Hi {name} 👋")
else:
    print("Hello there 😊")

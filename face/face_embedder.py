from deepface import DeepFace

def extract_embedding(image_path):
    """
    Takes a JPEG image path
    Returns a 512-D face embedding list or None
    """
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        return result[0]["embedding"]
    except Exception as e:
        print("Face embedding error:", e)
        return None

"""
Copy faces from PostgreSQL to Pi's JSON database
"""

import json
import numpy as np
from memory import get_all_faces  # Original memory system

def copy_faces_to_pi():
    """Copy faces from PostgreSQL to Pi's JSON format"""
    try:
        # Get faces from PostgreSQL
        faces = get_all_faces()  # This function needs to be added to memory.py
        
        if not faces:
            print("No faces found in PostgreSQL database")
            return
        
        # Convert to Pi format
        pi_faces = {}
        for name, embedding in faces.items():
            if isinstance(embedding, (list, tuple)):
                pi_faces[name] = embedding
            elif hasattr(embedding, 'tolist'):
                pi_faces[name] = embedding.tolist()
        
        # Save to Pi's JSON file
        with open('face_database.json', 'w') as f:
            json.dump(pi_faces, f)
        
        print(f"Copied {len(pi_faces)} faces to Pi database:")
        for name in pi_faces.keys():
            print(f"  - {name}")
            
    except Exception as e:
        print(f"Error copying faces: {e}")
        print("Creating empty Pi database...")
        with open('face_database.json', 'w') as f:
            json.dump({}, f)

if __name__ == "__main__":
    copy_faces_to_pi()
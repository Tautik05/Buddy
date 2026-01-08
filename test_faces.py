#!/usr/bin/env python3
"""
Quick Face Database Test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import get_all_faces
import numpy as np

def test_faces():
    print("🧠 FACE DATABASE TEST")
    print("=" * 30)
    
    try:
        faces = get_all_faces()
        print(f"Found {len(faces)} faces in database:")
        
        for name, embedding in faces.items():
            print(f"  - {name}: {type(embedding)} shape={embedding.shape if hasattr(embedding, 'shape') else 'N/A'}")
            
            # Check if embedding is valid
            if isinstance(embedding, np.ndarray) and embedding.size > 0:
                print(f"    ✅ Valid embedding (size: {embedding.size})")
            else:
                print(f"    ❌ Invalid embedding")
        
        if len(faces) == 0:
            print("⚠️  No faces found in database!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_faces()
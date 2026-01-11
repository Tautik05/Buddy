#!/usr/bin/env python3
"""
Cleanup script to remove invalid face names from database
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def cleanup_invalid_faces():
    """Remove faces with invalid names (sentences instead of names)"""
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        with conn.cursor() as cur:
            # Get all face names
            cur.execute("SELECT name FROM faces")
            faces = cur.fetchall()
            
            print("Current faces in database:")
            for face in faces:
                print(f"  - {face[0]}")
            
            # Remove faces with multiple words or non-alphabetic characters
            invalid_faces = []
            for face in faces:
                name = face[0]
                if len(name.split()) > 1 or not name.replace(' ', '').isalpha():
                    invalid_faces.append(name)
            
            if invalid_faces:
                print(f"\nRemoving invalid faces: {invalid_faces}")
                for invalid_name in invalid_faces:
                    cur.execute("DELETE FROM faces WHERE name = %s", (invalid_name,))
                conn.commit()
                print("Cleanup completed!")
            else:
                print("No invalid faces found!")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    cleanup_invalid_faces()
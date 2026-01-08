#!/usr/bin/env python3
"""
Quick database connection test
"""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("DATABASE_URL not found in environment")
    exit(1)

print("Testing connection to database...")
print(f"URL: {DATABASE_URL[:50]}...")

try:
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    print("Database connection successful")
    
    with conn.cursor() as cur:
        # Test basic query
        cur.execute("SELECT 1")
        result = cur.fetchone()
        print(f"Basic query works: {result}")
        
        # Check if conversations table exists
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name = 'conversations'
        """)
        table_exists = cur.fetchone()
        print(f"Conversations table exists: {bool(table_exists)}")
        
        if table_exists:
            # Check table structure
            cur.execute("""
                SELECT column_name, data_type FROM information_schema.columns 
                WHERE table_name = 'conversations'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()
            print("Table columns:")
            for col_name, col_type in columns:
                print(f"   - {col_name}: {col_type}")
            
            # Test simple query
            cur.execute("SELECT COUNT(*) FROM conversations")
            count = cur.fetchone()[0]
            print(f"Total conversations: {count}")
    
    conn.close()
    print("All tests passed")
    
except Exception as e:
    print(f"Database error: {str(e)}")
    print(f"Error type: {type(e).__name__}")
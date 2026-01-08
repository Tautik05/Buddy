#!/usr/bin/env python3
"""
Detailed debug of the database issue
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Enable all logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

print("Step 1: Testing direct database connection...")
try:
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    print("Connection successful")
    
    print("\nStep 2: Testing cursor creation...")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        print("Cursor created")
        
        print("\nStep 3: Testing column query...")
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'conversations'
            ORDER BY ordinal_position
        """)
        columns = [row[0] for row in cur.fetchall()]
        print(f"Columns found: {columns}")
        
        print("\nStep 4: Testing search query...")
        query = "hello"
        cur.execute("""
            SELECT user_input, ai_response as buddy_reply, 
                   intent, timestamp as created_at
            FROM conversations
            WHERE user_input ILIKE %s OR ai_response ILIKE %s
            ORDER BY timestamp DESC
            LIMIT %s
        """, (f"%{query}%", f"%{query}%", 5))
        
        print("Query executed")
        
        rows = cur.fetchall()
        print(f"Rows fetched: {len(rows)}")
        
        result = [dict(row) for row in rows]
        print(f"Result converted: {result}")
        
    conn.close()
    print("Connection closed")
    
except Exception as e:
    print(f"Error: {repr(e)}")
    print(f"Error type: {type(e)}")
    print(f"Error args: {e.args}")
    import traceback
    traceback.print_exc()
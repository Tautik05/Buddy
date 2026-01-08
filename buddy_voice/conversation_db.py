import os
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

def init_conversation_db():
    """Initialize conversation database with PostgreSQL"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                user_input TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                emotion VARCHAR(50),
                intent VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_name VARCHAR(100)
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Conversation database initialized")
        
    except Exception as e:
        print(f"Database init error: {e}")

def save_conversation(user_input, ai_response, emotion="neutral", intent="unknown", user_name=None):
    """Save conversation to PostgreSQL"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (user_input, ai_response, emotion, intent, user_name)
            VALUES (%s, %s, %s, %s, %s)
        ''', (user_input, ai_response, emotion, intent, user_name))
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Save conversation error: {e}")

def get_recent_conversations(limit=10):
    """Get recent conversations for context"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_input, ai_response, timestamp 
            FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT %s
        ''', (limit,))
        
        conversations = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return conversations
        
    except Exception as e:
        print(f"Get conversations error: {e}")
        return []

def get_conversation_history(user_name=None, days=7):
    """Get conversation history for specific user"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        if user_name:
            cursor.execute('''
                SELECT user_input, ai_response, timestamp 
                FROM conversations 
                WHERE user_name = %s AND timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp DESC
            ''', (user_name, days))
        else:
            cursor.execute('''
                SELECT user_input, ai_response, timestamp 
                FROM conversations 
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp DESC
            ''', (days,))
        
        history = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return history
        
    except Exception as e:
        print(f"Get history error: {e}")
        return []
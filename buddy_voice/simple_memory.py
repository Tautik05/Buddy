import sqlite3
import os
import json
from typing import Dict, Any, Optional

DB_FILE = "buddy_memory.db"

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.7,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_memory(key: str, value: str, confidence: float = 0.7):
    """Save a key-value pair to memory"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO memory (key, value, confidence, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ''', (key, value, confidence))
    
    conn.commit()
    conn.close()

def get_memory(key: str) -> Optional[str]:
    """Retrieve a value by key"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT value FROM memory WHERE key = ?', (key,))
    result = cursor.fetchone()
    
    conn.close()
    
    return result[0] if result else None

def get_all_memory() -> Dict[str, Any]:
    """Get all memory as a dictionary"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT key, value FROM memory')
    results = cursor.fetchall()
    
    conn.close()
    
    return {key: value for key, value in results}

def delete_memory(key: str):
    """Delete a memory entry"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM memory WHERE key = ?', (key,))
    
    conn.commit()
    conn.close()
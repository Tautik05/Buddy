import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from contextlib import contextmanager
import json
import logging

# ------------------ BASIC SETUP ------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Check your .env file."
    )

# Optional: basic logging (safe for hackathon)
logging.basicConfig(level=logging.INFO)

# ------------------ DB CONNECTION ------------------

@contextmanager
def get_db():
    """
    Context-managed DB connection.
    Ensures proper close even if errors occur.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            DATABASE_URL,
            sslmode="require"   # Neon-safe
        )
        yield conn
    except Exception as e:
        logging.error(f"[DB ERROR] {e}")
        raise
    finally:
        if conn:
            conn.close()

# ------------------ INIT (RUN ONCE) ------------------

def init_db():
    """
    Initialize memory table.
    Call this ONCE when app starts.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    confidence FLOAT DEFAULT 1.0,
                    last_updated TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()

# ------------------ MEMORY OPERATIONS ------------------

def save_memory(key, value, confidence=1.0):
    """
    Store or update memory.
    Value is stored as JSON string for flexibility.
    """
    try:
        value_str = json.dumps(value)
    except Exception:
        value_str = str(value)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO memory (key, value, confidence, last_updated)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (key)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    confidence = EXCLUDED.confidence,
                    last_updated = NOW()
            """, (key, value_str, confidence))
            conn.commit()

def get_memory(key):
    """
    Fetch a single memory item.
    Returns Python object or None.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM memory WHERE key = %s",
                (key,)
            )
            row = cur.fetchone()
            if not row:
                return None

            try:
                return json.loads(row[0])
            except Exception:
                return row[0]

def get_all_memory(min_confidence=0.0):
    """
    Fetch all memory items above confidence threshold.
    Returns dict.
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT key, value
                FROM memory
                WHERE confidence >= %s
            """, (min_confidence,))
            rows = cur.fetchall()

    memory = {}
    for row in rows:
        try:
            memory[row["key"]] = json.loads(row["value"])
        except Exception:
            memory[row["key"]] = row["value"]

    return memory

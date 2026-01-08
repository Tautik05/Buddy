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
    except psycopg2.Error as e:
        logging.error(f"[DB ERROR] PostgreSQL error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"[DB ERROR] General error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

# ------------------ INIT (RUN ONCE) ------------------

def init_db():
    """
    Initialize memory, faces, and conversations tables.
    Call this ONCE when app starts.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            # Memory table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    confidence FLOAT DEFAULT 1.0,
                    last_updated TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Faces table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    name TEXT PRIMARY KEY,
                    embedding TEXT,
                    confidence FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_seen TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Conversations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    user_input TEXT NOT NULL,
                    buddy_reply TEXT NOT NULL,
                    intent TEXT DEFAULT 'unknown',
                    user_name TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            conn.commit()
            
            # Run migrations
            migrate_database(cur, conn)

def migrate_database(cur, conn):
    """
    Handle database migrations for existing tables.
    """
    try:
        # Check if conversations table exists and what columns it has
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'conversations'
            ORDER BY ordinal_position
        """)
        columns = [row[0] for row in cur.fetchall()]
        
        if columns:
            logging.info(f"Conversations table has columns: {columns}")
            
            # Add missing columns based on existing structure
            if 'buddy_reply' not in columns and 'ai_response' not in columns:
                try:
                    cur.execute("ALTER TABLE conversations ADD COLUMN buddy_reply TEXT")
                    logging.info("Added buddy_reply column")
                except Exception as e:
                    logging.warning(f"Could not add buddy_reply column: {e}")
            
            if 'created_at' not in columns and 'timestamp' not in columns:
                try:
                    cur.execute("ALTER TABLE conversations ADD COLUMN created_at TIMESTAMP DEFAULT NOW()")
                    logging.info("Added created_at column")
                except Exception as e:
                    logging.warning(f"Could not add created_at column: {e}")
            
            if 'intent' not in columns:
                try:
                    cur.execute("ALTER TABLE conversations ADD COLUMN intent TEXT DEFAULT 'unknown'")
                    logging.info("Added intent column")
                except Exception as e:
                    logging.warning(f"Could not add intent column: {e}")
            
            if 'user_name' not in columns:
                try:
                    cur.execute("ALTER TABLE conversations ADD COLUMN user_name TEXT")
                    logging.info("Added user_name column")
                except Exception as e:
                    logging.warning(f"Could not add user_name column: {e}")
            
            conn.commit()
        
    except Exception as e:
        logging.warning(f"Migration failed: {e}")

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
    Returns dict with better error handling.
    """
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT key, value, confidence, last_updated
                    FROM memory
                    WHERE confidence >= %s
                    ORDER BY last_updated DESC
                """, (min_confidence,))
                rows = cur.fetchall()

        memory = {}
        for row in rows:
            try:
                # Try to parse as JSON first
                parsed_value = json.loads(row["value"])
                memory[row["key"]] = parsed_value
            except (json.JSONDecodeError, TypeError):
                # If not JSON, store as string
                memory[row["key"]] = row["value"]
            except Exception as e:
                logging.warning(f"Error parsing memory value for key '{row['key']}': {e}")
                memory[row["key"]] = row["value"]

        return memory
    except Exception as e:
        logging.error(f"Error retrieving memory: {e}")
        return {}

def search_memory(query):
    """
    Search memory by key or value content.
    Returns matching memory items.
    """
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT key, value, confidence, last_updated
                    FROM memory
                    WHERE key ILIKE %s OR value ILIKE %s
                    ORDER BY confidence DESC, last_updated DESC
                """, (f"%{query}%", f"%{query}%"))
                rows = cur.fetchall()

        results = []
        for row in rows:
            try:
                parsed_value = json.loads(row["value"])
            except:
                parsed_value = row["value"]
            
            results.append({
                'key': row['key'],
                'value': parsed_value,
                'confidence': row['confidence'],
                'last_updated': str(row['last_updated'])
            })
        
        return results
    except Exception as e:
        logging.error(f"Error searching memory: {e}")
        return []

# ------------------ FACE OPERATIONS ------------------

def save_face(name, embedding, confidence=1.0):
    """
    Store or update face embedding.
    """
    try:
        embedding_str = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
    except Exception:
        embedding_str = str(embedding)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO faces (name, embedding, confidence, created_at, last_seen)
                VALUES (%s, %s, %s, NOW(), NOW())
                ON CONFLICT (name)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    confidence = EXCLUDED.confidence,
                    last_seen = NOW()
            """, (name, embedding_str, confidence))
            conn.commit()

def get_face(name):
    """
    Fetch a face embedding by name.
    Returns numpy array or None.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM faces WHERE name = %s",
                (name,)
            )
            row = cur.fetchone()
            if not row:
                return None

            try:
                import numpy as np
                return np.array(json.loads(row[0]))
            except Exception:
                return None

def get_all_faces():
    """
    Fetch all face embeddings.
    Returns dict of {name: embedding}.
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT name, embedding FROM faces")
            rows = cur.fetchall()

    faces = {}
    for row in rows:
        try:
            import numpy as np
            faces[row["name"]] = np.array(json.loads(row["embedding"]))
        except Exception:
            continue

    return faces

# ------------------ CONVERSATION OPERATIONS ------------------

def save_conversation(user_input, buddy_reply, intent="unknown", user_name=None):
    """
    Save conversation to database with adaptive column mapping.
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:  # Use regular cursor for schema queries
                # Check what columns exist
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'conversations'
                """)
                columns = [row[0] for row in cur.fetchall()]
                
                if not columns:
                    return
                
                # Since both ai_response and buddy_reply exist, populate both
                if 'ai_response' in columns and 'buddy_reply' in columns:
                    cur.execute("""
                        INSERT INTO conversations (user_input, ai_response, buddy_reply, intent, user_name, timestamp)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                    """, (user_input, buddy_reply, buddy_reply, intent, user_name))
                elif 'ai_response' in columns:
                    cur.execute("""
                        INSERT INTO conversations (user_input, ai_response, intent, user_name, timestamp)
                        VALUES (%s, %s, %s, %s, NOW())
                    """, (user_input, buddy_reply, intent, user_name))
                elif 'buddy_reply' in columns:
                    cur.execute("""
                        INSERT INTO conversations (user_input, buddy_reply, intent, user_name, timestamp)
                        VALUES (%s, %s, %s, %s, NOW())
                    """, (user_input, buddy_reply, intent, user_name))
                else:
                    # Minimal insert
                    cur.execute("""
                        INSERT INTO conversations (user_input, timestamp)
                        VALUES (%s, NOW())
                    """, (user_input,))
                
                conn.commit()
    except Exception as e:
        logging.warning(f"Failed to save conversation: {str(e)}")

def search_conversations(query, limit=5):
    """
    Search conversations with adaptive column mapping.
    Returns recent relevant conversations.
    """
    try:
        # Skip search if query is empty or invalid
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            return []
            
        with get_db() as conn:
            # Use regular cursor for schema queries
            with conn.cursor() as schema_cur:
                schema_cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'conversations'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in schema_cur.fetchall()]
                
                if not columns:
                    return []
            
            # Use RealDictCursor for data queries
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use ai_response since it exists
                if 'ai_response' in columns:
                    cur.execute("""
                        SELECT user_input, ai_response as buddy_reply, 
                               intent, timestamp as created_at
                        FROM conversations
                        WHERE user_input ILIKE %s OR ai_response ILIKE %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (f"%{query}%", f"%{query}%", limit))
                else:
                    # Search only user input
                    cur.execute("""
                        SELECT user_input, timestamp as created_at
                        FROM conversations
                        WHERE user_input ILIKE %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (f"%{query}%", limit))
                
                rows = cur.fetchall()
                return [dict(row) for row in rows]
                
    except Exception as e:
        logging.warning(f"Conversation search failed: {str(e)}")
        return []

def get_recent_conversations(limit=10):
    """
    Get recent conversations for context.
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT user_input, buddy_reply, intent, created_at
                FROM conversations
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            
            return [dict(row) for row in cur.fetchall()]

def update_face_name(old_name, new_name):
    """
    Update face name in database.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE faces SET name = %s WHERE name = %s",
                (new_name, old_name)
            )
            conn.commit()
            return cur.rowcount > 0

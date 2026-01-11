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

def save_memory(key, value, confidence=1.0, user_name=None):
    """
    Store or update memory with user-specific keys.
    Value is stored as JSON string for flexibility.
    """
    # Create user-specific key if user_name provided
    if user_name:
        memory_key = f"{user_name}_{key}"
    else:
        memory_key = key
    
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
            """, (memory_key, value_str, confidence))
            conn.commit()

def get_memory(key, user_name=None):
    """
    Fetch a single memory item with user-specific support.
    Returns Python object or None.
    """
    # Try user-specific key first if user_name provided
    if user_name:
        memory_key = f"{user_name}_{key}"
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM memory WHERE key = %s",
                    (memory_key,)
                )
                row = cur.fetchone()
                if row:
                    try:
                        return json.loads(row[0])
                    except Exception:
                        return row[0]
    
    # Fallback to generic key
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

def get_all_memory(min_confidence=0.0, user_name=None):
    """
    Fetch all memory items above confidence threshold with user-specific support.
    Returns dict with better error handling.
    """
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if user_name:
                    # Get user-specific memories
                    cur.execute("""
                        SELECT key, value, confidence, last_updated
                        FROM memory
                        WHERE confidence >= %s AND key LIKE %s
                        ORDER BY last_updated DESC
                    """, (min_confidence, f"{user_name}_%"))
                else:
                    # Get all memories
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
                # Remove user prefix from key for display
                display_key = row["key"]
                if user_name and display_key.startswith(f"{user_name}_"):
                    display_key = display_key[len(f"{user_name}_"):]
                
                # Try to parse as JSON first
                parsed_value = json.loads(row["value"])
                memory[display_key] = parsed_value
            except (json.JSONDecodeError, TypeError):
                # If not JSON, store as string
                memory[display_key] = row["value"]
            except Exception as e:
                logging.warning(f"Error parsing memory value for key '{row['key']}': {e}")
                memory[display_key] = row["value"]

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
            with conn.cursor() as cur:
                # Check what columns exist
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'conversations'
                """)
                columns = [row[0] for row in cur.fetchall()]
                
                if not columns:
                    return
                
                # Determine which timestamp column to use
                timestamp_col = 'created_at' if 'created_at' in columns else 'timestamp'
                
                # Since both ai_response and buddy_reply might exist, populate both
                if 'ai_response' in columns and 'buddy_reply' in columns:
                    cur.execute(f"""
                        INSERT INTO conversations (user_input, ai_response, buddy_reply, intent, user_name, {timestamp_col})
                        VALUES (%s, %s, %s, %s, %s, NOW())
                    """, (user_input, buddy_reply, buddy_reply, intent, user_name))
                elif 'ai_response' in columns:
                    cur.execute(f"""
                        INSERT INTO conversations (user_input, ai_response, intent, user_name, {timestamp_col})
                        VALUES (%s, %s, %s, %s, NOW())
                    """, (user_input, buddy_reply, intent, user_name))
                elif 'buddy_reply' in columns:
                    cur.execute(f"""
                        INSERT INTO conversations (user_input, buddy_reply, intent, user_name, {timestamp_col})
                        VALUES (%s, %s, %s, %s, NOW())
                    """, (user_input, buddy_reply, intent, user_name))
                else:
                    # Minimal insert
                    cur.execute(f"""
                        INSERT INTO conversations (user_input, {timestamp_col})
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
            
            # Determine timestamp column
            timestamp_col = 'created_at' if 'created_at' in columns else 'timestamp'
            
            # Use RealDictCursor for data queries
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use ai_response or buddy_reply, whichever exists
                if 'buddy_reply' in columns:
                    cur.execute(f"""
                        SELECT user_input, buddy_reply, 
                               intent, {timestamp_col} as created_at
                        FROM conversations
                        WHERE user_input ILIKE %s OR buddy_reply ILIKE %s
                        ORDER BY {timestamp_col} DESC
                        LIMIT %s
                    """, (f"%{query}%", f"%{query}%", limit))
                elif 'ai_response' in columns:
                    cur.execute(f"""
                        SELECT user_input, ai_response as buddy_reply, 
                               intent, {timestamp_col} as created_at
                        FROM conversations
                        WHERE user_input ILIKE %s OR ai_response ILIKE %s
                        ORDER BY {timestamp_col} DESC
                        LIMIT %s
                    """, (f"%{query}%", f"%{query}%", limit))
                else:
                    # Search only user input
                    cur.execute(f"""
                        SELECT user_input, {timestamp_col} as created_at
                        FROM conversations
                        WHERE user_input ILIKE %s
                        ORDER BY {timestamp_col} DESC
                        LIMIT %s
                    """, (f"%{query}%", limit))
                
                rows = cur.fetchall()
                return [dict(row) for row in rows]
                
    except Exception as e:
        logging.warning(f"Conversation search failed: {str(e)}")
        return []

def get_recent_conversations(user_name=None, limit=10):
    """
    Get recent conversations for context with adaptive column mapping.
    NOW SUPPORTS user_name FILTERING!
    """
    try:
        with get_db() as conn:
            # Check what columns exist
            with conn.cursor() as schema_cur:
                schema_cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'conversations'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in schema_cur.fetchall()]
                
                if not columns:
                    return []
            
            # Determine which columns are available
            timestamp_col = 'created_at' if 'created_at' in columns else 'timestamp'
            reply_col = 'buddy_reply' if 'buddy_reply' in columns else 'ai_response'
            
            # Use RealDictCursor for data queries
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query based on available columns and user filter
                if user_name:
                    # Filter by user_name if provided
                    query = f"""
                        SELECT user_input, {reply_col} as buddy_reply, intent, {timestamp_col} as created_at
                        FROM conversations
                        WHERE user_name = %s
                        ORDER BY {timestamp_col} DESC
                        LIMIT %s
                    """
                    cur.execute(query, (user_name, limit))
                else:
                    # Get all conversations
                    query = f"""
                        SELECT user_input, {reply_col} as buddy_reply, intent, {timestamp_col} as created_at
                        FROM conversations
                        ORDER BY {timestamp_col} DESC
                        LIMIT %s
                    """
                    cur.execute(query, (limit,))
                
                rows = cur.fetchall()
                
                # Format results for brain service
                formatted = []
                for row in rows:
                    formatted.append({
                        'user': row.get('user_input', ''),
                        'buddy': row.get('buddy_reply', ''),
                        'intent': row.get('intent', 'unknown')
                    })
                
                return formatted
                
    except Exception as e:
        logging.warning(f"Get recent conversations failed: {str(e)}")
        return []

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

def clear_conversations():
    """
    Clear all conversations from database to keep it light and spacious.
    Called on sleep/shutdown for fresh start.
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM conversations")
                conn.commit()
                logging.info("Cleared all conversations from database")
                return True
    except Exception as e:
        logging.error(f"Failed to clear conversations: {e}")
        return False

def migrate_unknown_memory(new_user_name):
    """
    Migrate memory from Unknown user to newly recognized user.
    Called when face recognition learns a new name.
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                # Find all Unknown user memories
                cur.execute(
                    "SELECT key, value, confidence FROM memory WHERE key LIKE %s",
                    ("Unknown_%",)
                )
                unknown_memories = cur.fetchall()
                
                if unknown_memories:
                    # Migrate each memory to the new user
                    for key, value, confidence in unknown_memories:
                        # Extract the memory type (remove "Unknown_" prefix)
                        memory_type = key[8:]  # Remove "Unknown_" (8 chars)
                        new_key = f"{new_user_name}_{memory_type}"
                        
                        # Insert with new user key
                        cur.execute("""
                            INSERT INTO memory (key, value, confidence, last_updated)
                            VALUES (%s, %s, %s, NOW())
                            ON CONFLICT (key)
                            DO UPDATE SET
                                value = EXCLUDED.value,
                                confidence = EXCLUDED.confidence,
                                last_updated = NOW()
                        """, (new_key, value, confidence))
                    
                    # Delete old Unknown memories
                    cur.execute("DELETE FROM memory WHERE key LIKE %s", ("Unknown_%",))
                    conn.commit()
                    
                    logging.info(f"Migrated {len(unknown_memories)} memories from Unknown to {new_user_name}")
                    return len(unknown_memories)
                    
        return 0
    except Exception as e:
        logging.error(f"Failed to migrate unknown memory: {e}")
        return 0
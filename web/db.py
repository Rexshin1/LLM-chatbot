import sqlite3
import os
import uuid
from datetime import datetime, date

def get_db_path():
    return os.getenv("REXA_DB_PATH", "web/rexa.db")

def get_db():
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            google_id TEXT UNIQUE,
            name TEXT,
            email TEXT,
            avatar_url TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            guest_session_id TEXT,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            model TEXT,
            created_at TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS guest_usage (
            guest_session_id TEXT,
            day TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY(guest_session_id, day)
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT,
            expires_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)
        conn.commit()

# --- User & Session Operations ---

def get_user_by_google_id(google_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE google_id = ?", (google_id,)).fetchone()
        return dict(row) if row else None

def create_user(google_id, name, email, avatar_url):
    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (id, google_id, name, email, avatar_url, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, google_id, name, email, avatar_url, now, now)
        )
        conn.commit()
    return user_id

def get_user_by_session_id(session_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT u.* FROM users u JOIN sessions s ON u.id = s.user_id WHERE s.id = ?",
            (session_id,)
        ).fetchone()
        return dict(row) if row else None

def create_session(user_id):
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    # Simple expiry date (e.g. 7 days from now)
    expires_at = datetime.utcnow().isoformat() # Will verify per request or expire on client cookie
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (session_id, user_id, now, expires_at)
        )
        conn.commit()
    return session_id

def delete_session(session_id):
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()

# --- Guest Limit Operations ---

def get_guest_usage_today(guest_session_id):
    today = date.today().isoformat()
    with get_db() as conn:
        row = conn.execute(
            "SELECT count FROM guest_usage WHERE guest_session_id = ? AND day = ?",
            (guest_session_id, today)
        ).fetchone()
        return row["count"] if row else 0

def increment_guest_usage(guest_session_id):
    today = date.today().isoformat()
    with get_db() as conn:
        row = conn.execute(
            "SELECT count FROM guest_usage WHERE guest_session_id = ? AND day = ?",
            (guest_session_id, today)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE guest_usage SET count = count + 1 WHERE guest_session_id = ? AND day = ?",
                (guest_session_id, today)
            )
        else:
            conn.execute(
                "INSERT INTO guest_usage (guest_session_id, day, count) VALUES (?, ?, 1)",
                (guest_session_id, today)
            )
        conn.commit()

# --- Conversation & Message Operations ---

def create_conversation(user_id=None, guest_session_id=None, title="New Chat"):
    conv_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, guest_session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, user_id, guest_session_id, title, now, now)
        )
        conn.commit()
    return conv_id

def get_conversation(conv_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        return dict(row) if row else None

def get_user_conversations(user_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

def get_guest_conversations(guest_session_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE guest_session_id = ? AND user_id IS NULL ORDER BY updated_at DESC",
            (guest_session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

def delete_conversation(conv_id):
    with get_db() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()

def create_message(conversation_id, role, content, model=None):
    msg_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, model, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, model, now)
        )
        # Update updated_at of conversation
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id)
        )
        conn.commit()
    return msg_id

def get_conversation_messages(conversation_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,)
        ).fetchall()
        return [dict(r) for r in rows]

# --- Conversation Migration ---

def migrate_guest_chats(guest_session_id, user_id):
    with get_db() as conn:
        # Check if there are conversations for this guest session
        conn.execute(
            "UPDATE conversations SET user_id = ? WHERE guest_session_id = ? AND user_id IS NULL",
            (user_id, guest_session_id)
        )
        conn.commit()

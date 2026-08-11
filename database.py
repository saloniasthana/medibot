import os
import uuid
import contextlib

import bcrypt
import psycopg2
import psycopg2.errors
from dotenv import load_dotenv

load_dotenv()

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

CREATE_CHAT_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

ADD_SESSION_ID_COLUMN = """
ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS session_id TEXT;
"""


@contextlib.contextmanager
def get_cursor():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    finally:
        conn.close()


def init_db():
    with get_cursor() as cur:
        cur.execute(CREATE_USERS_TABLE)
        cur.execute(CREATE_CHAT_HISTORY_TABLE)
        cur.execute(ADD_SESSION_ID_COLUMN)


def new_session_id():
    return str(uuid.uuid4())


def create_user(username, password):
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with get_cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, password_hash),
            )
        return True, "Account created. You can log in now."
    except psycopg2.errors.UniqueViolation:
        return False, "That username is already taken."


def verify_login(username, password):
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, password_hash FROM users WHERE username = %s",
            (username,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    user_id, password_hash = row
    if bcrypt.checkpw(password.encode(), password_hash.encode()):
        return user_id
    return None


def save_message(user_id, session_id, role, content):
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO chat_history (user_id, session_id, role, content) VALUES (%s, %s, %s, %s)",
            (user_id, session_id, role, content),
        )


def get_session_messages(session_id):
    with get_cursor() as cur:
        cur.execute(
            "SELECT role, content FROM chat_history WHERE session_id = %s ORDER BY created_at ASC",
            (session_id,),
        )
        rows = cur.fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def list_user_sessions(user_id):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT c1.session_id,
                   (SELECT content FROM chat_history c2
                    WHERE c2.session_id = c1.session_id AND c2.role = 'user'
                    ORDER BY c2.created_at ASC LIMIT 1) AS title,
                   MAX(c1.created_at) AS last_activity
            FROM chat_history c1
            WHERE c1.user_id = %s AND c1.session_id IS NOT NULL
            GROUP BY c1.session_id
            ORDER BY last_activity DESC
            """,
            (user_id,),
        )
        rows = cur.fetchall()
    return [
        {"session_id": session_id, "title": title, "last_activity": last_activity}
        for session_id, title, last_activity in rows
    ]

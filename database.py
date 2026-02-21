"""
database.py — SQLite face embedding store
"""

import sqlite3
import numpy as np
import pickle
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "faces.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the users table if it doesn't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT    NOT NULL,
                embedding BLOB  NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    print(f"[DB] Database ready at: {DB_PATH}")


def add_user(name: str, embedding: np.ndarray):
    """Insert a new face embedding into the database."""
    blob = pickle.dumps(embedding.astype(np.float32))
    with get_connection() as conn:
        conn.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, blob))
        conn.commit()
    print(f"[DB] Added user: '{name}'")


def get_all_users() -> list[tuple[str, np.ndarray]]:
    """Return list of (name, embedding) tuples for all stored users."""
    with get_connection() as conn:
        rows = conn.execute("SELECT name, embedding FROM users").fetchall()
    return [(name, pickle.loads(blob)) for name, blob in rows]


def list_users():
    """Print all registered users."""
    with get_connection() as conn:
        rows = conn.execute("SELECT id, name, created_at FROM users ORDER BY id").fetchall()
    if not rows:
        print("[DB] No users registered yet.")
    else:
        print(f"\n{'ID':<5} {'Name':<25} {'Registered'}")
        print("-" * 50)
        for row in rows:
            print(f"{row[0]:<5} {row[1]:<25} {row[2]}")
    return rows


def delete_user(name: str):
    """Delete all embeddings for a given name."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM users WHERE name = ?", (name,))
        conn.commit()
    print(f"[DB] Deleted {cur.rowcount} record(s) for '{name}'")


if __name__ == "__main__":
    init_db()
    list_users()

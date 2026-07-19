"""SQLite cache for validator evaluation results."""

import os
import sqlite3

DB_PATH = os.path.join(os.environ.get("DATA_DIR", "/app/data"), "validator_cache.db")


def get_connection():
    import bittensor as bt
    bt.logging.info(f"Cache DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            uid INTEGER, year INTEGER, repo_id TEXT,
            passed INTEGER, score REAL,
            evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (uid, year, repo_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            week INTEGER PRIMARY KEY,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def get_cached_result(conn, uid: int, year: int, repo_id: str):
    """Returns (passed, score) or None if not cached."""
    row = conn.execute(
        "SELECT passed, score FROM evaluations WHERE uid=? AND year=? AND repo_id=?",
        (uid, year, repo_id)
    ).fetchone()
    if row is None:
        return None
    return bool(row[0]), row[1]


def save_result(conn, uid: int, year: int, repo_id: str, passed: bool, score: float = 0.0):
    conn.execute(
        """INSERT OR REPLACE INTO evaluations
           (uid, year, repo_id, passed, score)
           VALUES (?, ?, ?, ?, ?)""",
        (uid, year, repo_id, int(passed), score)
    )
    conn.commit()


def is_week_evaluated(conn, week: int) -> bool:
    row = conn.execute("SELECT 1 FROM eval_runs WHERE week=?", (week,)).fetchone()
    return row is not None


def mark_week_evaluated(conn, week: int):
    conn.execute("INSERT OR IGNORE INTO eval_runs (week) VALUES (?)", (week,))
    conn.commit()


def cleanup_after_uid(conn, uid: int):
    """Delete all evaluations created after the last evaluation of the given UID."""
    cur = conn.execute(
        "DELETE FROM evaluations WHERE evaluated_at > "
        "(SELECT MAX(evaluated_at) FROM evaluations WHERE uid = ?)",
        (uid,)
    )
    conn.commit()
    return cur.rowcount

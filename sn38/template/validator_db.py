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
            uid INTEGER, year INTEGER, repo_id TEXT, round INTEGER,
            passed INTEGER, score REAL,
            score_unknown REAL DEFAULT 0.0,
            score_known REAL DEFAULT 0.0,
            synced INTEGER DEFAULT 0,
            evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (uid, year, round)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            week INTEGER PRIMARY KEY,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    _migrate(conn, bt)
    return conn


def _migrate(conn, bt):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < 1:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(evaluations)").fetchall()}
        if "round" not in cols:
            bt.logging.info("Migration v1: adding round, score_unknown, score_known, synced")
            conn.execute("ALTER TABLE evaluations ADD COLUMN round INTEGER DEFAULT 2")
            conn.execute("ALTER TABLE evaluations ADD COLUMN score_unknown REAL DEFAULT 0.0")
            conn.execute("ALTER TABLE evaluations ADD COLUMN score_known REAL DEFAULT 0.0")
            conn.execute("ALTER TABLE evaluations ADD COLUMN synced INTEGER DEFAULT 0")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()


def get_cached_result(conn, uid: int, year: int, repo_id: str):
    """Returns (passed, score) or None if not cached."""
    row = conn.execute(
        "SELECT passed, score FROM evaluations WHERE uid=? AND year=? AND repo_id=?",
        (uid, year, repo_id)
    ).fetchone()
    if row is None:
        return None
    return bool(row[0]), row[1]


def save_result(conn, uid: int, year: int, repo_id: str, passed: bool, score: float = 0.0,
                score_unknown: float = 0.0, score_known: float = 0.0, eval_round: int = 0):
    conn.execute(
        """INSERT OR REPLACE INTO evaluations
           (uid, year, repo_id, passed, score, score_unknown, score_known, round, synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        (uid, year, repo_id, int(passed), score, score_unknown, score_known, eval_round)
    )
    conn.commit()


def is_week_evaluated(conn, week: int) -> bool:
    row = conn.execute("SELECT 1 FROM eval_runs WHERE week=?", (week,)).fetchone()
    return row is not None


def mark_week_evaluated(conn, week: int):
    conn.execute("INSERT OR IGNORE INTO eval_runs (week) VALUES (?)", (week,))
    conn.commit()


def get_unsynced_eval_details(conn):
    """Returns list of unsynced evaluations."""
    rows = conn.execute(
        "SELECT uid, year, repo_id, passed, score, score_unknown, score_known, round "
        "FROM evaluations WHERE synced = 0"
    ).fetchall()
    return [
        {"uid": r[0], "year": r[1], "repo_id": r[2], "passed": bool(r[3]),
         "score": r[4], "score_unknown": r[5], "score_known": r[6], "round": r[7]}
        for r in rows
    ]


def mark_synced(conn, uid: int, year: int, repo_id: str, eval_round: int):
    conn.execute(
        "UPDATE evaluations SET synced = 1 WHERE uid = ? AND year = ? AND repo_id = ? AND round = ?",
        (uid, year, repo_id, eval_round)
    )
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

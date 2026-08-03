"""SQLite cache for validator evaluation results."""

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.environ.get("DATA_DIR", "/app/data"), "validator_cache.db")


def get_connection():
    logger.info(f"Cache DB: {DB_PATH}")
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS svd_spectra (
            uid INTEGER,
            round INTEGER,
            spectra BLOB,
            PRIMARY KEY (uid, round)
        )
    """)
    conn.commit()
    _migrate(conn)
    return conn


def _migrate(conn):
    #version = conn.execute("PRAGMA user_version").fetchone()[0]
    pass

def save_svd_spectra(conn, uid: int, eval_round: int, spectra: dict):
    """Save SVD spectra for pairwise dedup."""
    import io
    import torch
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in spectra.items()}, buf)
    conn.execute(
        "INSERT OR REPLACE INTO svd_spectra (uid, round, spectra) VALUES (?, ?, ?)",
        (uid, eval_round, buf.getvalue())
    )
    conn.commit()


def load_all_svd_spectra(conn, eval_round: int, device="cpu"):
    """Load all SVD spectra for a round. Returns {uid: spectra}."""
    rows = conn.execute(
        "SELECT uid, spectra FROM svd_spectra WHERE round=?",
        (eval_round,)
    ).fetchall()
    import io
    import torch
    result = {}
    for uid, blob in rows:
        spectra = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        result[uid] = {k: v.to(device) for k, v in spectra.items()}
    return result


def get_cached_result(conn, uid: int, year: int, repo_id: str, eval_round: int):
    """Returns (passed, score) or None if not cached."""
    row = conn.execute(
        "SELECT passed, score FROM evaluations WHERE uid=? AND year=? AND repo_id=? AND round=?",
        (uid, year, repo_id, eval_round)
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

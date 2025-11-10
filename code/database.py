# database.py
"""SQLite database for logging VC card game sessions."""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Use Railway Volume if available, otherwise local directory
if os.path.exists("/data"):
    DB_FILE = Path("/data") / "game_data.db"
else:
    DB_FILE = Path(__file__).parent / "game_data.db"


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Game sessions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            timestamp_start DATETIME NOT NULL,
            timestamp_end DATETIME,
            seed INTEGER,
            signal_mode TEXT,
            signal_cost REAL,
            completed BOOLEAN DEFAULT 0
        )
    """)

    # Stage actions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stage_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            stage INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            purchases JSON,
            investments JSON,
            signals_spent REAL,
            stakes_invested REAL,
            budget_remaining REAL,
            FOREIGN KEY (session_id) REFERENCES game_sessions(id)
        )
    """)

    # Game results table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            total_invested REAL,
            total_payoff REAL,
            net_return REAL,
            net_return_pct REAL,
            n_invested INTEGER,
            ace_hits INTEGER,
            king_hits INTEGER,
            queen_hits INTEGER,
            player_weights JSON,
            FOREIGN KEY (session_id) REFERENCES game_sessions(id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"[db] Database initialized at {DB_FILE}")


def create_session(team_name: str, seed: int, signal_mode: str, signal_cost: float) -> int:
    """Create a new game session and return its ID."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO game_sessions (team_name, timestamp_start, seed, signal_mode, signal_cost)
        VALUES (?, ?, ?, ?, ?)
    """, (team_name, datetime.now(), seed, signal_mode, signal_cost))

    session_id = cur.lastrowid
    conn.commit()
    conn.close()

    print(f"[db] Created session {session_id} for team '{team_name}'")
    return session_id


def log_stage_action(
    session_id: int,
    stage: int,
    purchases: dict,
    investments: dict,
    signals_spent: float,
    stakes_invested: float,
    budget_remaining: float
):
    """Log actions taken in a stage."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO stage_actions (
            session_id, stage, timestamp,
            purchases, investments,
            signals_spent, stakes_invested, budget_remaining
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, stage, datetime.now(),
        json.dumps(purchases), json.dumps(investments),
        signals_spent, stakes_invested, budget_remaining
    ))

    conn.commit()
    conn.close()

    print(f"[db] Logged Stage {stage} actions for session {session_id}")


def log_game_results(session_id: int, results: dict):
    """Log final game results."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO game_results (
            session_id, total_invested, total_payoff,
            net_return, net_return_pct, n_invested,
            ace_hits, king_hits, queen_hits, player_weights
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        results.get("invested", 0.0),
        results.get("invested", 0.0) + results.get("net_return", 0.0),  # total_payoff = invested + net_return
        results.get("net_return", 0.0),
        results.get("net_return_pct", 0.0),
        results.get("n_invested", 0),
        results.get("ace_hits", 0),
        results.get("king_hits", 0),
        results.get("queen_hits", 0),
        json.dumps(results.get("player_weights", []))
    ))

    conn.commit()
    conn.close()

    print(f"[db] Logged results for session {session_id}")


def close_session(session_id: int):
    """Update session end timestamp."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        UPDATE game_sessions
        SET timestamp_end = ?
        WHERE id = ?
    """, (datetime.now(), session_id))

    conn.commit()
    conn.close()

    print(f"[db] Closed session {session_id}")


def mark_session_completed(session_id: int):
    """Mark a session as completed (player clicked End Game)."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        UPDATE game_sessions
        SET completed = 1
        WHERE id = ?
    """, (session_id,))

    conn.commit()
    conn.close()

    print(f"[db] Marked session {session_id} as completed")


def delete_session(session_id: int):
    """Delete a session and all its related data (for abandoned games)."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Delete related records first (foreign key constraint)
    cur.execute("DELETE FROM game_results WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM stage_actions WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM game_sessions WHERE id = ?", (session_id,))

    conn.commit()
    conn.close()

    print(f"[db] Deleted session {session_id} and all related data")


if __name__ == "__main__":
    # Test database creation
    init_db()
    print("[db] Database module ready")

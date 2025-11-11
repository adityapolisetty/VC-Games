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
    # Ensure directory exists for Railway volume
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[db] Database path: {DB_FILE}")
    print(f"[db] Database directory exists: {DB_FILE.parent.exists()}")

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Helper function to check if table exists
    def table_exists(table_name):
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cur.fetchone() is not None

    # Migration: Add 'completed' column if it doesn't exist (only if table exists)
    if table_exists('game_sessions'):
        try:
            cur.execute("SELECT completed FROM game_sessions LIMIT 1")
        except sqlite3.OperationalError:
            print("[db] Adding 'completed' column to game_sessions table...")
            cur.execute("ALTER TABLE game_sessions ADD COLUMN completed BOOLEAN DEFAULT 0")
            conn.commit()
            print("[db] Migration complete!")

    # Migration: Add 'concentration_index' and 'stage1_fraction' columns if they don't exist (only if table exists)
    if table_exists('game_results'):
        try:
            cur.execute("SELECT concentration_index FROM game_results LIMIT 1")
        except sqlite3.OperationalError:
            print("[db] Adding 'concentration_index' column to game_results table...")
            cur.execute("ALTER TABLE game_results ADD COLUMN concentration_index REAL")
            conn.commit()
            print("[db] Migration complete!")

        try:
            cur.execute("SELECT stage1_fraction FROM game_results LIMIT 1")
        except sqlite3.OperationalError:
            print("[db] Adding 'stage1_fraction' column to game_results table...")
            cur.execute("ALTER TABLE game_results ADD COLUMN stage1_fraction REAL")
            conn.commit()
            print("[db] Migration complete!")

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
            gross_return_mult REAL,
            net_return REAL,
            net_return_pct REAL,
            n_invested INTEGER,
            ace_hits INTEGER,
            king_hits INTEGER,
            queen_hits INTEGER,
            player_weights JSON,
            concentration_index REAL,
            stage1_fraction REAL,
            FOREIGN KEY (session_id) REFERENCES game_sessions(id)
        )
    """)

    conn.commit()

    # Migrate existing tables to add gross_return_mult if needed
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(game_results)")
        columns = [row[1] for row in cur.fetchall()]

        if 'gross_return_mult' not in columns:
            print("[db] Migrating game_results table to add gross_return_mult column...")
            cur.execute("ALTER TABLE game_results ADD COLUMN gross_return_mult REAL")
            conn.commit()
            print("[db] Migration completed successfully")
    except Exception as e:
        print(f"[db] Migration check/execution failed: {e}")

    conn.close()
    print(f"[db] Database initialized at {DB_FILE}")


def create_session(team_name: str, seed: int, signal_mode: str, signal_cost: float) -> int:
    """Create a new game session and return its ID."""
    try:
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
    except Exception as e:
        print(f"[db] ERROR creating session: {e}")
        print(f"[db] Database file: {DB_FILE}")
        print(f"[db] DB exists: {DB_FILE.exists()}")
        raise


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
            gross_return_mult, net_return, net_return_pct, n_invested,
            ace_hits, king_hits, queen_hits, player_weights,
            concentration_index, stage1_fraction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        results.get("invested", 0.0),
        results.get("invested", 0.0) + results.get("net_return", 0.0),  # total_payoff = invested + net_return
        results.get("gross_return_mult", 0.0),
        results.get("net_return", 0.0),
        results.get("net_return_pct", 0.0),
        results.get("n_invested", 0),
        results.get("ace_hits", 0),
        results.get("king_hits", 0),
        results.get("queen_hits", 0),
        json.dumps(results.get("player_weights", [])),
        results.get("concentration_index", None),
        results.get("stage1_fraction", None)
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


def get_leaderboard(limit: int = 10) -> list:
    """Get top players ranked by gross return multiplier.

    Only includes completed sessions with custom names (excludes 'Team Alpha').
    Filters to median signal games only for fair comparison.
    Returns list of dicts with: rank, team_name, gross_return_mult, n_invested
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    cur = conn.cursor()

    cur.execute("""
        SELECT
            gs.team_name,
            gr.gross_return_mult,
            gr.n_invested,
            gs.timestamp_end
        FROM game_results gr
        JOIN game_sessions gs ON gr.session_id = gs.id
        WHERE gs.completed = 1
          AND gs.team_name IS NOT NULL
          AND gs.team_name != ''
          AND gs.team_name != 'Team Alpha'
          AND gs.signal_mode = 'median'
        ORDER BY gr.gross_return_mult DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    conn.close()

    # Convert to list of dicts with rank
    leaderboard = []
    for i, row in enumerate(rows, start=1):
        leaderboard.append({
            'rank': i,
            'team_name': row['team_name'],
            'gross_return_mult': row['gross_return_mult'],
            'n_invested': row['n_invested'],
            'timestamp': row['timestamp_end']
        })

    print(f"[db] Retrieved median signal leaderboard with {len(leaderboard)} entries")
    return leaderboard


def get_leaderboard_by_signal_type(signal_type: str, limit: int = 10) -> list:
    """Get ALL players ranked by gross return multiplier for a specific signal type.

    Includes all completed sessions (no exclusions).
    Returns list of dicts with: rank, team_name, gross_return_mult, n_invested, total_signals
    Note: Fetches ALL entries, assigns proper ranks with tie handling, displays top `limit`
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    cur = conn.cursor()

    # Fetch ALL entries (no LIMIT in SQL), we'll handle ranking and display limit in Python
    # Sort by: gross_return_mult DESC, total_signals ASC (ties broken by lower signal cost), timestamp ASC
    cur.execute("""
        SELECT
            gs.team_name,
            (gr.total_payoff * 1.0 / gr.total_invested) AS gross_return_mult,
            gr.n_invested,
            gs.timestamp_end,
            (SELECT COALESCE(SUM(sa.signals_spent), 0)
             FROM stage_actions sa
             WHERE sa.session_id = gs.id) AS total_signals
        FROM game_results gr
        JOIN game_sessions gs ON gr.session_id = gs.id
        WHERE gs.completed = 1
          AND gs.team_name IS NOT NULL
          AND gs.team_name != ''
          AND gs.signal_mode = ?
        ORDER BY gross_return_mult DESC, total_signals ASC, gs.timestamp_end ASC
    """, (signal_type,))

    rows = cur.fetchall()
    conn.close()

    # Convert to list of dicts with proper ranking
    # Ties broken by signal cost (already in ORDER BY), so each position gets unique rank
    leaderboard = []
    for i, row in enumerate(rows, start=1):
        leaderboard.append({
            'rank': i,  # Sequential rank (ties already broken by signal cost)
            'team_name': row['team_name'],
            'gross_return_mult': row['gross_return_mult'],
            'n_invested': row['n_invested'],
            'total_signals': row['total_signals'],
            'timestamp': row['timestamp_end']
        })

    print(f"[db] Retrieved {signal_type} signal leaderboard with {len(leaderboard)} entries (all players ranked)")
    return leaderboard


if __name__ == "__main__":
    # Test database creation
    init_db()
    print("[db] Database module ready")

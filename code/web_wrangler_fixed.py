# web_wrangler.py - REFACTORED: Single persistent server architecture
import json
import pathlib
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import uuid

import pandas as pd

HTTPServer.allow_reuse_address = True
_BROWSER_OPENED = False  # guard to open browser only once per process

# REFACTORED: Single server instance with dynamic routing based on game state
_SERVER_INSTANCE = None  # Singleton server
_SERVER_LOCK = threading.Lock()  # Thread-safe server access
_SESSION_LOCK = threading.Lock()  # CRITICAL FIX: Thread-safe session data access
_GAME_STATE = {
    'stage': 0,  # Current game stage (0=not started, 1=stage1, 2=stage2, 3=results)
    'ctx': {},   # Current context data
    'ready': threading.Event(),  # Signals when stage transition is complete
}
_SESSION_EVENT = threading.Event()  # Signals when user submits data
_SESSION_DATA = None  # Stores submitted data

_HTML = pathlib.Path(__file__).with_name("stage_actions.html")
_ASSETS_DIR = pathlib.Path(__file__).with_name("assets")
_FONT = pathlib.Path(__file__).with_name("imperial.woff2")
_SSP_REG = pathlib.Path(__file__).with_name("SourceSansPro-Regular.woff2")
_SSP_REG_ALT = pathlib.Path(__file__).with_name("SourceSansPro-Regular.ttf.woff2")
_SSP_SEMIBOLD = pathlib.Path(__file__).with_name("SourceSansPro-Semibold.woff2")
_SSP_SEMIBOLD_ALT = pathlib.Path(__file__).with_name("SourceSansPro-Semibold.ttf.woff2")


def _prev_signals_map(df: pd.DataFrame) -> dict[int, list[int]]:
    out = {}
    for _, r in df.iterrows():
        sigs = []
        for k in (1, 2, 3, 4):
            v = r.get(f"s{k}")
            if pd.notna(v) and v is not None and str(v) != "None":
                sigs.append(k)
        if sigs:
            out[int(r["card_id"])] = sigs
    return out


def _prev_invest_map(df: pd.DataFrame) -> dict[int, float]:
    out = {}
    for _, r in df.iterrows():
        tot = float(r.get("inv1", 0) or 0) + float(r.get("inv2", 0) or 0) + float(r.get("inv3", 0) or 0)
        if tot > 0:
            out[int(r["card_id"])] = tot
    return out


class _H(BaseHTTPRequestHandler):
    """Request handler with dynamic routing based on game state"""

    def log_message(self, format, *args):
        """Suppress default logging to reduce noise"""
        pass

    def do_HEAD(self):
        """Handle HEAD requests for client polling"""
        global _GAME_STATE

        if self.path == "/results":
            # Client is checking if results are ready
            with _SERVER_LOCK:
                if _GAME_STATE['stage'] == 3 and _GAME_STATE['ready'].is_set():
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                else:
                    # Results not ready yet
                    self.send_response(503)  # Service Unavailable
                    self.end_headers()
            return

        # Default: method not allowed
        self.send_response(405)
        self.end_headers()

    def do_GET(self):
        global _GAME_STATE, _SESSION_DATA

        # Health check endpoint for Railway
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        # Assets handler
        if self.path.startswith("/assets/"):
            fname = self.path.rsplit("/", 1)[-1]
            p = _ASSETS_DIR / fname
            if p.exists():
                try:
                    data = p.read_bytes()
                except Exception:
                    self.send_error(404)
                    return
                ctype = "font/woff2" if p.suffix == ".woff2" else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self.end_headers()
                self.wfile.write(data)
                return

        # Font assets (legacy paths)
        if self.path == "/assets/imperial.woff2":
            try:
                data = _FONT.read_bytes()
            except FileNotFoundError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "font/woff2")
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/assets/SourceSansPro-Regular.woff2":
            try:
                data = _SSP_REG.read_bytes()
            except FileNotFoundError:
                try:
                    data = _SSP_REG_ALT.read_bytes()
                except FileNotFoundError:
                    self.send_error(404)
                    return
            self.send_response(200)
            self.send_header("Content-Type", "font/woff2")
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/assets/SourceSansPro-Semibold.woff2":
            try:
                data = _SSP_SEMIBOLD.read_bytes()
            except FileNotFoundError:
                try:
                    data = _SSP_SEMIBOLD_ALT.read_bytes()
                except FileNotFoundError:
                    self.send_error(404)
                    return
            self.send_response(200)
            self.send_header("Content-Type", "font/woff2")
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(data)
            return

        # Main page - ALWAYS serve landing page (Stage 0)
        if self.path == "/":
            html = _HTML.read_text(encoding="utf-8")

            # Always serve Stage 0 (landing page) at root
            seed = {
                "stage": 0,
                "totalBudget": 100.0,
                "budgetRemaining": 0.0,
                "cards": [],
                "prevSignals": {},
                "prevInvest": {},
                "stage_history": [],
                "stage1_invested": [],
            }

            html = html.replace("</head>", f"<script>window.SEED={json.dumps(seed)};</script></head>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            # CRITICAL: Aggressive cache-busting to prevent stale state
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        # Game page - serve active game (Stages 1-2)
        if self.path == "/game" or self.path.startswith("/game?"):
            with _SERVER_LOCK:
                ctx = _GAME_STATE['ctx'].copy()
                stage = _GAME_STATE['stage']

            if stage == 0:
                # No game active yet - redirect to landing
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return

            # Serve game with actual state
            html = _HTML.read_text(encoding="utf-8")
            seed = {
                "stage": ctx.get("stage", stage),
                "totalBudget": ctx.get("total_budget", 100.0),
                "budgetRemaining": ctx.get("wallet", 0.0),
                "cards": ctx.get("cards", []),
                "prevSignals": ctx.get("prev_signals", {}),
                "prevInvest": ctx.get("prev_invest", {}),
                "stage_history": ctx.get("stage_history", []),
                "stage1_invested": ctx.get("stage1_invested", []),
            }

            html = html.replace("</head>", f"<script>window.SEED={json.dumps(seed)};</script></head>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            # CRITICAL: Aggressive cache-busting to prevent stale state
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        # Results page
        if self.path == "/results":
            with _SERVER_LOCK:
                ctx = _GAME_STATE['ctx'].copy()
                stage = _GAME_STATE['stage']

            if stage != 3:
                # Results not ready yet
                self.send_response(503)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Results not ready</h1><p>Simulation in progress...</p></body></html>")
                return

            stats = ctx.get("results", {})
            page = _results_page(stats)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            # CRITICAL: Aggressive cache-busting to prevent stale results
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))
            return

        self.send_error(404)

    def do_POST(self):
        global _GAME_STATE, _SESSION_DATA, _SESSION_EVENT

        if self.path == "/submit":
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n).decode("utf-8"))

            # CRITICAL: Validate that server is at the correct stage
            submitted_stage = data.get("stage", -1)
            with _SERVER_LOCK:
                current_stage = _GAME_STATE['stage']

            # If submission doesn't match current stage, reject it
            if submitted_stage != current_stage:
                print(f"[server] WARNING: Rejected submission for stage {submitted_stage} (server at stage {current_stage})")
                self.send_response(409)  # Conflict
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {"status": "error", "message": f"Server not ready (at stage {current_stage}, expected {submitted_stage})"}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            # Store submitted data (CRITICAL FIX: Thread-safe access)
            with _SESSION_LOCK:
                _SESSION_DATA = data
            _SESSION_EVENT.set()

            # Update player name in results if applicable
            with _SERVER_LOCK:
                if isinstance(_GAME_STATE['ctx'].get("results"), dict):
                    _GAME_STATE['ctx']["results"]["player"] = data.get("player_name") or ""

            # Send success response
            print(f"[server] Accepted submission for stage {submitted_stage}")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"status": "success", "message": "Data received"}
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        if self.path == "/end":
            # Signal end of game (CRITICAL FIX: Thread-safe access)
            with _SESSION_LOCK:
                _SESSION_DATA = {"action": "end_game"}
            _SESSION_EVENT.set()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if self.path == "/reset":
            # Reset game state (used when user clicks restart or game ends)
            print("[server] Received /reset request - clearing all game state")

            # CRITICAL FIX: Signal the waiting game loop to exit properly
            # Set end_game marker so run_ui() returns None and game loop restarts
            with _SESSION_LOCK:
                _SESSION_DATA = {"action": "end_game"}
            _SESSION_EVENT.set()  # Wake up the waiting run_ui() call

            # Now reset game state for the new game
            reset_game_state()  # Use the proper reset function

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        self.send_error(404)


def _results_page(stats: dict) -> str:
    """Generate results HTML page"""
    hit_ace_label = "Yes" if (stats.get("ace_hits", 0) or 0) > 0 else "No"

    # Format gross return multipliers (0 decimals when 0, 2 decimals otherwise)
    g1_val = stats.get('g1', 0)
    g2_val = stats.get('g2', 0)
    gross_val = stats.get('gross_return_mult', 0)
    g1_formatted = f"{g1_val:.0f}" if g1_val == 0 else f"{g1_val:.2f}"
    g2_formatted = f"{g2_val:.0f}" if g2_val == 0 else f"{g2_val:.2f}"
    gross_formatted = f"{gross_val:.0f}" if gross_val == 0 else f"{gross_val:.2f}"

    # Format Ace/King/Queen payoff strings with stage breakdown
    # Show "-" when 0, otherwise show two-line format with * for times and x for multiplier

    # ACE
    ace_payoff_s1 = stats.get('ace_payoff_s1', 0)
    ace_invested_s1 = stats.get('ace_invested_s1', 0)
    if ace_payoff_s1 == 0:
        ace_s1_str = "-"
    else:
        ace_mult_s1 = ace_payoff_s1 / ace_invested_s1 if ace_invested_s1 > 0 else 0
        ace_s1_str = f'£{ace_payoff_s1:.0f}<br><span style="font-size:11px;color:#6b7280;">(£{ace_invested_s1:.0f} * {ace_mult_s1:.2f}x)</span>'

    ace_payoff_s2 = stats.get('ace_payoff_s2', 0)
    ace_invested_s2 = stats.get('ace_invested_s2', 0)
    if ace_payoff_s2 == 0:
        ace_s2_str = "-"
    else:
        ace_mult_s2 = ace_payoff_s2 / ace_invested_s2 if ace_invested_s2 > 0 else 0
        ace_s2_str = f'£{ace_payoff_s2:.0f}<br><span style="font-size:11px;color:#6b7280;">(£{ace_invested_s2:.0f} * {ace_mult_s2:.2f}x)</span>'

    ace_payoff_total = stats.get('ace_payoff', 0)
    ace_invested_total = stats.get('ace_invested', 0)
    if ace_payoff_total == 0:
        ace_total_str = "-"
    else:
        ace_total_str = f'£{ace_payoff_total:.0f}'

    # KING
    king_payoff_s1 = stats.get('king_payoff_s1', 0)
    king_invested_s1 = stats.get('king_invested_s1', 0)
    if king_payoff_s1 == 0:
        king_s1_str = "-"
    else:
        king_mult_s1 = king_payoff_s1 / king_invested_s1 if king_invested_s1 > 0 else 0
        king_s1_str = f'£{king_payoff_s1:.0f}<br><span style="font-size:11px;color:#6b7280;">(£{king_invested_s1:.0f} * {king_mult_s1:.2f}x)</span>'

    king_payoff_s2 = stats.get('king_payoff_s2', 0)
    king_invested_s2 = stats.get('king_invested_s2', 0)
    if king_payoff_s2 == 0:
        king_s2_str = "-"
    else:
        king_mult_s2 = king_payoff_s2 / king_invested_s2 if king_invested_s2 > 0 else 0
        king_s2_str = f'£{king_payoff_s2:.0f}<br><span style="font-size:11px;color:#6b7280;">(£{king_invested_s2:.0f} * {king_mult_s2:.2f}x)</span>'

    king_payoff_total = stats.get('king_payoff', 0)
    king_invested_total = stats.get('king_invested', 0)
    if king_payoff_total == 0:
        king_total_str = "-"
    else:
        king_total_str = f'£{king_payoff_total:.0f}'

    # QUEEN
    queen_payoff_s1 = stats.get('queen_payoff_s1', 0)
    queen_invested_s1 = stats.get('queen_invested_s1', 0)
    if queen_payoff_s1 == 0:
        queen_s1_str = "-"
    else:
        queen_mult_s1 = queen_payoff_s1 / queen_invested_s1 if queen_invested_s1 > 0 else 0
        queen_s1_str = f'£{queen_payoff_s1:.0f}<br><span style="font-size:11px;color:#6b7280;">(£{queen_invested_s1:.0f} * {queen_mult_s1:.2f}x)</span>'

    queen_payoff_s2 = stats.get('queen_payoff_s2', 0)
    queen_invested_s2 = stats.get('queen_invested_s2', 0)
    if queen_payoff_s2 == 0:
        queen_s2_str = "-"
    else:
        queen_mult_s2 = queen_payoff_s2 / queen_invested_s2 if queen_invested_s2 > 0 else 0
        queen_s2_str = f'£{queen_payoff_s2:.0f}<br><span style="font-size:11px;color:#6b7280;">(£{queen_invested_s2:.0f} * {queen_mult_s2:.2f}x)</span>'

    queen_payoff_total = stats.get('queen_payoff', 0)
    queen_invested_total = stats.get('queen_invested', 0)
    if queen_payoff_total == 0:
        queen_total_str = "-"
    else:
        queen_total_str = f'£{queen_payoff_total:.0f}'

    # Generate leaderboard rows
    leaderboard = stats.get("leaderboard", [])  # All entries with proper ranks
    signal_type_label = stats.get("signal_type_label", "")
    current_player_name = stats.get('player', '')

    if leaderboard:
        leaderboard_rows = ""
        current_player_in_top10 = False
        current_player_entry = None

        # Find current player in full leaderboard
        for entry in leaderboard:
            if entry['team_name'] == current_player_name:
                current_player_entry = entry
                break

        # Display top 10 only
        for i, entry in enumerate(leaderboard[:10]):
            # Check if this is the current player
            is_current_player = entry['team_name'] == current_player_name
            if is_current_player:
                current_player_in_top10 = True

            # Black background with white text for current player
            if is_current_player:
                row_style = 'background:#000000;color:#ffffff;'
                text_color = '#ffffff'
                return_color = '#ffffff'
            else:
                row_style = ''
                text_color = '#111827'
                return_color = '#059669' if entry.get('gross_return_mult', 0) >= 1.0 else '#c53030'

            leaderboard_rows += f"""
            <tr style="{row_style}">
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:center;font-weight:700;color:{text_color};">#{entry['rank']}</td>
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;color:{text_color};">{entry['team_name']}</td>
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:center;font-weight:700;color:{return_color};">{entry.get('gross_return_mult', 0):.2f}X</td>
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:center;color:{text_color};">£{entry.get('total_signals', 0):.0f}</td>
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:center;color:{text_color};">{entry.get('n_invested', 0)}</td>
            </tr>
            """

        # Add current player's stats if not in top 10
        if not current_player_in_top10 and current_player_entry:
            leaderboard_rows += f"""
            <tr style="background:#000000;color:#ffffff;border-top:2px solid #6b7280;">
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:center;font-weight:700;color:#ffffff;">#{current_player_entry['rank']}</td>
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#ffffff;font-weight:700;">{current_player_entry['team_name']} (You)</td>
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:center;font-weight:700;color:#ffffff;">{current_player_entry.get('gross_return_mult', 0):.2f}X</td>
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:center;color:#ffffff;">£{current_player_entry.get('total_signals', 0):.0f}</td>
              <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:center;color:#ffffff;">{current_player_entry.get('n_invested', 0)}</td>
            </tr>
            """

        leaderboard_empty_msg = ""
    else:
        leaderboard_rows = """
            <tr>
              <td colspan="5" style="padding:40px;text-align:center;color:#6b7280;">
                No players yet! Be the first to complete a {signal_type_label} signal game.
              </td>
            </tr>
        """
        leaderboard_empty_msg = ""

    # Serialize frontier data to JSON outside the f-string to avoid escaping issues
    frontier_data_json = json.dumps(stats.get('frontier_all_alphas', {}))
    frontier_data_v2_json = json.dumps(stats.get('frontier_all_alphas_v2', {}))
    player_position_json = json.dumps(stats.get('player_position', {}))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Performance - VC Card Games</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
  @font-face{{font-family:'Source Sans Pro'; font-style:normal; font-weight:400; font-display:swap; src: local('Source Sans Pro Regular'), local('SourceSansPro-Regular'), url('/assets/SourceSansPro-Regular.woff2') format('woff2'), url('/assets/SourceSansPro-Regular.ttf.woff2') format('woff2');}}
  @font-face{{font-family:'Source Sans Pro'; font-style:normal; font-weight:600; font-display:swap; src: local('Source Sans Pro Semibold'), local('SourceSansPro-Semibold'), url('/assets/SourceSansPro-Semibold.woff2') format('woff2'), url('/assets/SourceSansPro-Semibold.ttf.woff2') format('woff2');}}
  :root {{ --blue:#2b6cb0; --red:#c53030; --green:#059669; --bg:#ffffff; --panel:#ffffff; --cta:#f59e0b; --ctatxt:#111827; --b:#e5e7eb; --infoW:72px; }}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:"Source Sans Pro", system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif; background:var(--bg); color:#111827; -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale}}

  .brandbar{{display:flex;align-items:center;justify-content:center;gap:12px;padding:8px 12px;border-bottom:1px solid var(--b);background:var(--panel);position:sticky;top:0;z-index:20;font-size:22px}}
  .brand-center{{font-weight:900}}

  header.nav{{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;border-bottom:1px solid var(--b);background:var(--panel);position:sticky;top:40px;z-index:10}}
  .nav .title{{font-weight:600;font-size:14px;color:#111827}}
  .right{{display:flex;gap:16px}}
  .budget{{font-weight:700}}
  .muted{{color:#4b5563}}

  .wrap{{display:grid;grid-template-columns:1fr calc(420px - 170px + (var(--infoW)/2));gap:20px;padding:40px 20px 20px;margin-left:72px}}

  .content-panel{{border:1px solid var(--b);border-radius:12px;padding:20px;background:var(--panel)}}

  .tab-nav{{display:flex;gap:8px;margin-bottom:20px;border-bottom:2px solid var(--b);padding-bottom:8px}}
  .tab-btn{{padding:10px 16px;background:transparent;border:none;color:#6b7280;font-weight:600;font-size:15px;cursor:pointer;border-radius:6px 6px 0 0;transition:all .15s}}
  .tab-btn:hover{{background:#f3f4f6;color:#111827}}
  .tab-btn.active{{background:#111827;color:#e5e7eb}}

  .tab-content{{display:none}}
  .tab-content.active{{display:block}}

  .summary-box{{width:100%}}

  .stat-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
  .stat{{display:flex;justify-content:space-between;gap:10px;padding:12px;border:1px solid var(--b);border-radius:10px;background:#fafafa}}
  .stat-label{{color:#6b7280;font-size:14px}}
  .stat-value{{font-weight:700;color:#111827}}

  aside{{position:sticky;top:104px;height:fit-content;border:1px solid var(--b);border-radius:12px;padding:16px;background:var(--panel)}}

  .btn, .btn-cta, .btn-cta-sm, .btn-ghost {{cursor:pointer}}
  .btn{{padding:10px 14px;border-radius:10px;border:1px solid var(--b);background:#111827;color:#e5e7eb;transition:transform .15s, filter .15s, box-shadow .15s;font-weight:700}}
  .btn[disabled]{{opacity:.5;cursor:not-allowed}}
  .btn-cta, .btn-cta-sm, .btn-ghost{{padding:10px 14px;border-radius:10px;border:1px solid var(--b);background:#111827;color:#e5e7eb;width:auto;font-weight:700}}
  .btn-ghost{{min-width:50px;text-align:center;background:#ffffff;color:#111827;margin-left:16px}}
  .btn:hover, .btn-cta:hover, .btn-cta-sm:hover, .btn-ghost:hover{{transform:translateY(-1px);filter:brightness(1.1) saturate(1.1);box-shadow:0 6px 16px rgba(0,0,0,.18)}}
  .btn:active, .btn-cta:active, .btn-cta-sm:active, .btn-ghost:active{{transform:none;filter:none;box-shadow:none}}

  .overlay{{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;align-items:center;justify-content:center;z-index:50}}
  .overlay .msg{{background:var(--panel);border:1px solid var(--b);border-radius:14px;padding:24px 28px;font-weight:800}}

  #frontierChart{{width:100%;max-width:900px;height:600px;margin:0 auto}}
  .js-plotly-plot .plotly .main-svg{{overflow:visible !important}}

  /* Black slider styling */
  input[type="range"]{{
    -webkit-appearance:none;
    appearance:none;
    height:4px;
    background:#000000;
    border-radius:2px;
    outline:none;
  }}
  input[type="range"]::-webkit-slider-thumb{{
    -webkit-appearance:none;
    width:14px;
    height:14px;
    background:#ffffff;
    border:1px solid #000000;
    border-radius:50%;
    cursor:pointer;
  }}
  input[type="range"]::-moz-range-thumb{{
    width:14px;
    height:14px;
    background:#ffffff;
    border:1px solid #000000;
    border-radius:50%;
    cursor:pointer;
  }}
</style>
</head>
<body>

<div class="brandbar">
  <div class="brand-center">{stats.get('player','') or 'Team Alpha'}'s performance</div>
</div>

<header class="nav">
  <div class="title">End of Game</div>
  <div class="right">
    <div class="budget">Invested £{stats.get('invested',0):.2f}</div>
    <div class="budget">Signals £{stats.get('signals_spent',0):.2f}</div>
    <div class="budget">Remaining £{stats.get('wallet_left',0):.2f}</div>
  </div>
</header>

<div class="wrap">
  <section class="content-panel">
    <!-- Tab Navigation -->
    <div class="tab-nav">
      <button class="tab-btn active" data-tab="summary">Summary</button>
      <button class="tab-btn" data-tab="frontier">Frontier</button>
      <button class="tab-btn" data-tab="leaderboard">Leaderboard</button>
    </div>

    <!-- Summary Tab -->
    <div id="summary-tab" class="tab-content active">
      <!-- Info Boxes Row -->
      <div class="summary-box" style="margin-bottom:24px;">
        <div class="stat-grid">
          <div class="stat"><div class="stat-label">Budget</div><div class="stat-value">£100</div></div>
          <div class="stat"><div class="stat-label">Piles invested</div><div class="stat-value">{stats.get('n_invested',0)}</div></div>
          <div class="stat"><div class="stat-label">Spent on signals</div><div class="stat-value">£{stats.get('signals_spent',0):.0f}</div></div>
        </div>
      </div>

      <!-- Breakdown Table (Left) and Distribution (Right) -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start;">
        <!-- Left: Breakdown Table -->
        <div style="border:1px solid var(--b);border-radius:12px;overflow:hidden;background:var(--panel);">
          <div style="padding:20px 20px 16px 20px;border-bottom:1px solid var(--b);background:#f9fafb;">
            <h4 style="margin:0;color:#111827;">Breakdown</h4>
          </div>
          <div style="overflow:hidden;">
            <table style="width:100%;border-collapse:collapse;">
              <thead style="background:#f9fafb;border-bottom:2px solid var(--b);">
                <tr>
                  <th style="padding:12px 16px;text-align:left;font-size:14px;font-weight:700;color:#111827;">Metric</th>
                  <th style="padding:12px 16px;text-align:right;font-size:14px;font-weight:700;color:#111827;">Stage 1</th>
                  <th style="padding:12px 16px;text-align:right;font-size:14px;font-weight:700;color:#111827;">Stage 2</th>
                  <th style="padding:12px 16px;text-align:right;font-size:14px;font-weight:700;color:#111827;">Total</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;">Invested</td>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:right;font-weight:700;color:#111827;">£{stats.get('investable_stage1',0):.0f}</td>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:right;font-weight:700;color:#111827;">£{stats.get('investable_stage2',0):.0f}</td>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:right;font-weight:700;color:#111827;">£{stats.get('invested',0):.0f}</td>
                </tr>
                <tr>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;">Payoff</td>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:right;font-weight:700;color:#111827;">£{stats.get('stage1_payoff',0):.0f}</td>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:right;font-weight:700;color:#111827;">£{stats.get('stage2_payoff',0):.0f}</td>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:right;font-weight:700;color:#111827;">£{(stats.get('stage1_payoff',0) + stats.get('stage2_payoff',0)):.0f}</td>
                </tr>
                <tr>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;">Gross return (payoff/budget)</td>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:right;font-weight:700;color:{'#059669' if g1_val >= 1.0 else '#c53030'};">{g1_formatted}X</td>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:right;font-weight:700;color:{'#059669' if g2_val >= 1.0 else '#c53030'};">{g2_formatted}X</td>
                  <td style="padding:10px 16px;border-bottom:1px solid var(--b);font-size:14px;text-align:right;font-weight:700;color:{'#059669' if gross_val >= 1.0 else '#c53030'};">{gross_formatted}X</td>
                </tr>
                <tr>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#6b7280;font-weight:500;">Payoff from Ace Hits</td>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;text-align:right;">{ace_s1_str}</td>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;text-align:right;">{ace_s2_str}</td>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;text-align:right;">{ace_total_str}</td>
                </tr>
                <tr>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#6b7280;font-weight:500;">Payoff from King Hits</td>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;text-align:right;">{king_s1_str}</td>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;text-align:right;">{king_s2_str}</td>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;text-align:right;">{king_total_str}</td>
                </tr>
                <tr>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#6b7280;font-weight:500;">Payoff from Queen Hits</td>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;text-align:right;">{queen_s1_str}</td>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;text-align:right;">{queen_s2_str}</td>
                  <td style="padding:12px 16px;border-bottom:1px solid var(--b);font-size:14px;color:#111827;text-align:right;">{queen_total_str}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Right: Return Distribution Histogram -->
        <div style="border:1px solid var(--b);border-radius:12px;padding:20px;background:var(--panel);">
          <h4 style="margin:0 0 12px 0;color:#111827;">Distribution of Returns</h4>
          <p style="font-size:13px;color:#6b7280;margin:0 0 16px 0;">
            Based on 50,000 simulations of your allocation and signal acquisition strategy
          </p>
          <div id="histogramChart" style="width:100%;min-height:200px;"></div>
          <div style="margin-top:-8px;padding:12px;background:#f9fafb;border-radius:8px;font-size:13px;color:#6b7280;">
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">
              <div><strong>Mean Return:</strong> {stats.get('sim_metadata',{}).get('mean',0):.2f}X</div>
              <div><strong>Max Return:</strong> {stats.get('sim_metadata',{}).get('max',0):.2f}X</div>
              <div><strong>Std Dev:</strong> {stats.get('sim_metadata',{}).get('std',0):.2f}X</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Frontier Tab -->
    <div id="frontier-tab" class="tab-content">
      <!-- Slider controls -->
      <div style="margin-bottom:24px;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:40px;">
          <!-- Alpha slider -->
          <div>
            <label style="display:flex;justify-content:space-between;margin-bottom:8px;font-weight:600;font-size:13px;color:#111827;">
              <span>Stage 1 allocation</span>
              <span id="alphaValue" style="color:#000000;">10%</span>
            </label>
            <input type="range" id="alphaSlider" min="0" max="100" step="5" value="10"
                   style="width:100%;">
            <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:14px;color:#6b7280;">
              <span>0%</span>
              <span>100%</span>
            </div>
          </div>

          <!-- Signal count slider -->
          <div>
            <label style="display:flex;justify-content:space-between;margin-bottom:8px;font-weight:600;font-size:13px;color:#111827;">
              <span>Number of Signals</span>
              <span id="signalValue" style="color:#000000;">All</span>
            </label>
            <input type="range" id="signalSlider" min="0" max="10" step="1" value="10"
                   style="width:100%;">
            <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:14px;color:#6b7280;">
              <span>0 signals</span>
              <span>All</span>
            </div>
          </div>

          <!-- Frontier version toggle -->
          <div style="margin-top:16px;">
            <label style="display:flex;align-items:center;gap:8px;font-weight:600;font-size:13px;color:#111827;cursor:pointer;">
              <input type="checkbox" id="useFrontierV2" style="width:18px;height:18px;cursor:pointer;">
              <span>Use enhanced frontier (v2)</span>
            </label>
            <div style="font-size:12px;color:#6b7280;margin-top:4px;margin-left:26px;">
              V2 explores additional Stage 2 concentration strategies
            </div>
          </div>
        </div>
      </div>

      <div id="frontierChart"></div>

      <!-- Detail panel -->
      <div id="detailPanel" style="margin-top:16px;padding:16px;border:1px solid var(--b);border-radius:8px;background:#f9fafb;display:none;">
        <h4 style="margin:0 0 12px 0;color:#111827;">Strategy Details</h4>
        <div style="margin-bottom:8px;"><strong>Signals:</strong> <span id="detailSignals">-</span></div>
        <div style="margin-bottom:8px;"><strong>Portfolio Weights:</strong></div>
        <div id="detailWeights" style="font-family:monospace;font-size:13px;color:#6b7280;margin-bottom:12px;"></div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">
          <div style="padding:8px;background:white;border-radius:6px;text-align:center;">
            <div style="font-size:12px;color:#6b7280;">Ace hits</div>
            <div id="detailAce" style="font-size:18px;font-weight:700;color:#111827;">-</div>
          </div>
          <div style="padding:8px;background:white;border-radius:6px;text-align:center;">
            <div style="font-size:12px;color:#6b7280;">King hits</div>
            <div id="detailKing" style="font-size:18px;font-weight:700;color:#111827;">-</div>
          </div>
          <div style="padding:8px;background:white;border-radius:6px;text-align:center;">
            <div style="font-size:12px;color:#6b7280;">Queen hits</div>
            <div id="detailQueen" style="font-size:18px;font-weight:700;color:#111827;">-</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Leaderboard Tab -->
    <div id="leaderboard-tab" class="tab-content">
      <h3 style="margin-top:0;">Top 10 Players</h3>
      <p style="color:#6b7280;font-size:14px;margin:0 0 20px 0;">Ranked by gross return multiplier • Only {signal_type_label.lower()} signal games</p>

      <div style="max-width:900px;margin:0 auto;border:1px solid var(--b);border-radius:12px;overflow:hidden;background:var(--panel);">
        <table style="width:100%;border-collapse:collapse;">
          <thead style="background:#f9fafb;border-bottom:2px solid var(--b);">
            <tr>
              <th style="padding:12px 16px;text-align:center;font-size:14px;font-weight:700;color:#111827;white-space:nowrap;">Rank</th>
              <th style="padding:12px 16px;text-align:left;font-size:14px;font-weight:700;color:#111827;white-space:nowrap;">Player</th>
              <th style="padding:12px 16px;text-align:center;font-size:14px;font-weight:700;color:#111827;white-space:nowrap;">Gross Return</th>
              <th style="padding:12px 16px;text-align:center;font-size:14px;font-weight:700;color:#111827;white-space:nowrap;">Signal Cost</th>
              <th style="padding:12px 16px;text-align:center;font-size:14px;font-weight:700;color:#111827;white-space:nowrap;"># of Piles Invested</th>
            </tr>
          </thead>
          <tbody>
            {leaderboard_rows}
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <aside>
    <h3 style="margin:0 0 16px 0;">Actions</h3>
    <button id="endBtn" class="btn">Quit Game</button>
  </aside>
</div>

<div id="ov" class="overlay"><div class="msg">Hope you enjoyed the game</div></div>

<script>
// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const targetTab = btn.dataset.tab;

    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Update content
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(targetTab + '-tab').classList.add('active');
  }});
}});

// Quit game button - close tab and return to landing
document.getElementById('endBtn').onclick = () => {{
  const btn = document.getElementById('endBtn');
  const overlay = document.getElementById('ov');
  const overlayMsg = overlay.querySelector('.msg');

  btn.disabled = true;
  overlay.style.display = 'flex';
  overlayMsg.textContent = 'Ending game...';

  // CRITICAL: Clear localStorage so landing page doesn't auto-redirect
  localStorage.clear();
  sessionStorage.clear();

  // Send end signal to server
  fetch('/end', {{method:'POST'}})
    .then(() => {{
      overlayMsg.textContent = 'Closing tab...';

      // Try to close the tab (works since it was opened by window.open)
      setTimeout(() => {{
        window.close();

        // If window didn't close (shouldn't happen for tabs opened by JS), show message
        setTimeout(() => {{
          overlayMsg.innerHTML = 'Game ended!<br><span style="font-size:14px;font-weight:400;margin-top:8px;display:block;">You can close this tab now</span>';
        }}, 500);
      }}, 500);
    }})
    .catch((err) => {{
      console.error('[quit] Error ending game:', err);
      overlayMsg.textContent = 'Error ending game. You can close this tab.';
    }});
}};

// Create Plotly frontier chart (matching vis_f.py formatting)
function createFrontierChart() {{
  // Select the appropriate frontier dataset based on toggle
  const frontierData = window.USE_FRONTIER_V2 ? window.FRONTIER_DATA_V2 : window.FRONTIER_DATA;

  // Check if frontier data is available
  if (!frontierData || !window.PLAYER_POSITION) {{
    document.getElementById('frontierChart').innerHTML = '<div style="padding:40px;text-align:center;color:#6b7280;">Frontier data not available</div>';
    return;
  }}

  // Get slider elements
  const alphaSlider = document.getElementById('alphaSlider');
  const signalSlider = document.getElementById('signalSlider');
  const alphaValue = document.getElementById('alphaValue');
  const signalValue = document.getElementById('signalValue');

  // Set default alpha to player's alpha
  const playerAlphaPct = window.PLAYER_POSITION.alpha_pct || 10;
  alphaSlider.value = playerAlphaPct;
  alphaValue.textContent = playerAlphaPct + '%';

  // Function to render chart with current slider values
  function renderChart() {{
    const selectedAlpha = parseInt(alphaSlider.value);
    const selectedSignalFilter = parseInt(signalSlider.value); // 10 = All, 0-9 = specific n

    // Update display values
    alphaValue.textContent = selectedAlpha + '%';
    signalValue.textContent = selectedSignalFilter === 10 ? 'All' : selectedSignalFilter + ' signals';

    // Get frontier data for selected alpha (using the appropriate dataset)
    const activeFrontierData = window.USE_FRONTIER_V2 ? window.FRONTIER_DATA_V2 : window.FRONTIER_DATA;
    const frontierDataForAlpha = activeFrontierData[selectedAlpha];
    if (!frontierDataForAlpha) {{
      document.getElementById('frontierChart').innerHTML = `<div style="padding:40px;text-align:center;color:#6b7280;">No data for alpha=${{selectedAlpha}}%</div>`;
      return;
    }}

    const pointsByN = frontierDataForAlpha.points_by_n; // Array of 10 elements (n=0 to n=9)

    // Build Plotly traces
    const traces = [];
    const ALPHA = 0.7;

    // Black-to-grey colorscale (darker range: dark grey=low concentration, black=high concentration)
    const colorscale = [[0, '#606060'], [1, '#000000']];

    for (let n = 0; n < pointsByN.length; n++) {{
      const points = pointsByN[n];

      // Skip if no points for this n, or if filtered out
      if (points.length === 0) continue;
      if (selectedSignalFilter < 10 && n !== selectedSignalFilter) continue;

      // Extract data arrays (use SD of gross returns)
      const sdVals = points.map(p => p.sd_gross);
      const meanVals = points.map(p => p.mean_gross);
      const concentrationVals = points.map(p => p.concentration);

      // Build hover text
      const hoverTexts = points.map(p => {{
        // Build weight labels for each pile
        const weightLabels = p.weights.map((w, i) => `P${{i+1}}: ${{(w*100).toFixed(1)}}%`);
        // Split into 3 rows: piles 1-3, 4-6, 7-9
        const weightRows = [
          weightLabels.slice(0, 3).join(' | '),
          weightLabels.slice(3, 6).join(' | '),
          weightLabels.slice(6, 9).join(' | ')
        ];

        // Use total_rounds from NPZ metadata (default to 200k if not present)
        const totalRounds = p.total_rounds || 200000;
        const scalePay = p.scale_pay || 0;

        // Calculate hit percentages
        const aceHitPct = (p.ace_hit_rate / totalRounds * 100).toFixed(2);
        const kingHitPct = (p.king_hit_rate / totalRounds * 100).toFixed(2);
        const queenHitPct = (p.queen_hit_rate / totalRounds * 100).toFixed(2);

        // Build hit rate string (show king/queen only if scale_pay == 1)
        let hitRateStr = `Ace hits: ${{aceHitPct}}%`;
        if (scalePay === 1) {{
          hitRateStr += ` | King hits: ${{kingHitPct}}% | Queen hits: ${{queenHitPct}}%`;
        }}

        return `<b>Signals: N = ${{n}}</b><br>` +
               `<b>Simulations:</b> ${{totalRounds.toLocaleString()}}<br>` +
               `<b>Mean Return:</b> ${{p.mean_gross.toFixed(3)}}X<br>` +
               `<b>Std Dev:</b> ${{p.sd_gross.toFixed(3)}}X<br>` +
               `<b>Sharpe Ratio:</b> ${{p.sharpe.toFixed(3)}}<br>` +
               `<b>Σw²:</b> ${{p.concentration.toFixed(3)}}<br><br>` +
               `<b>Portfolio Weights:</b><br>` +
               weightRows.join('<br>') + '<br><br>' +
               hitRateStr;
      }});

      traces.push({{
        x: sdVals,
        y: meanVals,
        mode: 'markers+text',
        name: `n=${{n}}`,
        marker: {{
          size: 16,
          color: concentrationVals,
          colorscale: colorscale,
          showscale: false,
          line: {{ width: 0 }}
        }},
        text: Array(sdVals.length).fill(String(n)),
        textposition: 'middle center',
        textfont: {{ size: 11, color: 'white', family: 'monospace' }},
        hovertext: hoverTexts,
        hovertemplate: '%{{hovertext}}<extra></extra>',
        showlegend: false,
        opacity: ALPHA,
        customdata: points.map(p => ({{ weights: p.weights, n: n, ace: p.ace_hit_rate, king: p.king_hit_rate, queen: p.queen_hit_rate }}))
      }});
    }}

    // Add player marker (red, same style as other markers)
    const playerSDGross = window.PLAYER_POSITION.sd_pct / 100.0; // Convert to gross SD
    const playerMean = window.PLAYER_POSITION.mean_gross;
    const playerN = window.PLAYER_POSITION.n_signals;
    const playerMaxGross = window.PLAYER_POSITION.max_gross || playerMean;
    const playerSharpe = window.PLAYER_POSITION.sharpe || 0;
    const playerWeights = window.PLAYER_POSITION.weights || [0,0,0,0,0,0,0,0,0];
    const playerConcentration = window.PLAYER_POSITION.concentration || 0;
    const playerAceHits = window.PLAYER_POSITION.ace_hits || 0;
    const playerKingHits = window.PLAYER_POSITION.king_hits || 0;
    const playerQueenHits = window.PLAYER_POSITION.queen_hits || 0;

    // Build player hovertext with simulation data
    const playerWeightLabels = playerWeights.map((w, i) => `P${{i+1}}: ${{(w*100).toFixed(1)}}%`);
    const playerWeightRows = [
      playerWeightLabels.slice(0, 3).join(' | '),
      playerWeightLabels.slice(3, 6).join(' | '),
      playerWeightLabels.slice(6, 9).join(' | ')
    ];

    const playerHoverText = `<b>Signals: N = ${{playerN}}</b><br>` +
                            `<b>Simulations:</b> 50,000<br>` +
                            `<b>Mean Return:</b> ${{playerMean.toFixed(3)}}X<br>` +
                            `<b>Max Return:</b> ${{playerMaxGross.toFixed(3)}}X<br>` +
                            `<b>Std Dev:</b> ${{playerSDGross.toFixed(3)}}X<br>` +
                            `<b>Sharpe Ratio:</b> ${{playerSharpe.toFixed(3)}}<br>` +
                            `<b>Σw²:</b> ${{playerConcentration.toFixed(3)}}<br><br>` +
                            `<b>Portfolio Weights:</b><br>` +
                            playerWeightRows.join('<br>') + '<br><br>' +
                            `Ace hits: ${{(playerAceHits / 50000 * 100).toFixed(2)}}% | ` +
                            `King hits: ${{(playerKingHits / 50000 * 100).toFixed(2)}}% | ` +
                            `Queen hits: ${{(playerQueenHits / 50000 * 100).toFixed(2)}}%`;

    traces.push({{
      x: [playerSDGross],
      y: [playerMean],
      mode: 'markers+text',
      name: 'You',
      marker: {{
        size: 16,
        color: '#FF0000',
        line: {{ width: 0 }}
      }},
      text: [String(playerN)],
      textposition: 'middle center',
      textfont: {{ size: 11, color: 'white', family: 'monospace' }},
      hovertext: [playerHoverText],
      hoverinfo: 'text',
      hoverlabel: {{ bgcolor: '#FF0000', font: {{ color: '#ffffff', family: 'Source Sans Pro', size: 13 }} }},
      showlegend: false,
      opacity: 1.0
    }});

    const layout = {{
      template: 'plotly_white',
      font: {{ family: 'Source Sans Pro, Arial, sans-serif', size: 14 }},
      xaxis: {{
        title: {{ text: 'SD of gross return multiple', font: {{ size: 16 }} }},
        tickfont: {{ size: 14 }},
        ticksuffix: 'X',
        showgrid: true,
        gridcolor: 'rgba(128,128,128,0.15)'
      }},
      yaxis: {{
        title: {{ text: 'Mean gross return multiple', font: {{ size: 16 }} }},
        tickfont: {{ size: 14 }},
        ticksuffix: 'X',
        showgrid: true,
        gridcolor: 'rgba(128,128,128,0.15)'
      }},
      hoverlabel: {{
        bgcolor: '#000000',
        font: {{ color: '#ffffff', family: 'Source Sans Pro', size: 13 }},
        align: 'left'
      }},
      height: 600,
      hovermode: 'closest',
      margin: {{ l: 70, r: 20, t: 40, b: 60 }},
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#fafafa'
    }};

    const config = {{ responsive: true, displayModeBar: false }};

    Plotly.newPlot('frontierChart', traces, layout, config);
  }}

  // Initial render
  renderChart();

  // Slider event listeners
  alphaSlider.addEventListener('input', renderChart);
  signalSlider.addEventListener('input', renderChart);
}}

// Create 5-bin distribution table with player indicator
function createHistogramChart() {{
  const simReturns = {json.dumps(stats.get('sim_returns', []))};
  const playerReturn = {stats.get('gross_return_mult', 0):.3f};

  console.log('[histogram] Simulation returns count:', simReturns.length);
  console.log('[histogram] Player return:', playerReturn);

  // Check if simulation data exists
  if (!simReturns || simReturns.length === 0) {{
    document.getElementById('histogramChart').innerHTML = '<div style="padding:40px;text-align:center;color:#6b7280;">No simulation data available</div>';
    return;
  }}

  // Always create exactly 5 bins
  const numBins = 5;
  const minVal = Math.min(...simReturns);
  const maxVal = Math.max(...simReturns);

  console.log('[histogram] Min:', minVal, 'Max:', maxVal);

  // Handle edge case where all returns are identical
  let binSize = (maxVal - minVal) / numBins;
  if (binSize === 0 || !isFinite(binSize)) {{
    binSize = 0.1; // Fallback for identical values
  }}

  // Create bins
  const bins = [];
  for (let i = 0; i < numBins; i++) {{
    const binMin = minVal + i * binSize;
    const binMax = minVal + (i + 1) * binSize;
    bins.push({{
      min: binMin,
      max: binMax,
      count: 0,
      label: `${{binMin.toFixed(2)}}× - ${{binMax.toFixed(2)}}×`
    }});
  }}

  // Count values in each bin
  const totalCount = simReturns.length;
  simReturns.forEach(val => {{
    if (binSize > 0 && isFinite(val)) {{
      let binIndex = Math.floor((val - minVal) / binSize);
      binIndex = Math.max(0, Math.min(numBins - 1, binIndex)); // Clamp to valid range
      bins[binIndex].count++;
    }} else {{
      // All values identical - put everything in first bin
      bins[0].count++;
    }}
  }});

  // Determine which bin the player is in
  let playerBinIndex = -1;
  if (isFinite(playerReturn)) {{
    if (binSize > 0) {{
      let binIndex = Math.floor((playerReturn - minVal) / binSize);
      playerBinIndex = Math.max(0, Math.min(numBins - 1, binIndex));
    }} else {{
      playerBinIndex = 0; // All in first bin
    }}
  }}

  console.log('[histogram] Player bin index:', playerBinIndex);
  console.log('[histogram] Bin counts:', bins.map(b => b.count));

  // Create table HTML
  let tableHTML = `
    <div style="max-height: 400px; overflow-y: auto; margin-bottom: 0; font-family: 'Source Sans Pro', system-ui, Arial, sans-serif;">
      <table style="width: 100%; border-collapse: collapse; background: #ffffff; margin-bottom: 0;">
        <thead style="position: sticky; top: 0; background: #f9fafb; z-index: 10;">
          <tr style="border-bottom: 2px solid #e5e7eb;">
            <th style="padding: 12px 16px; text-align: left; font-size: 14px; font-weight: 700; color: #111827;">Gross Return Range</th>
            <th style="padding: 12px 16px; text-align: right; font-size: 14px; font-weight: 700; color: #111827;">Frequency</th>
            <th style="padding: 12px 16px; text-align: right; font-size: 14px; font-weight: 700; color: #111827;">Probability</th>
          </tr>
        </thead>
        <tbody>
  `;

  bins.forEach((bin, index) => {{
    // Skip bins with zero count
    if (bin.count === 0) {{
      return;
    }}

    const probability = (bin.count / totalCount * 100).toFixed(2);
    const isPlayerBin = index === playerBinIndex;

    const rowStyle = isPlayerBin
      ? 'background: #000000; color: #ffffff;'
      : (index % 2 === 0 ? 'background: #ffffff;' : 'background: #f9fafb;');

    const textColor = isPlayerBin ? '#ffffff' : '#111827';
    const cellColor = isPlayerBin ? '#ffffff' : '#6b7280';

    tableHTML += `
      <tr style="${{rowStyle}}">
        <td style="padding: 10px 16px; font-size: 14px; border-bottom: 1px solid #e5e7eb; color: ${{textColor}}; font-weight: ${{isPlayerBin ? '700' : '400'}}">${{bin.label}}${{isPlayerBin ? ' ← Your return' : ''}}</td>
        <td style="padding: 10px 16px; font-size: 14px; text-align: right; border-bottom: 1px solid #e5e7eb; color: ${{cellColor}};">${{bin.count.toLocaleString()}}</td>
        <td style="padding: 10px 16px; font-size: 14px; text-align: right; border-bottom: 1px solid #e5e7eb; color: ${{cellColor}};">${{probability}}%</td>
      </tr>
    `;
  }});

  tableHTML += `
        </tbody>
      </table>
    </div>
  `;

  document.getElementById('histogramChart').innerHTML = tableHTML;
  console.log('[histogram] Created 5-bin distribution table');
}}

// Create histogram on page load
if (document.getElementById('histogramChart')) {{
  createHistogramChart();
}}

// Create chart when frontier tab is visible
document.querySelector('[data-tab="frontier"]').addEventListener('click', function() {{
  if (!document.getElementById('frontierChart').innerHTML) {{
    createFrontierChart();
  }}
}});

// Add event listener for v2 toggle
const v2Toggle = document.getElementById('useFrontierV2');
if (v2Toggle) {{
  v2Toggle.addEventListener('change', function() {{
    window.USE_FRONTIER_V2 = this.checked;
    createFrontierChart();  // Redraw chart with new dataset
  }});
}}

// Inject frontier data and player position from Python backend
window.FRONTIER_DATA = {frontier_data_json};
window.FRONTIER_DATA_V2 = {frontier_data_v2_json};
window.PLAYER_POSITION = {player_position_json};
window.USE_FRONTIER_V2 = false;  // Default to v1
</script>
</body>
</html>"""


def reset_game_state():
    """Reset game state to fresh defaults (called when starting new game)"""
    global _GAME_STATE, _SESSION_DATA, _SESSION_EVENT
    with _SERVER_LOCK:
        _GAME_STATE['stage'] = 0
        _GAME_STATE['ctx'] = {}
        _GAME_STATE['ready'].clear()
    # CRITICAL FIX: Thread-safe session data reset
    with _SESSION_LOCK:
        _SESSION_DATA = None
    # IMPORTANT: Don't clear the event here - let run_ui() handle it
    # Clearing it here can cause race conditions where the game loop
    # is still waiting on the event when /reset is called
    print("[server] Game state reset to defaults (stage=0, ctx cleared)")


def update_game_state(stage: int, ctx: dict):
    """Update game state for the persistent server"""
    global _GAME_STATE
    with _SERVER_LOCK:
        _GAME_STATE['stage'] = stage
        _GAME_STATE['ctx'] = ctx.copy()
        _GAME_STATE['ready'].set()  # Signal that state is ready
    print(f"[server] Game state updated: stage={stage}")


def start_persistent_server(port: int = 8765, open_browser: bool = False):
    """Start the persistent HTTP server (called once at startup)"""
    import os
    global _SERVER_INSTANCE, _BROWSER_OPENED

    with _SERVER_LOCK:
        if _SERVER_INSTANCE is not None:
            print("[server] Persistent server already running")
            return _SERVER_INSTANCE

        # Use PORT from environment (Railway) or fallback to default
        port = int(os.environ.get("PORT", port))
        host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"

        # Create server
        try:
            srv = HTTPServer((host, port), _H)
        except OSError as e:
            print(f"[server] Failed to bind to port {port}: {e}")
            raise

        # Start server thread
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        _SERVER_INSTANCE = srv

        actual_port = srv.server_address[1]
        url = f"http://127.0.0.1:{actual_port}/"
        print(f"[server] Persistent server started at {url}")

        # Open browser once
        if open_browser and not _BROWSER_OPENED:
            try:
                webbrowser.open(url)
                _BROWSER_OPENED = True
            except Exception:
                pass

        return srv


def run_ui(stage: int, df: pd.DataFrame, wallet: float, *, results: dict | None = None,
           port: int = 8765, open_browser: bool = False,
           signal_mode: str = "median", signal_cost: float = 3.0,
           stage1_invested: list | None = None, stage_history: list | None = None,
           session_id: int | None = None):
    """REFACTORED: Update game state and wait for user submission

    No longer creates/destroys servers - just updates state on persistent server.
    """
    global _SESSION_DATA, _SESSION_EVENT, _GAME_STATE

    # Ensure persistent server is running
    if _SERVER_INSTANCE is None:
        start_persistent_server(port=port, open_browser=open_browser)

    # Reset session state (CRITICAL FIX: Thread-safe access)
    with _SESSION_LOCK:
        _SESSION_DATA = None
    _SESSION_EVENT.clear()

    # Build context (handle Stage 0 with no cards)
    cards = []
    if not df.empty:
        cols = [c for c in ("card_id", "N", "med", "sum2", "second_rank") if c in df.columns]
        for _, r in df[cols].iterrows():
            rec = {"card_id": int(r.get("card_id"))}
            if "med" in cols: rec["med"] = int(r.get("med"))
            if "sum2" in cols: rec["sum2"] = int(r.get("sum2"))
            if "N" in cols: rec["N"] = int(r.get("N"))
            if "second_rank" in cols: rec["second_rank"] = int(r.get("second_rank"))
            cards.append(rec)

    ctx = {
        "stage": stage,
        "total_budget": 100.0,
        "wallet": float(wallet),
        "cards": cards,
        "prev_signals": _prev_signals_map(df),
        "prev_invest": _prev_invest_map(df),
        "results": results or {},
        "signal_mode": str(signal_mode),
        "signal_cost": float(signal_cost),
        "stage1_invested": stage1_invested or [],
        "stage_history": stage_history or [],
        "session_id": session_id,
    }

    # Update game state on persistent server
    update_game_state(stage, ctx)
    print(f"[web] Stage {stage} ready, waiting for user action...")

    # Wait for user submission
    if results and stage >= 3:
        # Results stage: wait for end game click
        print(f"[web] Results stage - waiting for 'End Game' click")
        if not _SESSION_EVENT.wait(timeout=300):  # 5 minute timeout
            print(f"[web] Warning: Results stage timed out")
            # CRITICAL FIX: Clean up stale state on timeout to prevent memory leak
            with _SESSION_LOCK:
                _SESSION_DATA = None
            _SESSION_EVENT.clear()
            return None
    else:
        # Game stage: wait for POST submission
        if not _SESSION_EVENT.wait(timeout=300):
            print(f"[web] Warning: Stage {stage} timed out waiting for submission")
            # CRITICAL FIX: Clean up stale state on timeout to prevent memory leak
            with _SESSION_LOCK:
                _SESSION_DATA = None
            _SESSION_EVENT.clear()
            return None

    # Get submitted data (CRITICAL FIX: Thread-safe access)
    with _SESSION_LOCK:
        actions = _SESSION_DATA

    # Check if this is an end_game signal (restart/quit)
    if actions and actions.get("action") == "end_game":
        print(f"[web] Stage {stage} received end_game signal - returning None to restart game loop")
        return None

    print(f"[web] Stage {stage} complete, returning actions")
    return actions


def shutdown_server():
    """Shutdown the persistent server (called on application exit)

    NOTE: Currently unused but kept for future graceful shutdown implementation.
    To use, register with atexit or signal handlers for clean exits.
    """
    global _SERVER_INSTANCE
    with _SERVER_LOCK:
        if _SERVER_INSTANCE:
            print("[server] Shutting down persistent server...")
            _SERVER_INSTANCE.shutdown()
            _SERVER_INSTANCE.server_close()
            _SERVER_INSTANCE = None
            print("[server] Server shutdown complete")

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

    # Generate leaderboard rows
    leaderboard = stats.get("leaderboard", [])
    signal_type_label = stats.get("signal_type_label", "")

    if leaderboard:
        leaderboard_rows = ""
        for entry in leaderboard:
            # Highlight current player if they're on the leaderboard
            row_style = 'background:#fef3c7;font-weight:700;' if entry['team_name'] == stats.get('player', '') else ''
            leaderboard_rows += f"""
            <tr style="{row_style}">
              <td style="padding:12px 16px;border-bottom:1px solid var(--b);text-align:center;font-weight:700;color:#111827;">#{entry['rank']}</td>
              <td style="padding:12px 16px;border-bottom:1px solid var(--b);color:#111827;">{entry['team_name']}</td>
              <td style="padding:12px 16px;border-bottom:1px solid var(--b);text-align:right;font-weight:700;color:{'#059669' if entry['net_return_pct'] >= 0 else '#c53030'};">{entry['net_return_pct']:.2f}%</td>
              <td style="padding:12px 16px;border-bottom:1px solid var(--b);text-align:right;color:#6b7280;">{entry['n_invested']}</td>
            </tr>
            """
        leaderboard_empty_msg = ""
    else:
        leaderboard_rows = """
            <tr>
              <td colspan="4" style="padding:40px;text-align:center;color:#6b7280;">
                No players yet! Be the first to complete a {signal_type_label} signal game.
              </td>
            </tr>
        """
        leaderboard_empty_msg = ""

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
  .tab-btn{{padding:10px 16px;background:transparent;border:none;color:#6b7280;font-weight:600;cursor:pointer;border-radius:6px 6px 0 0;transition:all .15s}}
  .tab-btn:hover{{background:#f3f4f6;color:#111827}}
  .tab-btn.active{{background:#111827;color:#e5e7eb}}

  .tab-content{{display:none}}
  .tab-content.active{{display:block}}

  .summary-box{{max-width:520px}}

  .stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
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

  #frontierChart{{width:100%;height:600px}}
  .js-plotly-plot .plotly .main-svg{{overflow:visible !important}}
</style>
</head>
<body>

<div class="brandbar">
  <div class="brand-center">{stats.get('player','') or 'Team Alpha'} Performance</div>
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
      <button class="tab-btn" data-tab="frontier">Frontier Analysis</button>
      <button class="tab-btn" data-tab="leaderboard">Leaderboard</button>
    </div>

    <!-- Summary Tab -->
    <div id="summary-tab" class="tab-content active">
      <h3 style="margin-top:0;">Performance Summary</h3>
      <div style="display:grid;grid-template-columns:520px 1fr;gap:24px;align-items:start;">
        <!-- Left: Stats Grid -->
        <div class="summary-box">
          <div class="stat-grid">
            <div class="stat"><div class="stat-label">Budget</div><div class="stat-value">£100.00</div></div>
            <div class="stat"><div class="stat-label">Total invested</div><div class="stat-value">£{stats.get('invested',0):.2f}</div></div>
            <div class="stat"><div class="stat-label">Net return on budget</div><div class="stat-value">{stats.get('net_return_pct',0):.2f}%</div></div>
            <div class="stat"><div class="stat-label">Spent on signals</div><div class="stat-value">£{stats.get('signals_spent',0):.2f}</div></div>
            <div class="stat"><div class="stat-label">Piles invested</div><div class="stat-value">{stats.get('n_invested',0)}</div></div>
            <div class="stat"><div class="stat-label">No. Ace hits</div><div class="stat-value">{hit_ace_label}</div></div>
            <div class="stat"><div class="stat-label">No. King hits</div><div class="stat-value">{stats.get('king_hits',0)}</div></div>
            <div class="stat"><div class="stat-label">No. Queen hits</div><div class="stat-value">{stats.get('queen_hits',0)}</div></div>
          </div>
        </div>

        <!-- Right: Return Distribution Histogram -->
        <div style="border:1px solid var(--b);border-radius:12px;padding:20px;background:var(--panel);">
          <h4 style="margin:0 0 12px 0;color:#111827;">Distribution of Returns</h4>
          <p style="font-size:13px;color:#6b7280;margin:0 0 16px 0;">
            Based on 10,000 simulations with your allocation strategy ({stats.get('sim_metadata',{}).get('n_signals',0)} {stats.get('sim_metadata',{}).get('signal_type','')} signals)
          </p>
          <div id="histogramChart" style="width:100%;height:350px;"></div>
          <div style="margin-top:16px;padding:12px;background:#f9fafb;border-radius:8px;font-size:13px;color:#6b7280;">
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;">
              <div><strong>Mean Return:</strong> {stats.get('sim_metadata',{}).get('mean',0):.2f}%</div>
              <div><strong>Std Dev:</strong> {stats.get('sim_metadata',{}).get('std',0):.2f}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Frontier Tab -->
    <div id="frontier-tab" class="tab-content">
      <h3 style="margin-top:0;">Mean-Variance Frontier</h3>
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
      <h3 style="margin-top:0;">Top 10 Players ({signal_type_label} Signal)</h3>
      <p style="color:#6b7280;font-size:14px;margin:0 0 20px 0;">Ranked by net return percentage • Only {signal_type_label.lower()} signal games • Custom names only</p>

      <div style="max-width:640px;margin:0 auto;border:1px solid var(--b);border-radius:12px;overflow:hidden;background:var(--panel);">
        <table style="width:100%;border-collapse:collapse;">
          <thead style="background:#f9fafb;border-bottom:2px solid var(--b);">
            <tr>
              <th style="padding:14px 20px;text-align:center;font-weight:700;color:#111827;width:70px;">Rank</th>
              <th style="padding:14px 20px;text-align:left;font-weight:700;color:#111827;">Player</th>
              <th style="padding:14px 20px;text-align:right;font-weight:700;color:#111827;width:120px;">Net Return</th>
              <th style="padding:14px 20px;text-align:center;font-weight:700;color:#111827;width:90px;">Piles</th>
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
  // Mockup frontier data (n=0 to n=5 signals)
  const frontierData = [
    {{ n: 0, sd: [15, 16, 17, 18, 19], mean: [5, 8, 10, 11, 12], weights: [[11,11,11,11,11,11,11,11,12], [11,11,11,11,11,11,11,11,12], [11,11,11,11,11,11,11,11,12], [11,11,11,11,11,11,11,11,12], [11,11,11,11,11,11,11,11,12]], ace: 0, king: 0, queen: 0 }},
    {{ n: 1, sd: [14, 15, 16, 17], mean: [12, 15, 17, 18], weights: [[20,15,15,10,10,10,10,5,5], [20,15,15,10,10,10,10,5,5], [20,15,15,10,10,10,10,5,5], [20,15,15,10,10,10,10,5,5]], ace: 1, king: 2, queen: 1 }},
    {{ n: 2, sd: [13, 14, 15], mean: [18, 22, 24], weights: [[25,20,15,12,10,8,5,3,2], [25,20,15,12,10,8,5,3,2], [25,20,15,12,10,8,5,3,2]], ace: 1, king: 3, queen: 2 }},
    {{ n: 3, sd: [12, 13], mean: [25, 28], weights: [[28,22,18,12,8,6,3,2,1], [28,22,18,12,8,6,3,2,1]], ace: 2, king: 3, queen: 2 }},
    {{ n: 4, sd: [11, 12], mean: [30, 32], weights: [[30,25,18,12,8,4,2,1,0], [30,25,18,12,8,4,2,1,0]], ace: 2, king: 4, queen: 3 }},
  ];

  // Player data
  const playerWeights = {stats.get('player_weights', '[0,0,0,0,0,0,0,0,0]')};
  const playerAce = {stats.get('ace_hits', 0)};
  const playerKing = {stats.get('king_hits', 0)};
  const playerQueen = {stats.get('queen_hits', 0)};

  // Calculate player position (mockup - place near middle)
  const playerSD = 13.5;
  const playerMean = 20;

  // Calculate color values (concentration measure)
  function calcConcentration(weights) {{
    return weights.reduce((sum, w) => sum + w*w, 0) / 10000; // Normalize to [0,1]
  }}

  const traces = [];
  const ALPHA = 0.7;

  // Add frontier traces
  frontierData.forEach(series => {{
    const colors = series.weights.map(w => calcConcentration(w));

    traces.push({{
      x: series.sd,
      y: series.mean,
      mode: 'markers+text',
      name: `n=${{series.n}}`,
      marker: {{
        size: 16,
        color: colors,
        colorscale: [[0, '#2b8cbe'], [1, '#08306b']],
        showscale: false,
        line: {{ width: 0 }}
      }},
      text: Array(series.sd.length).fill(String(series.n)),
      textposition: 'middle center',
      textfont: {{ size: 11, color: 'white' }},
      hovertemplate: `n=${{series.n}}<br>Mean: %{{y:.2f}}%<br>SD: %{{x:.2f}}%<extra></extra>`,
      showlegend: false,
      opacity: ALPHA,
      customdata: series.weights.map((w, i) => ({{ weights: w, n: series.n, ace: series.ace, king: series.king, queen: series.queen }}))
    }});
  }});

  // Add player marker (red)
  traces.push({{
    x: [playerSD],
    y: [playerMean],
    mode: 'markers+text',
    name: 'You',
    marker: {{
      size: 20,
      color: '#c53030',
      line: {{ width: 2, color: '#fff' }}
    }},
    text: ['You'],
    textposition: 'middle center',
    textfont: {{ size: 11, color: 'white', weight: 700 }},
    hovertemplate: 'Your Strategy<br>Mean: %{{y:.2f}}%<br>SD: %{{x:.2f}}%<extra></extra>',
    showlegend: false,
    opacity: 0.9,
    customdata: [{{ weights: playerWeights, n: 'Player', ace: playerAce, king: playerKing, queen: playerQueen }}]
  }});

  const layout = {{
    template: 'plotly_white',
    font: {{ family: 'Roboto, Arial, sans-serif', size: 15 }},
    xaxis: {{
      title: {{ text: 'Standard Deviation (%)', font: {{ size: 13 }} }},
      tickfont: {{ size: 16 }},
      showgrid: true,
      gridcolor: 'rgba(128,128,128,0.1)'
    }},
    yaxis: {{
      title: {{ text: 'Mean Return (%)', font: {{ size: 13 }} }},
      tickfont: {{ size: 16 }},
      showgrid: true,
      gridcolor: 'rgba(128,128,128,0.1)'
    }},
    height: 600,
    hovermode: 'closest',
    margin: {{ l: 60, r: 10, t: 40, b: 50 }},
    plot_bgcolor: '#fafafa'
  }};

  const config = {{ responsive: true, displayModeBar: false }};

  Plotly.newPlot('frontierChart', traces, layout, config);

  // Add click handler for showing details
  document.getElementById('frontierChart').on('plotly_click', function(data) {{
    if (data.points.length > 0) {{
      const point = data.points[0];
      const customData = point.customdata;

      if (customData) {{
        const detailPanel = document.getElementById('detailPanel');
        document.getElementById('detailSignals').textContent = customData.n === 'Player' ? 'Your Strategy' : customData.n;
        document.getElementById('detailWeights').innerHTML = customData.weights.map((w, i) => `Pile ${{i+1}}: £${{w.toFixed(2)}}`).join('<br>');
        document.getElementById('detailAce').textContent = customData.ace;
        document.getElementById('detailKing').textContent = customData.king;
        document.getElementById('detailQueen').textContent = customData.queen;
        detailPanel.style.display = 'block';
      }}
    }}
  }});
}}

// Create histogram chart or probability table based on unique values
function createHistogramChart() {{
  const simReturns = {json.dumps(stats.get('sim_returns', []))};
  const playerReturn = {stats.get('net_return_pct', 0):.2f};

  console.log('[histogram] Simulation returns count:', simReturns.length);
  console.log('[histogram] Player return:', playerReturn);

  // Check if simulation data exists
  if (!simReturns || simReturns.length === 0) {{
    document.getElementById('histogramChart').innerHTML = '<div style="padding:40px;text-align:center;color:#6b7280;">No simulation data available</div>';
    return;
  }}

  // Count unique values and their frequencies
  const valueCounts = {{}};
  simReturns.forEach(val => {{
    const rounded = Math.round(val * 10) / 10;  // Round to 1 decimal place
    valueCounts[rounded] = (valueCounts[rounded] || 0) + 1;
  }});

  const uniqueValues = Object.keys(valueCounts).map(Number).sort((a, b) => a - b);
  const numUnique = uniqueValues.length;

  console.log('[histogram] Unique values:', numUnique);

  // Decision: histogram (>=50 unique) or probability table (<50 unique)
  if (numUnique >= 50) {{
    // Create 50-bin histogram using Plotly
    const minVal = Math.min(...simReturns);
    const maxVal = Math.max(...simReturns);
    const binSize = (maxVal - minVal) / 50;

    // Create bins
    const bins = new Array(50).fill(0);
    const binEdges = [];
    for (let i = 0; i <= 50; i++) {{
      binEdges.push(minVal + i * binSize);
    }}

    // Count values in each bin
    simReturns.forEach(val => {{
      const binIndex = Math.min(49, Math.floor((val - minVal) / binSize));
      bins[binIndex]++;
    }});

    // Calculate bin centers for x-axis
    const binCenters = [];
    for (let i = 0; i < 50; i++) {{
      binCenters.push((binEdges[i] + binEdges[i + 1]) / 2);
    }}

    // Create Plotly bar chart
    const trace = {{
      x: binCenters,
      y: bins,
      type: 'bar',
      marker: {{
        color: '#000000',  // Black bars
        line: {{ width: 0 }}
      }},
      width: binSize * 0.95,  // 95% of bin width for small gaps
      hovertemplate: 'Return: %{{x:.1f}}%<br>Frequency: %{{y}}<extra></extra>'
    }};

    const layout = {{
      plot_bgcolor: '#ffffff',  // White background
      paper_bgcolor: '#ffffff',
      font: {{ family: '"Source Sans Pro", system-ui, Arial, sans-serif', size: 12, color: '#111827' }},
      xaxis: {{
        title: {{ text: 'Net Return (%)', font: {{ size: 13, color: '#111827' }} }},
        tickfont: {{ size: 11, color: '#6b7280' }},
        showgrid: true,
        gridcolor: '#e5e7eb',
        zeroline: true,
        zerolinecolor: '#9ca3af',
        zerolinewidth: 1.5
      }},
      yaxis: {{
        title: {{ text: 'Frequency', font: {{ size: 13, color: '#111827' }} }},
        tickfont: {{ size: 11, color: '#6b7280' }},
        showgrid: true,
        gridcolor: '#e5e7eb'
      }},
      margin: {{ l: 60, r: 20, t: 20, b: 50 }},
      showlegend: false,
      bargap: 0
    }};

    const config = {{
      displayModeBar: false,
      responsive: true
    }};

    Plotly.newPlot('histogramChart', [trace], layout, config);

    // Add vertical line for player's actual return
    const shapes = [{{
      type: 'line',
      x0: playerReturn,
      x1: playerReturn,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: {{
        color: '#c53030',  // Red line
        width: 3,
        dash: 'dash'
      }}
    }}];

    const annotations = [{{
      x: playerReturn,
      y: 1,
      yref: 'paper',
      text: `Your Return: ${{playerReturn.toFixed(1)}}%`,
      showarrow: true,
      arrowhead: 2,
      arrowsize: 1,
      arrowwidth: 2,
      arrowcolor: '#c53030',
      ax: 0,
      ay: -40,
      font: {{ size: 11, color: '#c53030', weight: 700 }},
      bgcolor: '#ffffff',
      bordercolor: '#c53030',
      borderwidth: 1,
      borderpad: 4
    }}];

    Plotly.relayout('histogramChart', {{ shapes: shapes, annotations: annotations }});
  }} else {{
    // Create probability table for <50 unique values
    const totalCount = simReturns.length;
    const playerRounded = Math.round(playerReturn * 10) / 10;

    let tableHTML = `
      <div style="max-height: 400px; overflow-y: auto; font-family: 'Source Sans Pro', system-ui, Arial, sans-serif;">
        <table style="width: 100%; border-collapse: collapse; background: #ffffff;">
          <thead style="position: sticky; top: 0; background: #f9fafb; z-index: 10;">
            <tr style="border-bottom: 2px solid #e5e7eb;">
              <th style="padding: 12px 16px; text-align: left; font-size: 13px; font-weight: 600; color: #111827;">Return (%)</th>
              <th style="padding: 12px 16px; text-align: right; font-size: 13px; font-weight: 600; color: #111827;">Frequency</th>
              <th style="padding: 12px 16px; text-align: right; font-size: 13px; font-weight: 600; color: #111827;">Probability</th>
            </tr>
          </thead>
          <tbody>
    `;

    uniqueValues.forEach((value, index) => {{
      const count = valueCounts[value];
      const probability = (count / totalCount * 100).toFixed(2);
      const isPlayerReturn = Math.abs(value - playerRounded) < 0.01;

      const rowStyle = isPlayerReturn
        ? 'background: #fee2e2; border-left: 4px solid #c53030;'
        : (index % 2 === 0 ? 'background: #ffffff;' : 'background: #f9fafb;');

      const returnStyle = isPlayerReturn
        ? 'font-weight: 700; color: #c53030;'
        : 'color: #111827;';

      tableHTML += `
        <tr style="${{rowStyle}} border-bottom: 1px solid #e5e7eb;">
          <td style="padding: 10px 16px; font-size: 13px; ${{returnStyle}}">
            ${{value.toFixed(1)}}${{isPlayerReturn ? ' ← Your Return' : ''}}
          </td>
          <td style="padding: 10px 16px; text-align: right; font-size: 13px; color: #6b7280;">
            ${{count.toLocaleString()}}
          </td>
          <td style="padding: 10px 16px; text-align: right; font-size: 13px; color: #6b7280;">
            ${{probability}}%
          </td>
        </tr>
      `;
    }});

    tableHTML += `
          </tbody>
        </table>
      </div>
    `;

    document.getElementById('histogramChart').innerHTML = tableHTML;
  }}
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
           signal_mode: str = "median", signal_cost: float = 5.0,
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

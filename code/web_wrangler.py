# web_wrangler.py
import json
import pathlib
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket

import pandas as pd

HTTPServer.allow_reuse_address = True
_BROWSER_OPENED = False  # guard to open browser only once per process

_ACTIONS = None
_POSTED = threading.Event()
_END_EVENT = threading.Event()

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
    def do_GET(self):
        if self.path == "/favicon.ico":
            # Quietly satisfy browsers without error spam
            self.send_response(204)
            self.end_headers()
            return
        # Generic assets handler (serve from ./assets if present)
        if self.path.startswith("/assets/"):
            fname = self.path.rsplit("/", 1)[-1]
            p = _ASSETS_DIR / fname
            if p.exists():
                try:
                    data = p.read_bytes()
                except Exception:
                    self.send_error(404); return
                ctype = "font/woff2" if p.suffix == ".woff2" else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self.end_headers()
                self.wfile.write(data)
                return
        if self.path == "/assets/imperial.woff2":
            try:
                data = _FONT.read_bytes()
            except FileNotFoundError:
                self.send_error(404); return
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
                    self.send_error(404); return
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
                    self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", "font/woff2")
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/":
            html = _HTML.read_text(encoding="utf-8")
            seed = {
                "stage": self.server.ctx["stage"],
                "totalBudget": self.server.ctx["total_budget"],
                "budgetRemaining": self.server.ctx["wallet"],
                "cards": self.server.ctx["cards"],  # [{card_id,color,N}]
                "prevSignals": self.server.ctx["prev_signals"],
                "prevInvest": self.server.ctx["prev_invest"],
                "stage_history": self.server.ctx.get("stage_history", []),
                "stage1_invested": self.server.ctx.get("stage1_invested", []),
            }
            html = html.replace("</head>", f"<script>window.SEED={json.dumps(seed)};</script></head>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/results":
            stats = self.server.ctx.get("results", {})
            page = _results_page(stats)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))
            return

        self.send_error(404)

    def do_POST(self):
        global _ACTIONS
        if self.path == "/submit":
            n = int(self.headers.get("Content-Length", "0"))
            _ACTIONS = json.loads(self.rfile.read(n).decode("utf-8"))
            _POSTED.set()
            if isinstance(self.server.ctx.get("results"), dict):
                self.server.ctx["results"]["player"] = _ACTIONS.get("player_name") or ""
            self.send_response(200)
            self.end_headers()
            return

        if self.path == "/end":
            _END_EVENT.set()
            self.send_response(200)
            self.end_headers()
            return

        self.send_error(404)


def _results_page(stats: dict) -> str:
    # sys-wipe pretty label
    sw = stats.get("sys_wipe", "none")
    sw_label = {"none": "No", "blue": "Blue", "red": "Red", "both": "Blue & Red"}.get(sw, str(sw))

    return f"""<!doctype html><html><head>
<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Performance</title>
<style>
:root{{--bg:#0f172a;--panel:#0b1220;--b:#1f2937;--cta:#f59e0b;--ctat:#111827;--fg:#e5e7eb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}}
.brandbar{{display:flex;align-items:center;justify-content:center;gap:12px;padding:8px 12px;border-bottom:1px solid var(--b);background:#0a1020}}
.brand-left{{position:absolute;left:12px;font-weight:900}}
header{{display:flex;justify-content:space-between;align-items:center;padding:12px 20px;border-bottom:1px solid var(--b);background:var(--panel)}}
.hdr-right{{text-align:right}}
.hdr-right .meta{{font-size:12px;opacity:.85;line-height:1.3}}
.wrap{{max-width:960px;margin:20px auto;padding:0 20px}}
.card{{background:var(--panel);border:1px solid var(--b);border-radius:12px;padding:16px;margin:10px 0}}
.btn{{padding:14px 16px;font-weight:800;border-radius:12px;border:2px solid var(--cta);background:var(--cta);color:var(--ctat);cursor:pointer;transition:filter .15s,transform .15s,box-shadow .15s}}
.btn:hover{{filter:brightness(1.07) saturate(1.06);transform:translateY(-1px);box-shadow:0 6px 16px rgba(0,0,0,.35)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.stat{{display:flex;justify-content:space-between;gap:10px;padding:10px 12px;border:1px solid var(--b);border-radius:10px}}
.stat div:first-child{{opacity:.8}}
.overlay{{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;z-index:50}}
.overlay .box{{background:var(--panel);border:1px solid var(--b);border-radius:12px;padding:20px 24px;font-weight:800}}
</style></head><body>
<div class='brandbar'><div class='brand-left'>Imperial</div><div>{stats.get('player','')}</div></div>

<header>
  <div style='font-weight:800'>Performance</div>
  <div class='hdr-right'>
    <div class='meta'>
      Invested £{stats.get('invested',0):.2f} •
      Signals £{stats.get('signals_spent',0):.2f} •
      Remaining £{stats.get('wallet_left',0):.2f}
    </div>
  </div>
</header>

<div class='wrap'>
  <div class='card'>
    <h3>Summary</h3>
    <div class='grid2'>
      <div class='stat'><div>Total invested</div><div>£{stats.get('invested',0):.2f}</div></div>
      <div class='stat'><div>Spent on signals</div><div>£{stats.get('signals_spent',0):.2f}</div></div>
      <div class='stat'><div>Net return %</div><div>{stats.get('net_return_pct',0):.2f}%</div></div>
      <div class='stat'><div>Cards invested</div><div>{stats.get('n_invested',0)}</div></div>
      <div class='stat'><div>Invested cards wiped out</div><div>{stats.get('n_wiped',0)}</div></div>
      <div class='stat'><div>Systemic wipe-out</div><div>{sw_label}</div></div>
      <div class='stat'><div>Red cards invested</div><div>{stats.get('n_red_invested',0)}</div></div>
      <div class='stat'><div>Avg signals per invested card</div><div>{stats.get('avg_signals',0):.2f}</div></div>
    </div>
  </div>

  <button id='endBtn' class='btn' style='width:100%'>End Game</button>
</div>

<div id="ov" class="overlay"><div class="box">Hope you enjoyed the game</div></div>

<script>
document.getElementById('endBtn').onclick = () => {{
  const btn = document.getElementById('endBtn');
  btn.disabled = true;
  document.getElementById('ov').style.display = 'flex';
  fetch('/end', {{method:'POST'}}).catch(()=>{{}});
  setTimeout(()=>{{ window.close(); }}, 3000);
}};
</script>
</body></html>"""

def run_ui(stage: int, df: pd.DataFrame, wallet: float, *, results: dict | None = None,
           port: int = 8765, open_browser: bool = False,
           signal_mode: str = "median", signal_cost: float = 5.0,
           stage1_invested: list | None = None, stage_history: list | None = None):
    """Serve UI for a stage and return the posted decisions.

    stage1_invested: list of card_ids that were invested in Stage 1 (for Stage 2 restrictions)
    stage_history: list of dicts with 'signals' and 'stakes' for each completed stage
    """
    global _ACTIONS
    _ACTIONS = None
    _POSTED.clear()
    _END_EVENT.clear()

    # Build lightweight cards with optional median/top2 and second_rank fields
    cards_df = df.loc[df["alive"], :].copy()
    cols = [c for c in ("card_id", "color", "N", "med", "sum2", "second_rank") if c in cards_df.columns]
    cards = []
    for _, r in cards_df[cols].iterrows():
        rec = {"card_id": int(r.get("card_id")), "color": str(r.get("color", "blue"))}
        if "med" in cols: rec["med"] = int(r.get("med"))
        if "sum2" in cols: rec["sum2"] = int(r.get("sum2"))
        if "N" in cols: rec["N"] = int(r.get("N"))
        if "second_rank" in cols: rec["second_rank"] = int(r.get("second_rank"))
        cards.append(rec)

    ctx = {
        "stage": stage,
        "total_budget": 100.0,  # UI label only
        "wallet": float(wallet),
        "cards": cards,
        "prev_signals": _prev_signals_map(df),
        "prev_invest": _prev_invest_map(df),
        "results": results or {},
        "signal_mode": str(signal_mode),
        "signal_cost": float(signal_cost),
        "stage1_invested": stage1_invested or [],  # card_ids invested in Stage 1
        "stage_history": stage_history or [],  # history of previous stages
    }

    # Bind server; if requested port is busy, fall back to an ephemeral port
    try:
        srv = HTTPServer(("127.0.0.1", port), _H)
    except OSError as e:
        # Address in use — fallback
        try:
            srv = HTTPServer(("127.0.0.1", 0), _H)
        except OSError:
            raise e
    srv.ctx = ctx

    threading.Thread(target=srv.serve_forever, daemon=True).start()
    actual_port = getattr(srv, 'server_address', (None, port))[1]
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"[web] Serving stage {stage} UI at {url}")
    global _BROWSER_OPENED
    if open_browser and not _BROWSER_OPENED:
        try:
            webbrowser.open(url)
            _BROWSER_OPENED = True
        except Exception:
            pass

    _POSTED.wait()  # wait for stage POST

    if results:  # Results stage: wait for end event
        _END_EVENT.wait(timeout=3600)
        time.sleep(0.2)

    srv.shutdown()
    return _ACTIONS

# sim_viz.py — Interactive viewer for card_game_v7 NPZ outputs
# Two‑panel comparison with additional tabs:
#   Frontier, Value of Info, Distributions, Posteriors, One‑Off
# Reads simulation NPZs from the local `output/` directory and an external
# posterior NPZ for the Posteriors/One‑Off views.

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from numpy.random import default_rng

# Import core model primitives for one-off analysis
from card_game import (
    _deal_cards_global_deck,
    round_seed,
    run_single_round,
    _load_mc_posteriors as _load_mc_full,
    NUM_PILES,
    ACE_RANK,
)


# ==============================
# App config: wide + light CSS
# ==============================
st.set_page_config(page_title="IB Card Game", layout="wide")
st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp, .block-container { background: #ffffff !important; }
      .block-container { padding-top: 1.25rem; padding-bottom: 1.0rem; }
      .stSelectbox label, .stNumberInput label { font-size: 0.92rem; margin-bottom: .18rem; }
      .stSelectbox > div[data-baseweb="select"] { min-height: 36px; }
      .stTabs [data-baseweb="tab-list"] { gap: .25rem; }
      .stTabs [data-baseweb="tab"] { padding: .35rem .7rem; }
      .js-plotly-plot .plotly .main-svg { overflow: visible !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Constants (must match v6)
# ==============================
OUTPUT_DIR = Path("output/").resolve()

BLUE = "#141450"   # Highest expected payoff
GREY = "#808080"   # E(payoff) weighted
RED  = "#8B0000"   # Top‑5 E(payoff) weighted
ALPHA = 0.7

CANON_SCALE_PARAM_DEFAULT = 0

SIGNAL_COSTS = [0, 3, 5, 7, 9, 11]
SCALE_PAYS   = [0, 1]
SCALE_PARAMS = [1/3, 1/4, 1/5, 1/6]
ACE_PAYOUTS  = [10, 20, 30, 40]

SIG_LABELS = ["Median", "Top 2 rank sum", "Max Rank", "Min Rank"]
SIG_KEY = {"Median": "median", "Top 2 rank sum": "top2", "Max Rank": "max", "Min Rank": "min"}

# External posterior NPZ (from precompute_posteriors.py)
POST_NPZ_DEFAULT = str((Path("output") / "post_mc.npz").resolve())

@st.cache_data(show_spinner=False)
def load_post_npz(npz_path: str):
    """Strict loader for empirical posterior NPZ. No fallbacks.

    If required arrays are missing, emits a clear error with the missing list
    and returns None so the caller can display a simple info panel.
    """
    p = Path(npz_path).resolve()
    if not p.exists():
        st.error(f"Posterior NPZ not found: {p}")
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            req = {
                "rmax_median_keys", "rmax_median_mat",
                "rmax_top2_keys",  "rmax_top2_mat",
                "rmax_max_keys",   "rmax_max_mat",
                "rmax_min_keys",   "rmax_min_mat",
                "ace_median_probs", "ace_top2_probs", "ace_max_probs", "ace_min_probs",
                "prior_rmax",
            }
            zset = set(z.files)
            missing = sorted(list(req - zset))
            if missing:
                st.error(
                    "Posterior NPZ is missing required arrays: " + ", ".join(missing)
                )
                st.caption("Re-run precompute_posteriors.py to regenerate post_mc.npz with the updated schema.")
                return None
            return {
                "med_x": np.asarray(z["rmax_median_keys"], float),
                "med_y": np.asarray(z["ace_median_probs"], float),
                "t2_x":  np.asarray(z["rmax_top2_keys"], float),
                "t2_y":  np.asarray(z["ace_top2_probs"], float),
                "mx_x":  np.asarray(z["rmax_max_keys"], float),
                "mx_y":  np.asarray(z["ace_max_probs"], float),
                "mn_x":  np.asarray(z["rmax_min_keys"], float),
                "mn_y":  np.asarray(z["ace_min_probs"], float),
            }
    except Exception as e:
        st.error(f"Failed to read posterior NPZ: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_full_post(npz_path: str):
    """Load full posterior tables and prior via card_game helper.

    Returns (rmax_median, rmax_top2, rmax_max, rmax_min, prior_rmax, meta_curves) or None on error.
    """
    try:
        return _load_mc_full(npz_path)
    except Exception as e:
        st.error(f"Failed to load full posteriors: {e}")
        return None

# ==============================
# Helpers: canonical ID builder
# ==============================
def _fmt_float_id(x: float, nd=4) -> str:
    s = f"{x:.{nd}f}"
    return s.replace(".", "p")

def _canonicalize(raw):
    sc  = float(raw["signal_cost"])
    sp  = int(raw["scale_pay"])
    s   = float(raw["scale_param"])
    ap  = float(raw["ace_payout"])

    s_norm  = s  if sp == 1 else CANON_SCALE_PARAM_DEFAULT

    key_tuple = (("sc", sc), ("sp", sp), ("s", s_norm), ("ap", ap))
    parts = [f"{k}{_fmt_float_id(v) if isinstance(v,float) else v}" for k,v in key_tuple]
    key_id = "_".join(parts)
    norm = {"signal_cost": sc, "scale_pay": sp, "scale_param": s_norm, "ace_payout": ap}
    return norm, key_id

def _file_for_params(raw):
    norm, key_id = _canonicalize(raw)
    p = (OUTPUT_DIR / f"{key_id}.npz").resolve()
    if not str(p).startswith(str(OUTPUT_DIR)):
        raise ValueError(f"NPZ path must be inside {OUTPUT_DIR}")
    return p, norm, key_id

def _label_for_regime_key(k: str) -> str:
    for lbl, kk in SIG_KEY.items():
        if kk == k:
            return lbl
    return k

# ==============================
# Lazy loader
# ==============================
@st.cache_data(show_spinner=False)
def load_keys(npz_path: str, keys: tuple[str, ...]):
    p = Path(npz_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    if not str(p).startswith(str(OUTPUT_DIR)):
        raise FileNotFoundError(f"Only files inside {OUTPUT_DIR} are allowed.")
    out = {}
    with np.load(p, allow_pickle=False) as z:
        zfiles = set(z.files)
        for k in keys:
            out[k] = z[k] if k in zfiles else None
    return out

# ==============================
# Plot helpers
# ==============================
def _ticks_100(vals):
    a = np.asarray(vals, float)
    if a.size == 0:
        return None
    lo = np.floor(a.min() / 100.0) * 100.0
    hi = np.ceil(a.max()  / 100.0) * 100.0
    if lo == hi:
        hi = lo + 100.0
    return np.arange(lo, hi + 0.1, 100.0)

def _ticks_step2(vals):
    """Even ticks every 2, spanning the data."""
    a = np.asarray(vals, float)
    if a.size == 0:
        return None
    lo = 2.0 * np.floor(a.min() / 2.0)
    hi = 2.0 * np.ceil(a.max()  / 2.0)
    if lo == hi:
        hi = lo + 2.0
    return np.arange(lo, hi + 1e-9, 2.0)

_DEF_FONT = dict(size=15)

def _padded_range(lo, hi, frac=0.03, min_pad=0.5):
    """Pad a [lo, hi] range on both sides to avoid clipping markers/lines."""
    lo = float(lo); hi = float(hi)
    if not np.all(np.isfinite([lo, hi])):
        return [lo, hi]
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    base = span if span > 0 else max(abs(lo), abs(hi), 1.0)
    pad = max(min_pad, frac * base)
    return [lo - pad, hi + pad]

def frontier_plot(sig_grid, sd_triplet, mean_triplet, title, y_range=None):
    sd_max, sd_lin, sd_sq = sd_triplet
    mu_max, mu_lin, mu_sq = mean_triplet

    fig = go.Figure()
    msize = [16] * len(sig_grid)

    # Legend order: E(payoff), Top‑5 E(payoff), Highest expected payoff
    fig.add_trace(go.Scatter(
        x=np.asarray(sd_lin, float) ** 2, y=mu_lin, mode="markers+text",
        marker=dict(size=msize, color=GREY, line=dict(width=0)),
        text=[str(int(v)) for v in sig_grid], textposition="middle center",
        textfont=dict(color="black", size=11), name="𝔼[payoff] weighted", opacity=ALPHA
    ))
    fig.add_trace(go.Scatter(
        x=np.asarray(sd_sq, float) ** 2, y=mu_sq, mode="markers+text",
        marker=dict(size=msize, color=RED, line=dict(width=0)),
        text=[str(int(v)) for v in sig_grid], textposition="middle center",
        textfont=dict(color="white", size=11), name="Top‑5 𝔼[payoff] weighted", opacity=ALPHA
    ))
    fig.add_trace(go.Scatter(
        x=np.asarray(sd_max, float) ** 2, y=mu_max, mode="markers+text",
        marker=dict(size=msize, color=BLUE, line=dict(width=0)),
        text=[str(int(v)) for v in sig_grid], textposition="middle center",
        textfont=dict(color="white", size=11), name="Highest 𝔼[payoff]", opacity=ALPHA
    ))

    # Build y-axis config with optional fixed range
    yaxis_cfg = dict(
        title=dict(text="Mean return (%)", font=_DEF_FONT),
        tickmode="auto",
        tickfont=_DEF_FONT,
    )
    if y_range is None:
        yaxis_cfg.update(autorange=True)
    else:
        yaxis_cfg.update(autorange=False, range=list(map(float, y_range)))

    fig.update_layout(
        template="plotly_white",
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0.5,
            xanchor="center",
            font=_DEF_FONT,
            itemsizing="constant",
            itemwidth=114  # ~3 cm at 96 DPI
        ),
        margin=dict(l=10, r=10, t=56, b=40),
        font=_DEF_FONT,
        xaxis=dict(title=dict(text="Variance of return ((%)^2)", font=_DEF_FONT),
                   autorange=True, tickmode="auto",
                   tickfont=_DEF_FONT),
        yaxis=yaxis_cfg,
        height=440,
        title=dict(text=title, y=0.995, x=0.0, xanchor="left", font=_DEF_FONT),
    )
    return fig

def value_of_info_plot(sig_grid, means, title, signal_cost=None, y_range=None):
    mu_max, mu_lin, mu_sq = means

    fig = go.Figure()
    # Grouped bars: E(payoff), Top‑5 E(payoff), Highest expected payoff
    fig.add_trace(go.Bar(x=sig_grid, y=mu_lin, name="𝔼[payoff] weighted", marker_color=GREY, opacity=ALPHA))
    fig.add_trace(go.Bar(x=sig_grid, y=mu_sq,  name="Top‑5 𝔼[payoff] weighted", marker_color=RED,  opacity=ALPHA))
    fig.add_trace(go.Bar(x=sig_grid, y=mu_max, name="Highest 𝔼[payoff]", marker_color=BLUE, opacity=ALPHA))

    subtitle = f" — Signal cost £{signal_cost:g}" if signal_cost is not None else ""

    # Build y-axis config with optional fixed range
    yaxis_cfg = dict(
        title=dict(text="Mean return (%)", font=_DEF_FONT),
        tickmode="auto",
        tickfont=_DEF_FONT,
    )
    if y_range is None:
        yaxis_cfg.update(autorange=True)
    else:
        yaxis_cfg.update(autorange=False, range=list(map(float, y_range)))

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=_DEF_FONT),
        margin=dict(l=10, r=10, t=68, b=56),
        font=_DEF_FONT,
        xaxis=dict(title=dict(text="Number of signals", font=_DEF_FONT),
                   autorange=True, tickmode="linear", dtick=1,
                   tickfont=_DEF_FONT),
        yaxis=yaxis_cfg,
        height=420,
        title=dict(text=title + subtitle, x=0.0, xanchor="left", y=0.98, yanchor="top", font=_DEF_FONT),
    )
    return fig

# marginal plots removed
    
def posterior_line(x, y, title, xlab):
    xs = np.asarray(x); ys = np.asarray(y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=BLUE)))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=56, b=40),
        font=_DEF_FONT,
        xaxis=dict(title=dict(text=xlab, font=_DEF_FONT), tickfont=_DEF_FONT),
        yaxis=dict(title=dict(text="P(Ace | signal)", font=_DEF_FONT), range=[0,1], tickfont=_DEF_FONT),
        height=400,
        title=dict(text=title, y=0.995, x=0.0, xanchor="left", font=_DEF_FONT),
    )
    return fig

def stacked_histogram(bin_centers, counts_triplet, title, normalize=True):
    max_c, lin_c, top5_c = counts_triplet
    max_c = np.asarray(max_c, float)
    lin_c = np.asarray(lin_c, float)
    top5_c = np.asarray(top5_c, float)
    tot = max_c + lin_c + top5_c
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            max_y = np.where(tot > 0, max_c / tot, 0.0)
            lin_y = np.where(tot > 0, lin_c / tot, 0.0)
            top5_y = np.where(tot > 0, top5_c / tot, 0.0)
        yaxis_cfg = dict(title=dict(text="Share by strategy", font=_DEF_FONT), range=[0,1], tickfont=_DEF_FONT)
        barnorm = None
    else:
        max_y, lin_y, top5_y = max_c, lin_c, top5_c
        yaxis_cfg = dict(title=dict(text="Counts", font=_DEF_FONT), tickfont=_DEF_FONT)
        barnorm = None

    fig = go.Figure()
    fig.add_bar(x=bin_centers, y=lin_y, name="𝔼[payoff] weighted", marker_color=GREY)
    fig.add_bar(x=bin_centers, y=top5_y, name="Top‑5 𝔼[payoff] weighted", marker_color=RED)
    fig.add_bar(x=bin_centers, y=max_y, name="Highest 𝔼[payoff]", marker_color=BLUE)
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        margin=dict(l=10, r=10, t=56, b=40),
        font=_DEF_FONT,
        xaxis=dict(title=dict(text="Net return bin (%)", font=_DEF_FONT), tickfont=_DEF_FONT),
        yaxis=yaxis_cfg,
        height=420,
        title=dict(text=title, y=0.995, x=0.0, xanchor="left", font=_DEF_FONT),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=_DEF_FONT),
    )
    return fig

# ==============================
# UI: two panels side-by-side
# ==============================
st.title("Simulation Visualiser")

# Only net returns are shoawn in percentages
pct_key = "net_return"

def _fmt_ratio(x: float) -> str:
    if abs(x - 1/3) < 1e-9: return "1/3"
    if abs(x - 1/4) < 1e-9: return "1/4"
    if abs(x - 1/5) < 1e-9: return "1/5"
    if abs(x - 1/6) < 1e-9: return "1/6"
    return f"{x:.4f}"

def _panel_controls(tag: str):
    st.markdown(f"### Panel {tag}")

    # Row 1: Ace payoff, Signal type, Signal cost
    c11, c12, c13 = st.columns([1, 1, 1])
    with c11:
        ap = st.select_slider(
            "Ace payoff",
            options=ACE_PAYOUTS,
            value=20,
            key=f"ap_{tag}",
            format_func=lambda v: f"{v}X"
        )
    with c12:
        sig_label = st.selectbox("Signal type", SIG_LABELS, index=0, key=f"sig_{tag}")
    with c13:
        sc = st.select_slider(
            "Signal cost £",
            options=SIGNAL_COSTS,
            value=0,
            key=f"sc_{tag}",
            format_func=lambda v: f"£{v}" if v > 0 else "Free"
        )

    # Row 2: Payoff scaling (toggle) + scaling param
    c21, c22 = st.columns([1, 1])
    with c21:
        sp_toggle = st.toggle("Payoff scaling (relative to ace)", value=False, key=f"sp_{tag}")
        sp = 1 if sp_toggle else 0
    with c22:
        s = st.select_slider(
            "Payoff scaling parameter",
            options=SCALE_PARAMS,
            value=1/4,
            key=f"s_{tag}",
            format_func=_fmt_ratio,
            disabled=(sp == 0),
        )

    regime_key = SIG_KEY[sig_label]
    cfg = dict(signal_cost=sc, scale_pay=sp, scale_param=s, ace_payout=ap)
    return cfg, regime_key

# Collect panel configurations (must come before path resolution)
left, right = st.columns(2)
with left:
    cfgA, regimeA = _panel_controls("A")
with right:
    cfgB, regimeB = _panel_controls("B")

# Build and check paths
def _resolve_npz(cfg):
    p, norm, kid = _file_for_params(cfg)
    return p, norm, kid

try:
    pathA, normA, _ = _resolve_npz(cfgA)
    pathB, normB, _ = _resolve_npz(cfgB)
except Exception as e:
    st.error(str(e))
    st.stop()

if not pathA.exists():
    st.error(f"Missing NPZ: {pathA}")
if not pathB.exists():
    st.error(f"Missing NPZ: {pathB}")
if not (pathA.exists() and pathB.exists()):
    st.stop()

# ==============================
# Load summaries
# ==============================
def _summary_keys_for(regime, pct_key):
    return (
        "sig_grid", "budget",
        f"mean_{pct_key}_{regime}_max", f"mean_{pct_key}_{regime}_linear", f"mean_{pct_key}_{regime}_top5",
        f"sd_{pct_key}_{regime}_max",   f"sd_{pct_key}_{regime}_linear",   f"sd_{pct_key}_{regime}_top5",
        "post_median_x", "post_median_y",
        "post_top2_x",   "post_top2_y",
        "params_norm", "params_raw"
    )

dataA = load_keys(str(pathA), _summary_keys_for(regimeA, pct_key))
dataB = load_keys(str(pathB), _summary_keys_for(regimeB, pct_key))

def _extract_summary(data, regime, pct_key):
    sig_grid = np.asarray(data["sig_grid"], int)
    budget   = float(data["budget"]) if data["budget"] is not None else 100.0
    mean_triplet = (
        np.asarray(data[f"mean_{pct_key}_{regime}_max"], float),
        np.asarray(data[f"mean_{pct_key}_{regime}_linear"], float),
        np.asarray(data[f"mean_{pct_key}_{regime}_top5"], float),
    )
    sd_triplet = (
        np.asarray(data[f"sd_{pct_key}_{regime}_max"], float),
        np.asarray(data[f"sd_{pct_key}_{regime}_linear"], float),
        np.asarray(data[f"sd_{pct_key}_{regime}_top5"], float),
    )
    try:
        params_norm = json.loads(str(data["params_norm"].astype(str)))
    except Exception:
        params_norm = {}
    return sig_grid, mean_triplet, sd_triplet, params_norm, budget

sigA, meanA, sdA, pnA, budA = _extract_summary(dataA, regimeA, pct_key)
sigB, meanB, sdB, pnB, budB = _extract_summary(dataB, regimeB, pct_key)

# Net returns are already in percentages from NPZ
meanA_u = meanA; sdA_u = sdA
meanB_u = meanB; sdB_u = sdB

tabs = st.tabs(["Mean-Variances", "Value of Info", "Posteriors"])

# ========== Frontier ==========
with tabs[0]:
    row = st.columns([1, 0.25])
    with row[0]:
        st.markdown(
            "<div style='font-size:0.9rem; font-style:italic;'>"
            "All returns are net returns in percentages — "
            "100*(Cash at end of game - Total budget)/Total budget"
            "</div>",
            unsafe_allow_html=True,
        )
    with row[1]:
        fix_y_frontier = st.toggle("Fix Y axis range across panels", value=True, key="fix_y_frontier")
    # Compute shared y-range if requested
    if fix_y_frontier:
        yvals_frontier = [*meanA_u, *meanB_u]
        y_min_f = float(min(np.min(a) for a in yvals_frontier))
        y_max_f = float(max(np.max(a) for a in yvals_frontier))
        y_range_frontier = _padded_range(y_min_f, y_max_f)
    else:
        y_range_frontier = None
    c = st.columns(2)
    with c[0]:
        st.subheader("A")
        st.plotly_chart(
            frontier_plot(sigA, sdA_u, meanA_u, f"Frontier — {_label_for_regime_key(regimeA)}", y_range_frontier),
            width="stretch",
            key="frontier_A",
        )
    with c[1]:
        st.subheader("B")
        st.plotly_chart(
            frontier_plot(sigB, sdB_u, meanB_u, f"Frontier — {_label_for_regime_key(regimeB)}", y_range_frontier),
            width="stretch",
            key="frontier_B",
        )

# ========== Value of Info ==========
with tabs[1]:
    row = st.columns([1, 0.25])
    with row[0]:
        st.markdown(
            "<div style='font-size:0.9rem; font-style:italic;'>"
            "All returns are net returns in percentages — "
            "100*(Cash at end of game - Total budget)/Total budget"
            "</div>",
            unsafe_allow_html=True,
        )
    with row[1]:
        fix_y_voinfo = st.toggle("Fix Y axis range across panels", value=True, key="fix_y_voinfo")
    # Compute shared y-range if requested
    if fix_y_voinfo:
        yvals_vo = [*meanA_u, *meanB_u]
        y_min_vo = float(min(np.min(a) for a in yvals_vo))
        y_max_vo = float(max(np.max(a) for a in yvals_vo))
        y_range_vo = _padded_range(y_min_vo, y_max_vo)
    else:
        y_range_vo = None
    c = st.columns(2)
    with c[0]:
        st.subheader("A")
        scA = pnA.get("signal_cost", None)
        st.plotly_chart(
            value_of_info_plot(sigA, meanA_u, "Value of Information", scA, y_range_vo),
            width="stretch",
            key="voi_A",
        )
    with c[1]:
        st.subheader("B")
        scB = pnB.get("signal_cost", None)
        st.plotly_chart(
            value_of_info_plot(sigB, meanB_u, "Value of Information", scB, y_range_vo),
            width="stretch",
            key="voi_B",
        )

# marginal tab removed

# ========== Posteriors ==========
with tabs[2]:
    # Use default posterior NPZ without extra UI
    post = load_post_npz(POST_NPZ_DEFAULT)
    c = st.columns(2)
    with c[0]:
        st.subheader("A")
        if post is None:
            st.info("Load a valid posterior NPZ to see curves.")
        else:
            if regimeA == "median":
                st.plotly_chart(posterior_line(post["med_x"], post["med_y"], "Posterior P(Ace | Median = x)", "Median"), width="stretch", key="post_A")
            elif regimeA == "top2":
                st.plotly_chart(posterior_line(post["t2_x"], post["t2_y"], "Posterior P(Ace | Top-2 sum = x)", "Top-2 sum"), width="stretch", key="post_A")
            elif regimeA == "max":
                st.plotly_chart(posterior_line(post["mx_x"], post["mx_y"], "P(Ace | Max rank = k)", "Max rank"), width="stretch", key="post_A")
            else:
                st.plotly_chart(posterior_line(post["mn_x"], post["mn_y"], "P(Ace | Min rank = k)", "Min rank"), width="stretch", key="post_A")
    with c[1]:
        st.subheader("B")
        if post is None:
            st.info("Load a valid posterior NPZ to see curves.")
        else:
            if regimeB == "median":
                st.plotly_chart(posterior_line(post["med_x"], post["med_y"], "Posterior P(Ace | Median = x)", "Median"), width="stretch", key="post_B")
            elif regimeB == "top2":
                st.plotly_chart(posterior_line(post["t2_x"], post["t2_y"], "Posterior P(Ace | Top-2 sum = x)", "Top-2 sum"), width="stretch", key="post_B")
            elif regimeB == "max":
                st.plotly_chart(posterior_line(post["mx_x"], post["mx_y"], "P(Ace | Max rank = k)", "Max rank"), width="stretch", key="post_B")
            else:
                st.plotly_chart(posterior_line(post["mn_x"], post["mn_y"], "P(Ace | Min rank = k)", "Min rank"), width="stretch", key="post_B")

## ========== Distributions (Disabled) ==========
# with tabs[2]:
#     st.markdown(
#         "<div style='font-size:0.9rem; font-style:italic;'>"
#         "Stacked histogram of net returns — composition by strategy"
#         "</div>",
#         unsafe_allow_html=True,
#     )
#     # Choose which panel's config to use
#     col = st.columns([1,1,1,1])
#     with col[0]:
#         panel_pick = st.radio("Panel", ["A", "B"], horizontal=True, key="dist_panel")
#     # Derive context for chosen panel
#     if panel_pick == "A":
#         path, regime, sig_grid_local = pathA, regimeA, sigA
#     else:
#         path, regime, sig_grid_local = pathB, regimeB, sigB
#     # Controls
#     with col[1]:
#         n_sig = st.select_slider("#signals", options=list(map(int, sig_grid_local.tolist())), value=int(sig_grid_local.max()), key="dist_nsig")
#     with col[2]:
#         normalize = st.toggle("Normalize to share", value=True, key="dist_norm")
#     with col[3]:
#         trim_zeros = st.toggle("Hide empty bins", value=True, key="dist_trim")
#
#     # Load hist arrays just-in-time for selected regime
#     keys = (
#         f"hist_counts_{regime}_max",
#         f"hist_counts_{regime}_linear",
#         f"hist_counts_{regime}_top5",
#         "hist_start", "hist_step", "hist_n",
#     )
#     try:
#         hk = load_keys(str(path), keys)
#     except Exception as e:
#         st.error(str(e))
#         hk = None
#     if hk is None or hk.get(f"hist_counts_{regime}_max") is None:
#         st.info("This NPZ does not include distribution histograms. Re-run simulation that saves hists.")
#     else:
#         h_max = np.asarray(hk[f"hist_counts_{regime}_max"], int)[int(n_sig)]
#         h_lin = np.asarray(hk[f"hist_counts_{regime}_linear"], int)[int(n_sig)]
#         h_top = np.asarray(hk[f"hist_counts_{regime}_top5"], int)[int(n_sig)]
#         hist_start = float(hk["hist_start"]) if hk.get("hist_start") is not None else -100.0
#         hist_step  = float(hk["hist_step"])  if hk.get("hist_step")  is not None else 1.0
#         hist_n     = int(hk["hist_n"])      if hk.get("hist_n")      is not None else len(h_max)
#         xs = hist_start + hist_step * (np.arange(hist_n) + 0.5)
#         if trim_zeros:
#             mask = (h_max + h_lin + h_top) > 0
#             xs = xs[mask]; h_max = h_max[mask]; h_lin = h_lin[mask]; h_top = h_top[mask]
#         title = f"Stacked histogram — {('Panel ' + panel_pick)} — {_label_for_regime_key(regime)} — n={int(n_sig)}"
#         st.plotly_chart(stacked_histogram(xs, (h_max, h_lin, h_top), title, normalize=normalize), use_container_width=True, key=f"hist_{panel_pick}")

# ========== One-Off (single board) ==========
# # with tabs[4]:
#     st.markdown(
#         "<div style='font-size:0.9rem; font-style:italic;'>Single fixed board: compare allocation vs concentration.</div>",
#         unsafe_allow_html=True,
#     )

#     # Controls row
#     c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
#     with c1:
#         seed_one = st.number_input("Seed", min_value=0, value=12345, step=1, key="one_seed")
#     with c2:
#         sig_label_one = st.selectbox("Signal type", SIG_LABELS, index=0, key="one_sig")
#         regime_one = SIG_KEY[sig_label_one]
#     with c3:
#         sp_toggle = st.toggle("Payoff scaling", value=False, key="one_sp")
#         sp = 1 if sp_toggle else 0
#         sparam = st.select_slider("Scaling param", options=SCALE_PARAMS, value=1/4, key="one_s", format_func=_fmt_ratio, disabled=(sp==0))
#     with c4:
#         apay = st.select_slider("Ace payoff (X)", options=ACE_PAYOUTS, value=20, key="one_ap")
#     with c5:
#         scost = st.select_slider("Signal cost £", options=SIGNAL_COSTS, value=0, key="one_sc")

#     post_npz_path = st.text_input("Posterior NPZ (from precompute)", POST_NPZ_DEFAULT, key="one_post_npz")
#     loaded = load_full_post(post_npz_path)
#     if loaded is None:
#         st.stop()
#     rmax_median, rmax_top2, rmax_max, rmax_min, prior_rmax, _ = loaded
#     table_map = {"median": rmax_median, "top2": rmax_top2, "max": rmax_max, "min": rmax_min}

#     # Deal one deterministic board (r=0) for this seed
#     has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(default_rng(round_seed(int(seed_one), 0)))
#     obs_map   = {"median": np.asarray(medians, int), "top2": np.asarray(top2sum, int), "max": np.asarray(max_rank, int), "min": np.asarray(min_rank, int)}
#     post_table = table_map[regime_one]
#     obs_vals   = obs_map[regime_one]

#     # Choose n_sigs for per-pile visuals
#     n_sig = st.slider("#signals observed (first n piles)", min_value=0, max_value=int(NUM_PILES), value=3, step=1, key="one_n")

#     # Compute scores and weights for the chosen n_sig
#     ranks_all = np.arange(2, 15, dtype=int)
#     if sp == 0:
#         h_vals = np.array([float(apay) if r == ACE_RANK else 0.0 for r in ranks_all], float)
#     else:
#         h_vals = np.array([float(apay) * (float(sparam) ** max(0, ACE_RANK - r)) for r in ranks_all], float)

#     scores = np.zeros(NUM_PILES, float)
#     for i in range(NUM_PILES):
#         if i < n_sig:
#             vec = np.asarray(post_table.get(int(obs_vals[i]), prior_rmax), float)
#         else:
#             vec = np.asarray(prior_rmax, float)
#         scores[i] = float(np.dot(h_vals, vec))

#     def w_max(sc):
#         p = np.array(sc, float); mx = np.max(p)
#         winners = np.where(p == mx)[0]
#         w = np.zeros_like(p, float); w[winners] = 1.0/len(winners)
#         return w
#     def w_lin(sc):
#         p = np.array(sc, float); s = float(np.sum(p))
#         return (p/s) if s>0 else (np.ones_like(p)/len(p))
#     def w_top5(sc):
#         p = np.array(sc, float); idx = np.argsort(p)[-5:]
#         w = np.zeros_like(p, float); sm = float(np.sum(p[idx]))
#         w[idx] = (p[idx]/sm) if sm>0 else (1.0/5.0)
#         return w
#     wA, wB, wC = w_max(scores), w_lin(scores), w_top5(scores)

#     # Allocation vs Expected Return scatter
#     fig_sc = go.Figure()
#     size_k = 80.0
#     fig_sc.add_trace(go.Scatter(x=scores, y=wB, mode="markers", name="E[payoff] weighted", marker=dict(size=(wB*size_k+6), color=GREY, opacity=0.9)))
#     fig_sc.add_trace(go.Scatter(x=scores, y=wC, mode="markers", name="Top‑5 E[payoff]", marker=dict(size=(wC*size_k+6), color=RED, opacity=0.7)))
#     fig_sc.add_trace(go.Scatter(x=scores, y=wA, mode="markers", name="Highest E[payoff]", marker=dict(size=(wA*size_k+6), color=BLUE, opacity=0.7)))
#     fig_sc.update_layout(
#         template="plotly_white", height=420,
#         xaxis=dict(title=dict(text="E[payoff] per pile", font=_DEF_FONT), tickfont=_DEF_FONT),
#         yaxis=dict(title=dict(text="Weight", font=_DEF_FONT), range=[0,1], tickfont=_DEF_FONT),
#         legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=_DEF_FONT),
#         margin=dict(l=10, r=10, t=56, b=40),
#         title=dict(text=f"Allocation vs Expected Return — n={n_sig}", x=0.0, xanchor="left", y=0.98, font=_DEF_FONT),
#     )

#     # Ownership–Diversification curve
#     order = np.argsort(scores)[::-1]
#     cum_x = np.arange(1, NUM_PILES+1)/float(NUM_PILES)
#     def cum_capture(w):
#         contrib = scores[order] * w[order]
#         cs = np.cumsum(contrib)
#         denom = float(np.sum(scores * w))
#         return (cs / denom) if denom > 0 else np.zeros_like(cs)
#     fig_own = go.Figure()
#     fig_own.add_trace(go.Scatter(x=cum_x, y=cum_capture(wB), mode="lines+markers", name="E[payoff] weighted", line=dict(color=GREY)))
#     fig_own.add_trace(go.Scatter(x=cum_x, y=cum_capture(wC), mode="lines+markers", name="Top‑5 E[payoff]", line=dict(color=RED)))
#     fig_own.add_trace(go.Scatter(x=cum_x, y=cum_capture(wA), mode="lines+markers", name="Highest E[payoff]", line=dict(color=BLUE)))
#     fig_own.update_layout(
#         template="plotly_white", height=420,
#         xaxis=dict(title=dict(text="Cumulative share of piles (sorted by E[payoff])", font=_DEF_FONT), tickfont=_DEF_FONT),
#         yaxis=dict(title=dict(text="Cumulative E[payoff] captured", font=_DEF_FONT), range=[0,1.02], tickfont=_DEF_FONT),
#         legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=_DEF_FONT),
#         margin=dict(l=10, r=10, t=56, b=40),
#         title=dict(text=f"Ownership–Diversification Curve — n={n_sig}", x=0.0, xanchor="left", y=0.98, font=_DEF_FONT),
#     )

#     colv = st.columns(2)
#     with colv[0]:
#         st.plotly_chart(fig_sc, use_container_width=True)
#     with colv[1]:
#         st.plotly_chart(fig_own, use_container_width=True)

#     # Sensitivity grid removed as requested

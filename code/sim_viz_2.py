# sim_viz_2.py — Interactive viewer for both one-off and 2-stage simulations
# Two-panel comparison with toggle for One-off (output/) vs 2-Stage (output_joint/)
# Tabs: Frontier, Value of Info, Posteriors

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
# Constants
# ==============================
BLUE = "#141450"   # Highest expected payoff
GREY = "#808080"   # E(payoff) weighted
RED  = "#8B0000"   # Top-5 E(payoff) weighted
ALPHA = 0.7

CANON_SCALE_PARAM_DEFAULT = 0

SIGNAL_COSTS = [0, 3, 5, 7, 9, 11]
SCALE_PAYS   = [0, 1]
SCALE_PARAMS = [1/3, 1/4, 1/5, 1/6]
ACE_PAYOUTS  = [10, 20, 30, 40]
STAGE1_ALLOC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

SIG_LABELS = ["Median", "Top 2 rank sum"]
SIG_KEY = {"Median": "median", "Top 2 rank sum": "top2", "Max Rank": "max", "Min Rank": "min"}

@st.cache_data(show_spinner=False)
def load_post_npz(npz_path: str, is_joint: bool = False):
    """Strict loader for empirical posterior NPZ. No fallbacks.

    If required arrays are missing, emits a clear error with the missing list
    and returns None so the caller can display a simple info panel.

    Args:
        npz_path: Path to NPZ file
        is_joint: True for 2-Stage (post_joint.npz), False for One-off (post_mc.npz)
    """
    p = Path(npz_path).resolve()
    if not p.exists():
        st.error(f"Posterior NPZ not found: {p}")
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            zset = set(z.files)

            if is_joint:
                # 2-Stage: joint posteriors P(Rmax | bucket, R2) and marginal P(Rmax | R2)
                req = {
                    "joint_median_keys", "joint_median_mat",
                    "joint_top2_keys", "joint_top2_mat",
                    "r2_marginal_mat",
                    "prior_rmax",
                }
                missing = sorted(list(req - zset))
                if missing:
                    st.error("Posterior NPZ is missing required arrays: " + ", ".join(missing))
                    st.caption("Re-run precomp_joint.py to regenerate post_joint.npz.")
                    return None
                return {
                    "is_joint": True,
                    "med_keys": np.asarray(z["joint_median_keys"], int),
                    "med_mat": np.asarray(z["joint_median_mat"], float),  # [K, 13(R2), 13(Rmax)]
                    "t2_keys": np.asarray(z["joint_top2_keys"], int),
                    "t2_mat": np.asarray(z["joint_top2_mat"], float),  # [T, 13, 13]
                    "r2_marginal_mat": np.asarray(z["r2_marginal_mat"], float),  # [13(R2), 13(Rmax)]
                    "prior_rmax": np.asarray(z["prior_rmax"], float),
                }
            else:
                # One-off: marginal posteriors P(Rmax | bucket)
                req = {
                    "rmax_median_keys", "rmax_median_mat",
                    "rmax_top2_keys", "rmax_top2_mat",
                    "rmax_max_keys", "rmax_max_mat",
                    "rmax_min_keys", "rmax_min_mat",
                    "ace_median_probs", "ace_top2_probs", "ace_max_probs", "ace_min_probs",
                    "prior_rmax",
                }
                missing = sorted(list(req - zset))
                if missing:
                    st.error("Posterior NPZ is missing required arrays: " + ", ".join(missing))
                    st.caption("Re-run precompute_posteriors.py to regenerate post_mc.npz with the updated schema.")
                    return None
                return {
                    "is_joint": False,
                    "med_x": np.asarray(z["rmax_median_keys"], float),
                    "med_y": np.asarray(z["ace_median_probs"], float),
                    "med_mat": np.asarray(z["rmax_median_mat"], float),  # [K, 13(Rmax)]
                    "t2_x": np.asarray(z["rmax_top2_keys"], float),
                    "t2_y": np.asarray(z["ace_top2_probs"], float),
                    "t2_mat": np.asarray(z["rmax_top2_mat"], float),  # [T, 13]
                    "mx_x": np.asarray(z["rmax_max_keys"], float),
                    "mx_y": np.asarray(z["ace_max_probs"], float),
                    "mx_mat": np.asarray(z["rmax_max_mat"], float),
                    "mn_x": np.asarray(z["rmax_min_keys"], float),
                    "mn_y": np.asarray(z["ace_min_probs"], float),
                    "mn_mat": np.asarray(z["rmax_min_mat"], float),
                }
    except Exception as e:
        st.error(f"Failed to read posterior NPZ: {e}")
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

def _file_for_params(raw, output_dir, mode, stage1_alloc=None):
    """Build file path based on mode (one-off vs 2-stage)."""
    norm, key_id = _canonicalize(raw)

    if mode == "2-Stage":
        # Add stage1_alloc suffix for dynamic model
        if stage1_alloc is None:
            raise ValueError("stage1_alloc required for 2-Stage mode")
        a_tag = f"a{int(round(float(stage1_alloc)*10)):02d}"
        filename = f"{key_id}_{a_tag}.npz"
    else:
        # One-off model: no suffix
        filename = f"{key_id}.npz"

    p = (output_dir / filename).resolve()
    if not str(p).startswith(str(output_dir)):
        raise ValueError(f"NPZ path must be inside {output_dir}")
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
def load_keys(npz_path: str, keys: tuple[str, ...], output_dir: str):
    p = Path(npz_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    output_dir_path = Path(output_dir).resolve()
    if not str(p).startswith(str(output_dir_path)):
        raise FileNotFoundError(f"Only files inside {output_dir} are allowed.")
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

    # Legend order: E(payoff), Top-5 E(payoff), Highest expected payoff
    fig.add_trace(go.Scatter(
        x=sd_lin, y=mu_lin, mode="markers+text",
        marker=dict(size=msize, color=GREY, line=dict(width=0)),
        text=[str(int(v)) for v in sig_grid], textposition="middle center",
        textfont=dict(color="black", size=11), name="𝔼[payoff] weighted", opacity=ALPHA,
        hovertemplate="<b>𝔼[payoff] weighted</b><br>Signals: %{text}<br>SD: %{x:.2f}%<br>Mean: %{y:.2f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=sd_sq, y=mu_sq, mode="markers+text",
        marker=dict(size=msize, color=RED, line=dict(width=0)),
        text=[str(int(v)) for v in sig_grid], textposition="middle center",
        textfont=dict(color="white", size=11), name="Top-5 𝔼[payoff] weighted", opacity=ALPHA,
        hovertemplate="<b>Top-5 𝔼[payoff] weighted</b><br>Signals: %{text}<br>SD: %{x:.2f}%<br>Mean: %{y:.2f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=sd_max, y=mu_max, mode="markers+text",
        marker=dict(size=msize, color=BLUE, line=dict(width=0)),
        text=[str(int(v)) for v in sig_grid], textposition="middle center",
        textfont=dict(color="white", size=11), name="Highest 𝔼[payoff]", opacity=ALPHA,
        hovertemplate="<b>Highest 𝔼[payoff]</b><br>Signals: %{text}<br>SD: %{x:.2f}%<br>Mean: %{y:.2f}%<extra></extra>"
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
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=_DEF_FONT),
        margin=dict(l=10, r=10, t=56, b=40),
        font=_DEF_FONT,
        xaxis=dict(title=dict(text="S.D. of return (%)", font=_DEF_FONT),
                   autorange=True, tickmode="auto",
                   tickfont=_DEF_FONT),
        yaxis=yaxis_cfg,
        height=440,
        title=dict(text=title, y=0.995, x=0.0, xanchor="left", font=_DEF_FONT),
        hovermode="closest",
        hoverdistance=20,
    )
    return fig

def value_of_info_plot(sig_grid, means, title, signal_cost=None, y_range=None):
    mu_max, mu_lin, mu_sq = means

    fig = go.Figure()
    # Grouped bars: E(payoff), Top-5 E(payoff), Highest expected payoff
    fig.add_trace(go.Bar(x=sig_grid, y=mu_lin, name="𝔼[payoff] weighted", marker_color=GREY, opacity=ALPHA))
    fig.add_trace(go.Bar(x=sig_grid, y=mu_sq,  name="Top-5 𝔼[payoff] weighted", marker_color=RED,  opacity=ALPHA))
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

def posterior_line(x, y, title, xlab, ylab="P(Ace | Signals)"):
    xs = np.asarray(x); ys = np.asarray(y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=BLUE)))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=30),
        font=_DEF_FONT,
        xaxis=dict(title=dict(text=xlab, font=_DEF_FONT), tickfont=_DEF_FONT),
        yaxis=dict(title=dict(text=ylab, font=_DEF_FONT), range=[0,1], tickfont=_DEF_FONT),
        height=320,
        title=dict(text=title, y=0.99, x=0.0, xanchor="left", font=_DEF_FONT),
    )
    return fig

# ==============================
# UI: MODE TOGGLE
# ==============================
st.title("Visualiser")

# Only net returns are shown in percentages
pct_key = "net_return"

def _fmt_ratio(x: float) -> str:
    if abs(x - 1/3) < 1e-9: return "1/3"
    if abs(x - 1/4) < 1e-9: return "1/4"
    if abs(x - 1/5) < 1e-9: return "1/5"
    if abs(x - 1/6) < 1e-9: return "1/6"
    return f"{x:.4f}"

@st.cache_data(show_spinner=False)
def load_frontier_npz(npz_path: str):
    """Load frontier NPZ file containing mean-variance efficient frontiers."""
    p = Path(npz_path).resolve()
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=True) as z:
            return {
                "sd_levels_by_n": z["sd_levels_by_n"],
                "best_means_by_n": z["best_means_by_n"],
                "best_weights_by_n": z["best_weights_by_n"],
                "sd_step": float(z["sd_step"]),
                "meta": json.loads(str(z["meta"])),
            }
    except Exception as e:
        st.error(f"Failed to load frontier NPZ: {e}")
        return None

def _panel_controls(tag: str):
    st.markdown(f"### Panel {tag}")

    # Row 1: Mode toggle + Stage allocation
    c01, c02 = st.columns([1, 1])
    with c01:
        mode = st.radio("Simulation Type", ["One-off", "2-Stage"], horizontal=True, key=f"mode_{tag}")
    with c02:
        if mode == "2-Stage":
            alpha = st.select_slider(
                "Budget allocation (%) - Stage 1 : 2",
                options=STAGE1_ALLOC,
                value=0.5,
                key=f"alpha_{tag}",
                format_func=lambda v: f"{round(v*100)}:{round((1-v)*100)} split"
            )
        else:
            alpha = None

    # Row 2: Ace payoff, Signal type, Signal cost
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

    # Determine output directory based on mode
    if mode == "One-off":
        output_dir = Path("output/").resolve()
    else:  # 2-Stage
        output_dir = Path("output_joint/").resolve()

    return cfg, regime_key, mode, alpha, output_dir

# ==============================
# PAGE SELECTOR
# ==============================
with st.expander("☰ Menu", expanded=False):
    page = st.radio("View", ["Simulation Results", "Mean-Variance Frontier"], horizontal=True)
st.markdown("---")

if page == "Mean-Variance Frontier":
    # ==============================
    # FRONTIER PAGE — TWO PANELS
    # ==============================
    st.header("Information-Limited Mean-Variance Frontier")
    st.caption("Efficient frontiers showing mean vs. standard deviation of net returns")

    # Small help + shared y-range toggle
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
        fix_y_mv = st.toggle("Fix Y axis range across panels", value=True, key="fix_y_mv")

    # Per-panel controls
    ctlA, ctlB = st.columns(2)
    with ctlA:
        st.markdown("### Panel A")
        frontier_sp_A = st.radio("Payoff scaling", ["Off (Ace-only)", "On (Scaled)"], horizontal=True, key="frontier_sp_A")
        sp_A = 1 if "On" in frontier_sp_A else 0
        frontier_sig_A = st.selectbox("Signal type", ["Median", "Top 2"], key="frontier_sig_A")
        sig_A = "median" if frontier_sig_A == "Median" else "top2"
        alpha_A = st.select_slider("Stage 1 allocation", options=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], value=0.5,
                                   format_func=lambda v: f"{int(v*100)}:{int((1-v)*100)}",
                                   key="frontier_alpha_A")
        max_n_A = st.slider("Max signals", min_value=0, max_value=9, value=9, key="max_n_sig_frontier_A")
    with ctlB:
        st.markdown("### Panel B")
        frontier_sp_B = st.radio("Payoff scaling", ["Off (Ace-only)", "On (Scaled)"], horizontal=True, key="frontier_sp_B")
        sp_B = 1 if "On" in frontier_sp_B else 0
        frontier_sig_B = st.selectbox("Signal type", ["Median", "Top 2"], key="frontier_sig_B")
        sig_B = "median" if frontier_sig_B == "Median" else "top2"
        alpha_B = st.select_slider("Stage 1 allocation", options=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], value=0.5,
                                   format_func=lambda v: f"{int(v*100)}:{int((1-v)*100)}",
                                   key="frontier_alpha_B")
        max_n_B = st.slider("Max signals", min_value=0, max_value=9, value=9, key="max_n_sig_frontier_B")

    # Resolve files
    frontier_dir = Path("frontier_output/").resolve()
    raw_A = dict(signal_cost=3.0, scale_pay=sp_A, scale_param=(0.25 if sp_A == 1 else 0.0), ace_payout=20.0)
    raw_B = dict(signal_cost=3.0, scale_pay=sp_B, scale_param=(0.25 if sp_B == 1 else 0.0), ace_payout=20.0)
    _, key_A = _canonicalize(raw_A)
    _, key_B = _canonicalize(raw_B)
    tag_A = f"a{int(round(float(alpha_A)*10)):02d}"
    tag_B = f"a{int(round(float(alpha_B)*10)):02d}"
    file_A = frontier_dir / f"{key_A}_{sig_A}_{tag_A}.npz"
    file_B = frontier_dir / f"{key_B}_{sig_B}_{tag_B}.npz"

    data_A = load_frontier_npz(str(file_A))
    data_B = load_frontier_npz(str(file_B))

    # Helper: gather means for y-range
    def _gather_means(fd, max_n):
        if fd is None:
            return []
        means = []
        mean_by_n = fd["best_means_by_n"]
        for n_sig in range(min(len(mean_by_n), max_n + 1)):
            mv = mean_by_n[n_sig]
            if len(mv) > 0:
                means.append(np.asarray(mv, float))
        return means

    y_range = None
    if fix_y_mv and (data_A is not None) and (data_B is not None):
        all_means = [*_gather_means(data_A, max_n_A), *_gather_means(data_B, max_n_B)]
        if len(all_means) > 0:
            y_min = float(min(np.min(a) for a in all_means))
            y_max = float(max(np.max(a) for a in all_means))
            y_range = _padded_range(y_min, y_max)

    # Helper: build figure
    def _build_fig(fd, max_n, y_range_override=None):
        if fd is None:
            return None
        fig = go.Figure()
        sd_by_n = fd["sd_levels_by_n"]
        mean_by_n = fd["best_means_by_n"]
        weights_by_n = fd["best_weights_by_n"]
        points = []
        all_sum_sq_weights = []
        for n_sig in range(min(len(sd_by_n), max_n + 1)):
            sd_vals = sd_by_n[n_sig]
            mean_vals = mean_by_n[n_sig]
            weights = weights_by_n[n_sig]
            if len(sd_vals) == 0:
                continue
            sum_sq_weights = [float(np.sum(np.asarray(w_vec, float) ** 2)) for w_vec in weights]
            points.append(dict(n=n_sig, sd=np.asarray(sd_vals, float), mean=np.asarray(mean_vals, float), ssq=np.asarray(sum_sq_weights, float)))
            all_sum_sq_weights.extend(sum_sq_weights)
        # Determine global frontier (Pareto-efficient) across all points
        combined = []  # (sd, mean, (series_idx, point_idx))
        for i, p in enumerate(points):
            sd_vals = p["sd"]; mean_vals = p["mean"]
            for j in range(len(sd_vals)):
                combined.append((float(sd_vals[j]), float(mean_vals[j]), (i, j)))
        combined.sort(key=lambda t: (t[0], -t[1]))
        frontier_pos = set()
        best_mean = -1e9
        for sd, mean, pos in combined:
            if mean > best_mean + 1e-12:
                frontier_pos.add(pos)
                best_mean = mean
        if len(all_sum_sq_weights) > 0:
            vmin_global = float(min(all_sum_sq_weights))
            vmax_global = float(max(all_sum_sq_weights))
        else:
            vmin_global = 0.0; vmax_global = 1.0
        for i, p in enumerate(points):
            n_sig = p["n"]; sd_vals = p["sd"]; mean_vals = p["mean"]; ssq = p["ssq"]
            # Keep all frontier points; keep every other interior point to reduce clutter
            keep_idx = []
            for j in range(len(sd_vals)):
                if (i, j) in frontier_pos or (j % 2 == 0):
                    keep_idx.append(j)
            if not keep_idx:
                continue
            xs = sd_vals[keep_idx]
            ys = mean_vals[keep_idx]
            cs = ssq[keep_idx]
            hover_texts = [f"n={n_sig}<br>Mean: {ys[k]:.2f}%<br>SD: {xs[k]:.2f}%<br>Σw²: {cs[k]:.3f}" for k in range(len(keep_idx))]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+text", name=f"n={n_sig}",
                marker=dict(size=16, color=cs, colorscale=[[0, "#2b8cbe"], [1, "#08306b"]],
                            cmin=vmin_global, cmax=vmax_global, showscale=False, line=dict(width=0)),
                text=[str(n_sig)] * len(keep_idx), textposition="middle center", textfont=dict(size=11, color="white"),
                hovertext=hover_texts, hoverinfo="text", showlegend=False, opacity=ALPHA,
            ))
        if len(all_sum_sq_weights) > 0:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(size=0, colorscale=[[0, "#2b8cbe"], [1, "#08306b"]],
                                                 cmin=vmin_global, cmax=vmax_global,
                                                 colorbar=dict(title=dict(text="Σw²<br>(concentration)", side="right"), len=0.5, y=0.75)),
                                     showlegend=False, hoverinfo="skip"))
        yaxis_cfg = dict(title=dict(text="Mean Net Return (%)", font=dict(size=18)), tickfont=dict(size=11), showgrid=True, gridcolor="rgba(128,128,128,0.1)")
        if y_range_override is not None:
            yaxis_cfg.update(range=list(map(float, y_range_override)))
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Roboto, Arial, sans-serif", size=15),
            xaxis=dict(title=dict(text="Standard Deviation (%)", font=dict(size=18)), tickfont=dict(size=11), showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
            yaxis=yaxis_cfg,
            height=600,
            hovermode="closest",
            margin=dict(l=60, r=100, t=40, b=60),
        )
        return fig

    # Render charts side-by-side
    colA, colB = st.columns(2)
    with colA:
        if data_A is None:
            st.info(f"Frontier data not found: {file_A.name}")
            st.caption("Run frontier.py to generate frontier data.")
        else:
            st.markdown("**Fixed parameters:** Signal cost = £3, Ace payoff = 20X" + (", Scale param = 0.25" if sp_A == 1 else ""))
            figA = _build_fig(data_A, max_n_A, y_range)
            st.plotly_chart(figA, use_container_width=True, key="mv_frontier_A")
            # Frontier details removed for cleaner layout

    with colB:
        if data_B is None:
            st.info(f"Frontier data not found: {file_B.name}")
            st.caption("Run frontier.py to generate frontier data.")
        else:
            st.markdown("**Fixed parameters:** Signal cost = £3, Ace payoff = 20X" + (", Scale param = 0.25" if sp_B == 1 else ""))
            figB = _build_fig(data_B, max_n_B, y_range)
            st.plotly_chart(figB, use_container_width=True, key="mv_frontier_B")
            # Frontier details removed for cleaner layout

    # Stop here to avoid executing Simulation Results code below
    st.stop()

else:
    # ==============================
    # SIMULATION RESULTS PAGE (EXISTING)
    # ==============================
    st.header("Simulation Results - Two-Panel Comparison")

    # Collect panel configurations (must come before path resolution)
    left, right = st.columns(2)
    with left:
        cfgA, regimeA, modeA, alphaA, outputA = _panel_controls("A")
    with right:
        cfgB, regimeB, modeB, alphaB, outputB = _panel_controls("B")

    # Build and check paths
    def _resolve_npz(cfg, output_dir, mode, alpha):
        p, norm, kid = _file_for_params(cfg, output_dir, mode, alpha)
        return p, norm, kid

    try:
        pathA, normA, _ = _resolve_npz(cfgA, outputA, modeA, alphaA)
        pathB, normB, _ = _resolve_npz(cfgB, outputB, modeB, alphaB)
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

    dataA = load_keys(str(pathA), _summary_keys_for(regimeA, pct_key), str(outputA))
    dataB = load_keys(str(pathB), _summary_keys_for(regimeB, pct_key), str(outputB))

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

tabs = st.tabs(["Frontier", "Value of Info", "Posteriors"])

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

# ========== Posteriors ==========
with tabs[2]:
    # Use posterior NPZ from appropriate directory based on panel mode
    postA_path = str((outputA / "post_joint.npz" if modeA == "2-Stage" else outputA / "post_mc.npz"))
    postB_path = str((outputB / "post_joint.npz" if modeB == "2-Stage" else outputB / "post_mc.npz"))

    postA = load_post_npz(postA_path, is_joint=(modeA == "2-Stage"))
    postB = load_post_npz(postB_path, is_joint=(modeB == "2-Stage"))

    c = st.columns(2)

    # Panel A
    with c[0]:
        st.subheader("A")
        if postA is None:
            st.info("Load a valid posterior NPZ to see curves.")
        elif postA.get("is_joint"):
            # 2-Stage mode: joint posteriors with interactive controls
            st.caption("Note: Signal type is controlled by the dropdown below (panel setting ignored)")

            # X-axis selector
            c_r2_1, c_r2_2 = st.columns([1, 2])
            c_r2_1, c_r2_2 = st.columns([1, 1])
            with c_r2_1:
                x_axis_opts = ["Median", "Top 2", "Second Rank"]
                x_axis_choice = st.selectbox("X-axis", x_axis_opts, key="x_axis_A")

            if x_axis_choice in ["Median", "Top 2"]:
                # Signal on X-axis, R2 as parameter
                signal_type = "median" if x_axis_choice == "Median" else "top2"

                with c_r2_2:
                    r2_val = st.slider("Second Rank (R2)", min_value=2, max_value=13, value=10, key="r2_A")

                # Extract data
                keys = postA["med_keys"] if signal_type == "median" else postA["t2_keys"]
                mat = postA["med_mat"] if signal_type == "median" else postA["t2_mat"]  # [K, 13(R2), 13(Rmax)]

                # Slice for R2 and Ace
                r2_idx = int(r2_val) - 2
                ace_idx = ACE_RANK - 2  # Ace = 14, idx = 12
                y_vals = mat[:, r2_idx, ace_idx]  # [K]
                x_vals = keys

                title = f"P(Ace | {x_axis_choice} = x and R2 = {r2_val})"
                xlab = x_axis_choice
                st.plotly_chart(posterior_line(x_vals, y_vals, title, xlab), width="stretch", key="post_A")

            else:  # "Second Rank" on X-axis
                # R2 on X-axis, signal as parameter - put controls on same row
                col1, col2 = st.columns([1, 2])
                with col1:
                    sig_type_choice = st.selectbox("Signal type", ["Median", "Top 2"], key="sig_type_A")

                signal_type = "median" if sig_type_choice == "Median" else "top2"
                keys = postA["med_keys"] if signal_type == "median" else postA["t2_keys"]
                mat = postA["med_mat"] if signal_type == "median" else postA["t2_mat"]

                with col2:
                    sig_val = st.slider(f"{sig_type_choice} value",
                                       min_value=int(keys.min()),
                                       max_value=int(keys.max()),
                                       value=int(keys[len(keys)//2]) if len(keys) > 0 else 8,
                                       key="sig_val_A")

                # Find bucket index
                bucket_idx = np.where(keys == sig_val)[0]
                if len(bucket_idx) == 0:
                    st.warning(f"No data for {sig_type_choice} = {sig_val}")
                else:
                    bucket_idx = bucket_idx[0]
                    ace_idx = ACE_RANK - 2
                    y_vals = mat[bucket_idx, :, ace_idx]  # [13(R2)]
                    x_vals = np.arange(2, 15)  # R2 values 2-14

                    title = f"P(Ace | {sig_type_choice} = {sig_val} and R2 = x)"
                    xlab = "Second Rank (R2)"
                    st.plotly_chart(posterior_line(x_vals, y_vals, title, xlab), width="stretch", key="post_A")
        else:
            # One-off mode: marginal posteriors (existing logic)
            scale_pay = pnA.get("scale_pay", 0)
            if scale_pay == 1:
                # Payoff scaling ON: show P(Max rank = k | signal)
                max_rank_choice = st.slider("Max Rank", min_value=2, max_value=14, value=14, key="max_rank_A")
                rmax_idx = max_rank_choice - 2

                if regimeA == "median":
                    x_vals = postA["med_x"]
                    y_vals = postA["med_mat"][:, rmax_idx]
                    sig_name = "Median"
                elif regimeA == "top2":
                    x_vals = postA["t2_x"]
                    y_vals = postA["t2_mat"][:, rmax_idx]
                    sig_name = "Top-2 sum"
                elif regimeA == "max":
                    x_vals = postA["mx_x"]
                    y_vals = postA["mx_mat"][:, rmax_idx]
                    sig_name = "Max rank"
                else:
                    x_vals = postA["mn_x"]
                    y_vals = postA["mn_mat"][:, rmax_idx]
                    sig_name = "Min rank"

                title = f"P(Max rank = {max_rank_choice} | {sig_name} = x)"
                ylab_text = f"P(Max rank = {max_rank_choice} | signal)"

                # Use modified posterior_line that accepts custom ylab
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", line=dict(color=BLUE)))
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=40, b=30),
                    font=_DEF_FONT,
                    xaxis=dict(title=dict(text=sig_name, font=_DEF_FONT), tickfont=_DEF_FONT),
                    yaxis=dict(title=dict(text=ylab_text, font=_DEF_FONT), range=[0,1], tickfont=_DEF_FONT),
                    height=320,
                    title=dict(text=title, y=0.99, x=0.0, xanchor="left", font=_DEF_FONT),
                )
                st.plotly_chart(fig, width="stretch", key="post_A")
            else:
                # Payoff scaling OFF: show P(Ace | signal)
                if regimeA == "median":
                    st.plotly_chart(posterior_line(postA["med_x"], postA["med_y"], "P(Ace | Median = x)", "Median"), width="stretch", key="post_A")
                elif regimeA == "top2":
                    st.plotly_chart(posterior_line(postA["t2_x"], postA["t2_y"], "P(Ace | Top-2 sum = x)", "Top-2 sum"), width="stretch", key="post_A")
                elif regimeA == "max":
                    st.plotly_chart(posterior_line(postA["mx_x"], postA["mx_y"], "P(Ace | Max rank = k)", "Max rank"), width="stretch", key="post_A")
                else:
                    st.plotly_chart(posterior_line(postA["mn_x"], postA["mn_y"], "P(Ace | Min rank = k)", "Min rank"), width="stretch", key="post_A")

    # Panel B
    with c[1]:
        st.subheader("B")
        if postB is None:
            st.info("Load a valid posterior NPZ to see curves.")
        elif postB.get("is_joint"):
            # 2-Stage mode: joint posteriors with interactive controls
            st.caption("Note: Signal type is controlled by the dropdown below (panel setting ignored)")

            # X-axis selector
            
            c_r2_1, c_r2_2 = st.columns([1, 2])
            c_r2_1, c_r2_2 = st.columns([1, 1])
            with c_r2_1:
                x_axis_opts = ["Median", "Top 2", "Second Rank"]
                x_axis_choice = st.selectbox("X-axis", x_axis_opts, key="x_axis_B")

            if x_axis_choice in ["Median", "Top 2"]:
                # Signal on X-axis, R2 as parameter
                signal_type = "median" if x_axis_choice == "Median" else "top2"

                with c_r2_2:
                    r2_val = st.slider("Second Rank (R2)", min_value=2, max_value=13, value=10, key="r2_B")

                # Extract data
                keys = postB["med_keys"] if signal_type == "median" else postB["t2_keys"]
                mat = postB["med_mat"] if signal_type == "median" else postB["t2_mat"]

                # Slice for R2 and Ace
                r2_idx = int(r2_val) - 2
                ace_idx = ACE_RANK - 2
                y_vals = mat[:, r2_idx, ace_idx]
                x_vals = keys

                title = f"P(Ace | {x_axis_choice} = x and R2 = {r2_val})"
                xlab = x_axis_choice
                st.plotly_chart(posterior_line(x_vals, y_vals, title, xlab), width="stretch", key="post_B")

            else:  # "Second Rank" on X-axis
                # R2 on X-axis, signal as parameter - put controls on same row
                col1, col2 = st.columns([1, 2])
                with col1:
                    sig_type_choice = st.selectbox("Signal type", ["Median", "Top 2"], key="sig_type_B")

                signal_type = "median" if sig_type_choice == "Median" else "top2"
                keys = postB["med_keys"] if signal_type == "median" else postB["t2_keys"]
                mat = postB["med_mat"] if signal_type == "median" else postB["t2_mat"]

                with col2:
                    sig_val = st.slider(f"{sig_type_choice} value",
                                       min_value=int(keys.min()),
                                       max_value=int(keys.max()),
                                       value=int(keys[len(keys)//2]) if len(keys) > 0 else 8,
                                       key="sig_val_B")

                # Find bucket index
                bucket_idx = np.where(keys == sig_val)[0]
                if len(bucket_idx) == 0:
                    st.warning(f"No data for {sig_type_choice} = {sig_val}")
                else:
                    bucket_idx = bucket_idx[0]
                    ace_idx = ACE_RANK - 2
                    y_vals = mat[bucket_idx, :, ace_idx]
                    x_vals = np.arange(2, 15)

                    title = f"P(Ace | {sig_type_choice} = {sig_val} and R2 = x)"
                    xlab = "Second Rank (R2)"
                    st.plotly_chart(posterior_line(x_vals, y_vals, title, xlab), width="stretch", key="post_B")
        else:
            # One-off mode: marginal posteriors
            scale_pay = pnB.get("scale_pay", 0)
            if scale_pay == 1:
                # Payoff scaling ON: show P(Max rank = k | signal)
                max_rank_choice = st.slider("Max Rank", min_value=2, max_value=14, value=14, key="max_rank_B")
                rmax_idx = max_rank_choice - 2

                if regimeB == "median":
                    x_vals = postB["med_x"]
                    y_vals = postB["med_mat"][:, rmax_idx]
                    sig_name = "Median"
                elif regimeB == "top2":
                    x_vals = postB["t2_x"]
                    y_vals = postB["t2_mat"][:, rmax_idx]
                    sig_name = "Top-2 sum"
                elif regimeB == "max":
                    x_vals = postB["mx_x"]
                    y_vals = postB["mx_mat"][:, rmax_idx]
                    sig_name = "Max rank"
                else:
                    x_vals = postB["mn_x"]
                    y_vals = postB["mn_mat"][:, rmax_idx]
                    sig_name = "Min rank"

                title = f"P(Max rank = {max_rank_choice} | {sig_name} = x)"
                ylab_text = f"P(Max rank = {max_rank_choice} | signal)"

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", line=dict(color=BLUE)))
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=40, b=30),
                    font=_DEF_FONT,
                    xaxis=dict(title=dict(text=sig_name, font=_DEF_FONT), tickfont=_DEF_FONT),
                    yaxis=dict(title=dict(text=ylab_text, font=_DEF_FONT), range=[0,1], tickfont=_DEF_FONT),
                    height=320,
                    title=dict(text=title, y=0.99, x=0.0, xanchor="left", font=_DEF_FONT),
                )
                st.plotly_chart(fig, width="stretch", key="post_B")
            else:
                # Payoff scaling OFF: show P(Ace | signal)
                if regimeB == "median":
                    st.plotly_chart(posterior_line(postB["med_x"], postB["med_y"], "P(Ace | Median = x)", "Median"), width="stretch", key="post_B")
                elif regimeB == "top2":
                    st.plotly_chart(posterior_line(postB["t2_x"], postB["t2_y"], "P(Ace | Top-2 sum = x)", "Top-2 sum"), width="stretch", key="post_B")
                elif regimeB == "max":
                    st.plotly_chart(posterior_line(postB["mx_x"], postB["mx_y"], "P(Ace | Max rank = k)", "Max rank"), width="stretch", key="post_B")
                else:
                    st.plotly_chart(posterior_line(postB["mn_x"], postB["mn_y"], "P(Ace | Min rank = k)", "Min rank"), width="stretch", key="post_B")

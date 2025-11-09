# vis_sim_res.py — Interactive viewer for simulation results and posteriors
# Two-panel comparison with toggle for One-off vs 2-Stage simulations
# Views: Simulation Results (Frontier, Value of Info) | Posteriors (Conditional, Joint Conditional)

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from numpy.random import default_rng

# Import core model primitives (ACE_RANK constant only)
from sim_res import ACE_RANK


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
      .stTabs [data-baseweb="tab"] { padding: .5rem 1rem; font-size: 1.1rem; font-weight: 600; }
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
def load_post_npz(npz_path: str):
    """Load unified posterior NPZ containing both conditional and joint conditional posteriors.

    Expected arrays in precomp_output/post_mc.npz:
    - Conditional posteriors: P(Rmax | signal)
      - rmax_median_keys, rmax_median_mat
      - rmax_top2_keys, rmax_top2_mat
    - Joint conditional posteriors: P(Rmax | signal, R2)
      - joint_median_keys, joint_median_mat
      - joint_top2_keys, joint_top2_mat
    - Common:
      - r2_marginal_mat: P(Rmax | R2)
      - prior_rmax: P(Rmax)
    """
    p = Path(npz_path).resolve()
    if not p.exists():
        st.error(f"Posterior NPZ not found: {p}")
        return None
    try:
        with np.load(p, allow_pickle=True) as z:
            zset = set(z.files)

            # Required arrays for unified posteriors
            req = {
                "joint_median_keys", "joint_median_mat",
                "joint_top2_keys", "joint_top2_mat",
                "rmax_median_keys", "rmax_median_mat",
                "rmax_top2_keys", "rmax_top2_mat",
                "r2_marginal_mat",
                "prior_rmax",
            }
            missing = sorted(list(req - zset))
            if missing:
                st.error("Posterior NPZ is missing required arrays: " + ", ".join(missing))
                st.caption("Re-run precomp.py to regenerate precomp_output/post_mc.npz.")
                return None

            return {
                # Joint conditional: P(Rmax | signal, R2)
                "joint_med_keys": np.asarray(z["joint_median_keys"], int),
                "joint_med_mat": np.asarray(z["joint_median_mat"], float),  # [K, 13(R2), 13(Rmax)]
                "joint_t2_keys": np.asarray(z["joint_top2_keys"], int),
                "joint_t2_mat": np.asarray(z["joint_top2_mat"], float),  # [T, 13, 13]
                # Conditional: P(Rmax | signal)
                "cond_med_keys": np.asarray(z["rmax_median_keys"], int),
                "cond_med_mat": np.asarray(z["rmax_median_mat"], float),  # [K, 13(Rmax)]
                "cond_t2_keys": np.asarray(z["rmax_top2_keys"], int),
                "cond_t2_mat": np.asarray(z["rmax_top2_mat"], float),  # [T, 13]
                # Common
                "r2_marginal_mat": np.asarray(z["r2_marginal_mat"], float),  # [13(R2), 13(Rmax)]
                "prior_rmax": np.asarray(z["prior_rmax"], float),
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
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0.5,
            xanchor="center",
            font=_DEF_FONT,
            itemsizing="constant",
            itemwidth=114  # ~3 cm at 96 DPI
        ),
        margin=dict(l=10, r=10, t=56, b=40),
        font=_DEF_FONT,
        xaxis=dict(title=dict(text="S.D. of return (%)", font=_DEF_FONT),
                   autorange=True, tickmode="auto",
                   tickfont=_DEF_FONT),
        yaxis=yaxis_cfg,
        height=520,
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
        height=520,
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
        height=420,
        title=dict(text=title, y=0.99, x=0.0, xanchor="left", font=_DEF_FONT),
    )
    return fig

def _render_posteriors_panel(tag: str, post_data: dict):
    """Render a single posteriors panel with independent controls.

    Args:
        tag: Panel identifier ("A" or "B")
        post_data: Loaded posterior data from load_post_npz()
    """
    st.subheader(f"Panel {tag}")

    if post_data is None:
        st.info("Load a valid posterior NPZ to see curves.")
        return

    # Row 1: Posterior type selector (always shown)
    post_type = st.radio("Posterior type", ["Conditional", "Joint Conditional"],
                        horizontal=True, key=f"post_type_{tag}")

    # Row 2: Payoff scaling toggle (always shown)
    scale_pay = 1 if st.toggle("Payoff scaling", value=False, key=f"scale_pay_post_{tag}") else 0

    # Row 3: Signal type selector (conditional on post_type)
    if post_type == "Conditional":
        signal_options = ["Median", "Top 2", "R2"]
    else:  # Joint Conditional
        signal_options = ["Median", "Top 2"]
    signal_label = st.selectbox("Signal type", signal_options, key=f"post_sig_{tag}")

    # Map signal label to internal type
    if signal_label == "Median":
        signal_type = "median"
    elif signal_label == "Top 2":
        signal_type = "top2"
    else:  # R2
        signal_type = "r2"

    # ========== CONDITIONAL POSTERIORS ==========
    if post_type == "Conditional":

        # P(Rmax | signal) - like one-off mode
        if scale_pay == 1:
            # Payoff scaling ON: show P(Max rank = k | signal)
            max_rank_choice = st.slider("Max Rank", min_value=2, max_value=14, value=14,
                                       key=f"max_rank_cond_{tag}")
            rmax_idx = max_rank_choice - 2

            if signal_type == "median":
                x_vals = post_data["cond_med_keys"]
                y_vals = post_data["cond_med_mat"][:, rmax_idx]
                sig_name = "Median"
            elif signal_type == "top2":
                x_vals = post_data["cond_t2_keys"]
                y_vals = post_data["cond_t2_mat"][:, rmax_idx]
                sig_name = "Top 2"
            else:  # r2
                x_vals = np.arange(2, 14)  # R2 values from 2 to 13 (cannot be 14/Ace)
                y_vals = post_data["r2_marginal_mat"][:, rmax_idx]
                sig_name = "R2"

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
                height=420,
                title=dict(text=title, y=0.99, x=0.0, xanchor="left", font=_DEF_FONT),
            )
            st.plotly_chart(fig, width="stretch", key=f"post_cond_{tag}")
        else:
            # Payoff scaling OFF: show P(Ace | signal)
            ace_idx = ACE_RANK - 2  # Ace = 14, idx = 12

            if signal_type == "median":
                x_vals = post_data["cond_med_keys"]
                y_vals = post_data["cond_med_mat"][:, ace_idx]
                sig_name = "Median"
            elif signal_type == "top2":
                x_vals = post_data["cond_t2_keys"]
                y_vals = post_data["cond_t2_mat"][:, ace_idx]
                sig_name = "Top 2"
            else:  # r2
                x_vals = np.arange(2, 14)  # R2 values from 2 to 13 (cannot be 14/Ace)
                y_vals = post_data["r2_marginal_mat"][:, ace_idx]
                sig_name = "R2"

            title = f"P(Ace | {sig_name} = x)"
            st.plotly_chart(posterior_line(x_vals, y_vals, title, sig_name),
                          width="stretch", key=f"post_cond_{tag}")

    # ========== JOINT CONDITIONAL POSTERIORS ==========
    else:  # post_type == "Joint Conditional"
        # P(Rmax | signal, R2) - like 2-Stage mode
        # X-axis selector
        col_x1, col_x2 = st.columns([1, 2])
        with col_x1:
            x_axis_opts = [signal_label, "Second Rank"]
            x_axis_choice = st.selectbox("X-axis", x_axis_opts, key=f"x_axis_joint_{tag}")

        if x_axis_choice == signal_label:
            # Signal on X-axis, R2 as parameter
            with col_x2:
                r2_val = st.slider("Second Rank (R2)", min_value=2, max_value=13, value=10,
                                  key=f"r2_joint_{tag}")

            # Extract data
            keys = post_data["joint_med_keys"] if signal_type == "median" else post_data["joint_t2_keys"]
            mat = post_data["joint_med_mat"] if signal_type == "median" else post_data["joint_t2_mat"]

            # Slice for R2 and Ace
            r2_idx = int(r2_val) - 2
            ace_idx = ACE_RANK - 2
            y_vals = mat[:, r2_idx, ace_idx]
            x_vals = keys

            title = f"P(Ace | {signal_label} = x and R2 = {r2_val})"
            xlab = signal_label
            st.plotly_chart(posterior_line(x_vals, y_vals, title, xlab),
                          width="stretch", key=f"post_joint_{tag}")

        else:  # "Second Rank" on X-axis
            # R2 on X-axis, signal as parameter
            keys = post_data["joint_med_keys"] if signal_type == "median" else post_data["joint_t2_keys"]
            mat = post_data["joint_med_mat"] if signal_type == "median" else post_data["joint_t2_mat"]

            with col_x2:
                sig_val = st.slider(f"{signal_label} value",
                                   min_value=int(keys.min()),
                                   max_value=int(keys.max()),
                                   value=int(keys[len(keys)//2]) if len(keys) > 0 else 8,
                                   key=f"sig_val_joint_{tag}")

            # Find bucket index
            bucket_idx = np.where(keys == sig_val)[0]
            if len(bucket_idx) == 0:
                st.warning(f"No data for {signal_label} = {sig_val}")
            else:
                bucket_idx = bucket_idx[0]
                ace_idx = ACE_RANK - 2
                y_vals = mat[bucket_idx, :, ace_idx]
                x_vals = np.arange(2, 15)  # R2 values 2-14

                title = f"P(Ace | {signal_label} = {sig_val} and R2 = x)"
                xlab = "Second Rank (R2)"
                st.plotly_chart(posterior_line(x_vals, y_vals, title, xlab),
                              width="stretch", key=f"post_joint_r2_{tag}")

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
                value=1.0,
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
# TOP-LEVEL VIEW SELECTOR
# ==============================

# View selector: Simulation Results vs Posteriors
view = st.radio("View", ["Simulation Results", "Posteriors"], horizontal=True, key="top_view")

# ==============================
# CONDITIONAL RENDERING: SIMULATION RESULTS vs POSTERIORS
# ==============================

if view == "Simulation Results":
    # ==============================
    # PANEL CONTROLS (SIMULATION RESULTS ONLY)
    # ==============================

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

    # ==============================
    # SIMULATION RESULTS TABS
    # ==============================
    # Simulation Results view: Frontier + Value of Info tabs
    tabs = st.tabs(["Mean-Variances", "Value of Info"])

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

else:  # view == "Posteriors"
    # ==============================
    # POSTERIORS VIEW (NEW)
    # ==============================

    # Load unified posterior NPZ (precomp_output/post_mc.npz)
    post_npz_path = Path("precomp_output/post_mc.npz").resolve()
    post_data = load_post_npz(str(post_npz_path))

    c = st.columns(2)

    # Panel A
    with c[0]:
        _render_posteriors_panel("A", post_data)

    # Panel B
    with c[1]:
        _render_posteriors_panel("B", post_data)

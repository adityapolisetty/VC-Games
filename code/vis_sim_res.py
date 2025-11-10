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
st.set_page_config(page_title="VC Card Game", layout="wide")
st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp, .block-container { background: #ffffff !important; }
      .block-container { padding-top: 1.25rem; padding-bottom: 1.0rem; }
      .stSelectbox label, .stNumberInput label { font-size: 0.92rem; margin-bottom: .18rem; }
      .stSelectbox > div[data-baseweb="select"] { min-height: 36px; }
      .stSelectbox > div[data-baseweb="select"] > div { border: 1px solid #000000 !important; border-radius: 4px; }
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

            # Check for count arrays (optional, for confidence intervals)
            has_counts = all(k in zset for k in [
                "joint_median_counts", "joint_top2_counts",
                "rmax_median_counts", "rmax_top2_counts", "r2_marginal_counts"
            ])

            if not has_counts:
                st.warning("⚠️ NPZ file does not contain count data for confidence intervals. Re-run precomp.py to enable confidence bands.")

            result = {
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

            # Add counts if available
            if has_counts:
                result.update({
                    "joint_med_counts": np.asarray(z["joint_median_counts"], int),
                    "joint_t2_counts": np.asarray(z["joint_top2_counts"], int),
                    "cond_med_counts": np.asarray(z["rmax_median_counts"], int),
                    "cond_t2_counts": np.asarray(z["rmax_top2_counts"], int),
                    "r2_marginal_counts": np.asarray(z["r2_marginal_counts"], int),
                })
            else:
                result.update({
                    "joint_med_counts": None,
                    "joint_t2_counts": None,
                    "cond_med_counts": None,
                    "cond_t2_counts": None,
                    "r2_marginal_counts": None,
                })

            return result
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

def posterior_line(x, y, title, xlab, ylab="P(Ace | Signals)", counts=None):
    """Create a line plot for posterior probabilities with optional confidence intervals.

    Args:
        x: X-axis values
        y: Y-axis values (probabilities)
        title: Plot title
        xlab: X-axis label
        ylab: Y-axis label
        counts: Optional count array for confidence intervals
    """
    xs = np.asarray(x)
    ys = np.asarray(y)
    fig = go.Figure()

    # Add main line trace
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

def _compute_binomial_ci(p, n, confidence=0.95):
    """Compute Wilson score confidence interval for binomial proportion.

    Args:
        p: Probability array (observed proportions)
        n: Count array (total sample sizes for each probability)
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) arrays
    """
    import scipy.stats

    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)

    # Z-score for the confidence level
    z = scipy.stats.norm.ppf((1 + confidence) / 2)

    # Wilson score interval formula
    # Handles edge cases better than normal approximation
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4*n**2))) / denominator

    # Handle zero sample size cases
    mask = (n > 0)
    lower = np.zeros_like(p)
    upper = np.ones_like(p)
    lower[mask] = np.clip(center[mask] - margin[mask], 0, 1)
    upper[mask] = np.clip(center[mask] + margin[mask], 0, 1)

    return lower, upper

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

    # Row 1: Posterior type + Payoff scaling (in 2 columns)
    col1, col2 = st.columns([1, 1])
    with col1:
        post_type = st.radio("Posterior type", ["Conditional", "Joint Conditional"],
                            horizontal=True, key=f"post_type_{tag}")
    with col2:
        scale_pay = 1 if st.toggle("Payoff scaling", value=False, key=f"scale_pay_post_{tag}") else 0

    # Row 2: Signal type selector
    if post_type == "Conditional":
        signal_options = ["Median", "Top 2 rank sum", "Second rank (R2)"]
    else:  # Joint Conditional
        signal_options = ["Median", "Top 2 rank sum"]
    signal_label = st.selectbox("Signal type", signal_options, key=f"post_sig_{tag}")

    # Map signal label to internal type
    if signal_label == "Median":
        signal_type = "median"
    elif signal_label == "Top 2 rank sum":
        signal_type = "top2"
    else:  # Second rank (R2)
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
                counts_mat = post_data["cond_med_counts"]
            elif signal_type == "top2":
                x_vals = post_data["cond_t2_keys"]
                y_vals = post_data["cond_t2_mat"][:, rmax_idx]
                sig_name = "Top 2 rank sum"
                counts_mat = post_data["cond_t2_counts"]
            else:  # r2
                x_vals = np.arange(2, 14)  # R2 values from 2 to 13 (cannot be 14/Ace)
                y_vals = post_data["r2_marginal_mat"][:, rmax_idx]
                sig_name = "Second rank (R2)"
                counts_mat = post_data["r2_marginal_counts"]

            title = f"P(Max rank = {max_rank_choice} | {sig_name} = x)"
            ylab_text = f"P(Max rank = {max_rank_choice} | signal)"

            fig = go.Figure()

            # Main line trace
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
                counts_mat = post_data["cond_med_counts"]
            elif signal_type == "top2":
                x_vals = post_data["cond_t2_keys"]
                y_vals = post_data["cond_t2_mat"][:, ace_idx]
                sig_name = "Top 2 rank sum"
                counts_mat = post_data["cond_t2_counts"]
            else:  # r2
                x_vals = np.arange(2, 14)  # R2 values from 2 to 13 (cannot be 14/Ace)
                y_vals = post_data["r2_marginal_mat"][:, ace_idx]
                sig_name = "Second rank (R2)"
                counts_mat = post_data["r2_marginal_counts"]

            title = f"P(Ace | {sig_name} = x)"
            # Compute sample sizes for confidence intervals
            sample_counts = counts_mat.sum(axis=1) if counts_mat is not None else None
            st.plotly_chart(posterior_line(x_vals, y_vals, title, sig_name, counts=sample_counts),
                          width="stretch", key=f"post_cond_{tag}")

    # ========== JOINT CONDITIONAL POSTERIORS ==========
    else:  # post_type == "Joint Conditional"
        # P(Rmax | signal, R2) - like 2-Stage mode

        # Show Max Rank slider when payoff scaling is ON
        if scale_pay == 1:
            max_rank_choice = st.slider("Max Rank", min_value=2, max_value=14, value=14,
                                       key=f"max_rank_joint_{tag}")
            rmax_idx = max_rank_choice - 2
            rank_label = f"Max rank = {max_rank_choice}"
        else:
            rmax_idx = ACE_RANK - 2  # Ace = 14, idx = 12
            rank_label = "Ace"

        # X-axis selector
        col_x1, col_x2 = st.columns([1, 2])
        with col_x1:
            x_axis_opts = [signal_label, "Second rank (R2)"]
            x_axis_choice = st.selectbox("X-axis", x_axis_opts, key=f"x_axis_joint_{tag}")

        if x_axis_choice == signal_label:
            # Signal on X-axis, R2 as parameter
            with col_x2:
                r2_val = st.slider("Second Rank (R2)", min_value=2, max_value=13, value=10,
                                  key=f"r2_joint_{tag}")

            # Extract data
            keys = post_data["joint_med_keys"] if signal_type == "median" else post_data["joint_t2_keys"]
            mat = post_data["joint_med_mat"] if signal_type == "median" else post_data["joint_t2_mat"]
            counts_mat = post_data["joint_med_counts"] if signal_type == "median" else post_data["joint_t2_counts"]

            # Slice for R2 and selected max rank
            r2_idx = int(r2_val) - 2
            y_vals = mat[:, r2_idx, rmax_idx]
            x_vals = keys

            # Compute sample sizes for confidence intervals
            # For joint posteriors at fixed R2, sum across all Rmax values for each signal bucket
            sample_counts = counts_mat[:, r2_idx, :].sum(axis=1) if counts_mat is not None else None

            title = f"P({rank_label} | {signal_label} = x and R2 = {r2_val})"
            xlab = signal_label
            st.plotly_chart(posterior_line(x_vals, y_vals, title, xlab, counts=sample_counts),
                          width="stretch", key=f"post_joint_{tag}")

        else:  # "Second rank (R2)" on X-axis
            # R2 on X-axis, signal as parameter
            keys = post_data["joint_med_keys"] if signal_type == "median" else post_data["joint_t2_keys"]
            mat = post_data["joint_med_mat"] if signal_type == "median" else post_data["joint_t2_mat"]
            counts_mat = post_data["joint_med_counts"] if signal_type == "median" else post_data["joint_t2_counts"]

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
                y_vals = mat[bucket_idx, :, rmax_idx]
                x_vals = np.arange(2, 14)  # Second rank values 2-13 (cannot be 14/Ace)

                # Compute sample sizes for confidence intervals
                # For fixed signal bucket, sum across all Rmax values for each R2
                sample_counts = counts_mat[bucket_idx, :, :].sum(axis=1) if counts_mat is not None else None

                title = f"P({rank_label} | {signal_label} = {sig_val} and R2 = x)"
                xlab = "Second rank (R2)"
                st.plotly_chart(posterior_line(x_vals, y_vals, title, xlab, counts=sample_counts),
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
            req = {"sd_levels_by_n", "best_means_by_n", "best_weights_by_n", "meta"}
            zset = set(z.files)
            missing = sorted(list(req - zset))
            if missing:
                st.error(f"Frontier NPZ missing arrays: {', '.join(missing)}")
                return None

            # Load data
            data = {
                "sd_step": float(z.get("sd_step", 0.1)),
                "sd_levels_by_n": z["sd_levels_by_n"],
                "best_means_by_n": z["best_means_by_n"],
                "best_weights_by_n": z["best_weights_by_n"],
                "meta": json.loads(str(z["meta"])),
            }

            # Load premium card hits if available
            if "best_ace_hits_by_n" in zset:
                data["best_ace_hits_by_n"] = z["best_ace_hits_by_n"]
            if "best_king_hits_by_n" in zset:
                data["best_king_hits_by_n"] = z["best_king_hits_by_n"]
            if "best_queen_hits_by_n" in zset:
                data["best_queen_hits_by_n"] = z["best_queen_hits_by_n"]

            return data
    except Exception as e:
        st.error(f"Failed to load frontier NPZ: {e}")
        return None

def _gather_means(fd):
    """Helper: gather means for y-range computation."""
    if fd is None:
        return []
    means = []
    mean_by_n = fd["best_means_by_n"]
    for n_sig in range(len(mean_by_n)):
        mv = mean_by_n[n_sig]
        if len(mv) > 0:
            means.append(np.asarray(mv, float))
    return means

def _coarsen_frontier(sd_vals, mean_vals, weights, ace_hits, king_hits, queen_hits, sd_step):
    """Re-bin frontier data to coarser SD granularity. Within each bin, select the point with highest mean return."""
    if len(sd_vals) == 0:
        return sd_vals, mean_vals, weights, ace_hits, king_hits, queen_hits

    # Bin by floor(sd / sd_step)
    bins = np.floor(np.asarray(sd_vals, float) / sd_step).astype(int)
    max_bin = int(np.max(bins))

    # For each bin, pick index with highest mean
    coarse_sd = []
    coarse_mean = []
    coarse_weights = []
    coarse_ace = []
    coarse_king = []
    coarse_queen = []

    for b in range(max_bin + 1):
        mask = (bins == b)
        if not np.any(mask):
            continue
        indices = np.flatnonzero(mask)
        means_in_bin = np.asarray(mean_vals, float)[mask]
        best_idx_in_bin = np.argmax(means_in_bin)
        best_idx = indices[best_idx_in_bin]

        coarse_sd.append(float(sd_vals[best_idx]))
        coarse_mean.append(float(mean_vals[best_idx]))
        coarse_weights.append(weights[best_idx])
        if len(ace_hits) > 0:
            coarse_ace.append(ace_hits[best_idx])
        if len(king_hits) > 0:
            coarse_king.append(king_hits[best_idx])
        if len(queen_hits) > 0:
            coarse_queen.append(queen_hits[best_idx])

    return (np.array(coarse_sd), np.array(coarse_mean), coarse_weights,
            np.array(coarse_ace) if len(coarse_ace) > 0 else np.array([]),
            np.array(coarse_king) if len(coarse_king) > 0 else np.array([]),
            np.array(coarse_queen) if len(coarse_queen) > 0 else np.array([]))

def _ssq_extents(fd):
    """Compute global Σw² range for consistent color mapping."""
    if fd is None:
        return None
    weights_by_n = fd["best_weights_by_n"]
    vals = []
    for n_sig in range(len(weights_by_n)):
        weights = weights_by_n[n_sig]
        for w_vec in weights:
            vals.append(float(np.sum(np.asarray(w_vec, float) ** 2)))
    if not vals:
        return None
    return (min(vals), max(vals))

def _build_frontier_fig(fd, sd_step, y_range_override=None, cmin_override=None, cmax_override=None, n_signals_filter="All"):
    """Build frontier figure with enhanced hover info and concentration coloring."""
    if fd is None:
        return None, []
    fig = go.Figure()
    sd_by_n = fd["sd_levels_by_n"]
    mean_by_n = fd["best_means_by_n"]
    weights_by_n = fd["best_weights_by_n"]
    ace_hits_by_n = fd.get("best_ace_hits_by_n", [])
    king_hits_by_n = fd.get("best_king_hits_by_n", [])
    queen_hits_by_n = fd.get("best_queen_hits_by_n", [])
    meta = fd.get("meta", {})
    total_rounds = meta.get("total_rounds", 100000)
    scale_pay = meta.get("params", {}).get("scale_pay", 0)

    points = []
    all_sum_sq_weights = []

    # Determine which signal counts to plot
    if n_signals_filter == "All":
        n_sigs_to_plot = range(len(sd_by_n))
    else:
        n_sigs_to_plot = [n_signals_filter] if n_signals_filter < len(sd_by_n) else []

    for n_sig in n_sigs_to_plot:
        sd_vals = sd_by_n[n_sig]
        mean_vals = mean_by_n[n_sig]
        weights = weights_by_n[n_sig]
        ace_hits = ace_hits_by_n[n_sig] if n_sig < len(ace_hits_by_n) else np.array([])
        king_hits = king_hits_by_n[n_sig] if n_sig < len(king_hits_by_n) else np.array([])
        queen_hits = queen_hits_by_n[n_sig] if n_sig < len(queen_hits_by_n) else np.array([])
        if len(sd_vals) == 0:
            continue

        # Apply SD coarsening
        sd_vals, mean_vals, weights, ace_hits, king_hits, queen_hits = _coarsen_frontier(
            sd_vals, mean_vals, weights, ace_hits, king_hits, queen_hits, sd_step
        )
        sum_sq_weights = [float(np.sum(np.asarray(w_vec, float) ** 2)) for w_vec in weights]
        points.append(dict(
            n=n_sig, sd=np.asarray(sd_vals, float), mean=np.asarray(mean_vals, float),
            ssq=np.asarray(sum_sq_weights, float), weights=weights,
            ace_hits=ace_hits, king_hits=king_hits, queen_hits=queen_hits
        ))
        all_sum_sq_weights.extend(sum_sq_weights)

    # Determine global frontier (Pareto-efficient) across all points
    combined = []
    for i, p in enumerate(points):
        for j in range(len(p["sd"])):
            combined.append((float(p["sd"][j]), float(p["mean"][j]), (i, j)))
    combined.sort(key=lambda t: (t[0], -t[1]))
    frontier_pos = set()
    best_mean = -1e9
    for sd, mean, pos in combined:
        if mean > best_mean + 1e-12:
            frontier_pos.add(pos)
            best_mean = mean

    if cmin_override is not None and cmax_override is not None:
        vmin_global, vmax_global = float(cmin_override), float(cmax_override)
    elif len(all_sum_sq_weights) > 0:
        vmin_global = float(min(all_sum_sq_weights))
        vmax_global = float(max(all_sum_sq_weights))
    else:
        vmin_global, vmax_global = 0.0, 1.0

    for i, p in enumerate(points):
        n_sig = p["n"]
        sd_vals, mean_vals, ssq = p["sd"], p["mean"], p["ssq"]
        weights_list = p["weights"]
        ace_hits_list, king_hits_list, queen_hits_list = p["ace_hits"], p["king_hits"], p["queen_hits"]

        # Keep all frontier points; keep every other interior point to reduce clutter
        keep_idx = [j for j in range(len(sd_vals)) if (i, j) in frontier_pos or (j % 2 == 0)]
        if not keep_idx:
            continue
        xs, ys, cs = sd_vals[keep_idx], mean_vals[keep_idx], ssq[keep_idx]

        # Build enhanced hover texts
        hover_texts = []
        for idx_k, k in enumerate(keep_idx):
            mean_val, sd_val = ys[idx_k], xs[idx_k]
            sharpe = mean_val / sd_val if sd_val > 0.001 else 0.0
            w_vec = np.asarray(weights_list[k], float)
            line1 = " | ".join([f"Pile {i+1}: {w_vec[i]:.1%}" for i in range(3)])
            line2 = " | ".join([f"Pile {i+1}: {w_vec[i]:.1%}" for i in range(3, 6)])
            line3 = " | ".join([f"Pile {i+1}: {w_vec[i]:.1%}" for i in range(6, 9)])

            ace_hit = int(ace_hits_list[k]) if k < len(ace_hits_list) else 0
            king_hit = int(king_hits_list[k]) if k < len(king_hits_list) else 0
            queen_hit = int(queen_hits_list[k]) if k < len(queen_hits_list) else 0

            hit_rate_str = ""
            if total_rounds and total_rounds > 0:
                ace_pct = (ace_hit / total_rounds) * 100
                hit_rate_str = f"<br><br><b>Hit Rates:</b><br>Ace: {ace_pct:.2f}%"
                if scale_pay == 1:
                    king_pct = (king_hit / total_rounds) * 100
                    queen_pct = (queen_hit / total_rounds) * 100
                    hit_rate_str += f", King: {king_pct:.2f}%, Queen: {queen_pct:.2f}%"

            sim_display = f"{total_rounds:,}" if total_rounds else "N/A"
            hover_text = (
                f"<b>Signals: N = {n_sig}</b><br>"
                f"<b>Simulations:</b> {sim_display}<br>"
                f"<b>Mean Net Return:</b> {mean_val:.2f}%<br>"
                f"<b>Std Dev:</b> {sd_val:.2f}%<br>"
                f"<b>Sharpe Ratio:</b> {sharpe:.2f}<br>"
                f"<b>Σw²:</b> {cs[idx_k]:.3f}"
                f"{hit_rate_str}<br>"
                f"<br><b>Weights (sorted high→low by Stage-1 EV):</b><br>"
                f"{line1}<br>{line2}<br>{line3}"
            )
            hover_texts.append(hover_text)

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", name=f"N = {n_sig}",
            marker=dict(size=16, color=cs, colorscale=[[0, "#2b8cbe"], [1, "#08306b"]],
                        cmin=vmin_global, cmax=vmax_global, showscale=False, line=dict(width=0)),
            text=[str(n_sig)] * len(keep_idx), textposition="middle center", textfont=dict(size=11, color="white"),
            hovertemplate="%{hovertext}<extra></extra>", hovertext=hover_texts, showlegend=False, opacity=ALPHA,
        ))

    yaxis_cfg = dict(title=dict(text="Mean Net Return (%)", font=_DEF_FONT), tickfont=_DEF_FONT, showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    if y_range_override is not None:
        yaxis_cfg.update(range=list(map(float, y_range_override)))
    fig.update_layout(
        template="plotly_white", font=dict(family="Roboto, Arial, sans-serif", size=15),
        xaxis=dict(title=dict(text="Standard Deviation (%)", font=_DEF_FONT), tickfont=_DEF_FONT, showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
        yaxis=yaxis_cfg, height=600, hovermode="closest", margin=dict(l=10, r=10, t=10, b=50),
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Roboto, Arial, sans-serif", bordercolor="#2b8cbe", align="left", namelength=-1),
    )
    return fig

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

# View selector: Simulation Results vs Efficient Frontiers vs Posteriors
view = st.radio("View", ["Simulation Results", "Mean-Variance Frontier", "Posteriors"], horizontal=True, key="top_view")

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

elif view == "Efficient Frontiers":
    # ==============================
    # EFFICIENT FRONTIERS VIEW
    # ==============================

    # Help text + shared y-range toggle
    row = st.columns([1, 0.25])
    with row[0]:
        st.markdown(
            "<div style='font-size:0.9rem; font-style:italic;'>"
            "All returns are net returns in percentages — "
            "100×(Cash at end - Total budget)/Total budget"
            "</div>",
            unsafe_allow_html=True,
        )
    with row[1]:
        fix_y_mv = st.toggle("Fix Y axis range", value=True, key="fix_y_mv")

    # Per-panel controls
    ctlA, ctlB = st.columns(2)
    with ctlA:
        st.markdown("### Panel A")
        topA = st.columns([2, 1])
        with topA[0]:
            frontier_sp_A = st.radio("Payoff scaling", ["Off (Ace-only)", "On (Scaled)"], horizontal=True, key="frontier_sp_A")
        with topA[1]:
            frontier_sig_A = st.selectbox("Signal type", ["Median", "Top 2 rank sum"], key="frontier_sig_A")
        sp_A = 1 if "On" in frontier_sp_A else 0
        sig_A = "median" if frontier_sig_A == "Median" else "top2"
        rowA = st.columns([1, 1, 1])
        with rowA[0]:
            alpha_A = st.select_slider("Stage 1 allocation",
                                       options=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                                               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
                                       value=0.1,
                                       format_func=lambda v: f"{int(v*100)}:{int((1-v)*100)}",
                                       key="frontier_alpha_A")
        with rowA[1]:
            signal_cost_A = st.select_slider("Signal cost", options=[0, 3, 9], value=3, format_func=lambda v: f"£{v}", key="signal_cost_A")
        with rowA[2]:
            sd_step_A = st.select_slider("SD binning", options=[0.1, 1, 2, 5], value=5, format_func=lambda v: f"{v}pp bin", key="sd_step_A")
        rowA2 = st.columns([1])
        with rowA2[0]:
            n_signals_A = st.select_slider("Number of signals", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "All"], value="All", format_func=lambda v: "All" if v == "All" else str(v), key="n_signals_A")
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    with ctlB:
        st.markdown("### Panel B")
        topB = st.columns([2, 1])
        with topB[0]:
            frontier_sp_B = st.radio("Payoff scaling", ["Off (Ace-only)", "On (Scaled)"], horizontal=True, key="frontier_sp_B")
        with topB[1]:
            frontier_sig_B = st.selectbox("Signal type", ["Median", "Top 2 rank sum"], index=1, key="frontier_sig_B")
        sp_B = 1 if "On" in frontier_sp_B else 0
        sig_B = "top2" if frontier_sig_B == "Top 2 rank sum" else "median"
        rowB = st.columns([1, 1, 1])
        with rowB[0]:
            alpha_B = st.select_slider("Stage 1 allocation",
                                       options=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                                               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
                                       value=0.1,
                                       format_func=lambda v: f"{int(v*100)}:{int((1-v)*100)}",
                                       key="frontier_alpha_B")
        with rowB[1]:
            signal_cost_B = st.select_slider("Signal cost", options=[0, 3, 9], value=3, format_func=lambda v: f"£{v}", key="signal_cost_B")
        with rowB[2]:
            sd_step_B = st.select_slider("SD binning", options=[0.1, 1, 2, 5], value=5, format_func=lambda v: f"{v}pp bin", key="sd_step_B")
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Resolve files
    frontier_dir = Path("frontier_output/").resolve()
    raw_A = dict(signal_cost=float(signal_cost_A), scale_pay=sp_A, scale_param=(0.25 if sp_A == 1 else 0.0), ace_payout=20.0)
    raw_B = dict(signal_cost=float(signal_cost_B), scale_pay=sp_B, scale_param=(0.25 if sp_B == 1 else 0.0), ace_payout=20.0)
    _, key_A = _canonicalize(raw_A)
    _, key_B = _canonicalize(raw_B)
    tag_A = f"a{int(round(float(alpha_A)*100)):03d}"
    tag_B = f"a{int(round(float(alpha_B)*100)):03d}"
    file_A = frontier_dir / f"{key_A}_{sig_A}_{tag_A}.npz"
    file_B = frontier_dir / f"{key_B}_{sig_B}_{tag_B}.npz"

    data_A = load_frontier_npz(str(file_A))
    data_B = load_frontier_npz(str(file_B))

    # Compute shared y-range if requested
    y_range = None
    if fix_y_mv and (data_A is not None) and (data_B is not None):
        all_means = [*_gather_means(data_A), *_gather_means(data_B)]
        if len(all_means) > 0:
            y_min = float(min(np.min(a) for a in all_means))
            y_max = float(max(np.max(a) for a in all_means))
            y_range = _padded_range(y_min, y_max)

    # Compute global Σw² range across both datasets for consistent color mapping
    extA = _ssq_extents(data_A)
    extB = _ssq_extents(data_B)
    if extA and extB:
        global_vmin = min(extA[0], extB[0])
        global_vmax = max(extA[1], extB[1])
    else:
        global_vmin, global_vmax = None, None

    # Render charts side-by-side
    colA, colB = st.columns(2)

    with colA:
        if data_A is None:
            st.info(f"Frontier data not found: {file_A.name}")
            st.caption("Run frontier.py to generate frontier data.")
        else:
            st.markdown(f"**Fixed:** Ace payoff=20X" + (", Scale param=0.25" if sp_A == 1 else ""))
            figA = _build_frontier_fig(data_A, sd_step_A, y_range, global_vmin, global_vmax)
            st.plotly_chart(figA, use_container_width=True, key="mv_frontier_A")

    with colB:
        if data_B is None:
            st.info(f"Frontier data not found: {file_B.name}")
            st.caption("Run frontier.py to generate frontier data.")
        else:
            st.markdown(f"**Fixed:** Ace payoff=20X" + (", Scale param=0.25" if sp_B == 1 else ""))
            figB = _build_frontier_fig(data_B, sd_step_B, y_range, global_vmin, global_vmax)
            st.plotly_chart(figB, use_container_width=True, key="mv_frontier_B")

    # Shared legend at bottom
    if (global_vmin is not None) and (global_vmax is not None):
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; gap: 20px; width: 100%; max-width: 700px; margin: 20px auto;'>
                <div style='white-space: nowrap;'>Legend: Σw² (portfolio concentration)</div>
                <div style='flex: 1;'>
                    <div style='height: 30px; background: linear-gradient(to right, #2b8cbe, #08306b); border: 1px solid #ccc; border-radius: 4px;'></div>
                    <div style='display: flex; justify-content: space-between; margin-top: 5px; font-size: 13px; color: #666;'>
                        <span>{global_vmin:.3f}</span>
                        <span>{global_vmax:.3f}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

else:  # view == "Posteriors"
    # ==============================
    # POSTERIORS VIEW
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

# vis_f.py — Mean-Variance Frontier Visualizer
# Interactive viewer for information-limited frontier results

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ==============================
# App config
# ==============================
st.set_page_config(page_title="IL Frontier Viewer", layout="wide")
st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp, .block-container { background: #ffffff !important; }
      .block-container { padding-top: 1.25rem; padding-bottom: 1.0rem; }
      .stSelectbox label, .stNumberInput label { font-size: 0.92rem; margin-bottom: .18rem; }
      .stSelectbox > div[data-baseweb="select"] { min-height: 36px; }
      .js-plotly-plot .plotly .main-svg { overflow: visible !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Constants
# ==============================
ALPHA = 0.7  # Marker opacity


# ==============================
# Helper functions
# ==============================
def _fmt_float_id(x: float, nd=4) -> str:
    """Format float for filename ID."""
    s = f"{x:.{nd}f}"
    return s.replace(".", "p")


def _canonicalize(raw):
    """Build canonical key_id from raw params (matches fns.py logic)."""
    sc = float(raw["signal_cost"])
    sp = int(raw["scale_pay"])
    s = float(raw["scale_param"])
    ap = float(raw["ace_payout"])

    # Normalize scale_param: use 0 (int) when scale_pay is off
    s_norm = s if sp == 1 else 0

    norm = dict(signal_cost=sc, scale_pay=sp, scale_param=s_norm, ace_payout=ap)

    # Build key_id: format as integer if it's an int type, otherwise format as float
    sc_id = _fmt_float_id(sc) if isinstance(sc, float) else str(sc)
    sp_id = str(sp)
    s_id = _fmt_float_id(s_norm) if isinstance(s_norm, float) else str(s_norm)
    ap_id = _fmt_float_id(ap) if isinstance(ap, float) else str(ap)
    key_id = f"sc{sc_id}_sp{sp_id}_s{s_id}_ap{ap_id}"

    return norm, key_id


def _padded_range(lo, hi, frac=0.03, min_pad=0.5):
    """Add padding to a range for better visualization."""
    span = hi - lo
    if span < 1e-9:
        return [lo - min_pad, hi + min_pad]
    pad = max(frac * span, min_pad)
    return [lo - pad, hi + pad]


@st.cache_data(show_spinner=False)
def load_frontier_npz(npz_path: str):
    """Load frontier NPZ file."""
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
                "sd_step": float(z.get("sd_step", 1.0)),
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


# ==============================
# FRONTIER PAGE
# ==============================
st.header("Mean-Variance Frontier")
st.caption("Efficient frontiers showing mean vs. standard deviation of net returns")

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
        frontier_sig_A = st.selectbox("Signal type", ["Median", "Top 2"], key="frontier_sig_A")
    sp_A = 1 if "On" in frontier_sp_A else 0
    sig_A = "median" if frontier_sig_A == "Median" else "top2"
    rowA = st.columns([1, 1, 1])
    with rowA[0]:
        # Match ALPHA_GRID from frontier.py (11 points, 0.1 step)
        alpha_A = st.select_slider("Stage 1 allocation",
                                   options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                   value=0.5,
                                   format_func=lambda v: f"{int(v*100)}:{int((1-v)*100)}",
                                   key="frontier_alpha_A")
    with rowA[1]:
        max_n_A = st.slider("Max signals", min_value=0, max_value=9, value=9, key="max_n_sig_frontier_A")
    with rowA[2]:
        st.empty()

with ctlB:
    st.markdown("### Panel B")
    topB = st.columns([2, 1])
    with topB[0]:
        frontier_sp_B = st.radio("Payoff scaling", ["Off (Ace-only)", "On (Scaled)"], horizontal=True, key="frontier_sp_B")
    with topB[1]:
        frontier_sig_B = st.selectbox("Signal type", ["Median", "Top 2"], key="frontier_sig_B")
    sp_B = 1 if "On" in frontier_sp_B else 0
    sig_B = "median" if frontier_sig_B == "Median" else "top2"
    rowB = st.columns([1, 1, 1])
    with rowB[0]:
        # Match ALPHA_GRID from frontier.py (11 points, 0.1 step)
        alpha_B = st.select_slider("Stage 1 allocation",
                                   options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                   value=0.5,
                                   format_func=lambda v: f"{int(v*100)}:{int((1-v)*100)}",
                                   key="frontier_alpha_B")
    with rowB[1]:
        max_n_B = st.slider("Max signals", min_value=0, max_value=9, value=9, key="max_n_sig_frontier_B")
    with rowB[2]:
        st.empty()

# Resolve files
frontier_dir = Path("frontier_output_m/").resolve()
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
def _build_fig(fd, max_n, y_range_override=None, cmin_override=None, cmax_override=None):
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
    combined = []
    for i, p in enumerate(points):
        sd_vals = p["sd"]
        mean_vals = p["mean"]
        for j in range(len(sd_vals)):
            combined.append((float(sd_vals[j]), float(mean_vals[j]), (i, j)))
    combined.sort(key=lambda t: (t[0], -t[1]))
    frontier_pos = set()
    best_mean = -1e9
    for sd, mean, pos in combined:
        if mean > best_mean + 1e-12:
            frontier_pos.add(pos)
            best_mean = mean

    if cmin_override is not None and cmax_override is not None:
        vmin_global = float(cmin_override)
        vmax_global = float(cmax_override)
    elif len(all_sum_sq_weights) > 0:
        vmin_global = float(min(all_sum_sq_weights))
        vmax_global = float(max(all_sum_sq_weights))
    else:
        vmin_global = 0.0
        vmax_global = 1.0

    for i, p in enumerate(points):
        n_sig = p["n"]
        sd_vals = p["sd"]
        mean_vals = p["mean"]
        ssq = p["ssq"]
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

    yaxis_cfg = dict(title=dict(text="Mean Return (%)", font=dict(size=13)), tickfont=dict(size=16), showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    if y_range_override is not None:
        yaxis_cfg.update(range=list(map(float, y_range_override)))
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Roboto, Arial, sans-serif", size=15),
        xaxis=dict(title=dict(text="Standard Deviation (%)", font=dict(size=13)), tickfont=dict(size=16), showgrid=True, gridcolor="rgba(128,128,128,0.1)"),
        yaxis=yaxis_cfg,
        height=600,
        hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=50),
    )
    return fig

# Compute global Σw² range across both datasets for consistent color mapping
def _ssq_extents(fd, max_n):
    if fd is None:
        return None
    weights_by_n = fd["best_weights_by_n"]
    vals = []
    for n_sig in range(min(len(weights_by_n), max_n + 1)):
        weights = weights_by_n[n_sig]
        for w_vec in weights:
            vals.append(float(np.sum(np.asarray(w_vec, float) ** 2)))
    if not vals:
        return None
    return (min(vals), max(vals))

extA = _ssq_extents(data_A, max_n_A)
extB = _ssq_extents(data_B, max_n_B)
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
        st.markdown("**Fixed:** Signal cost=£3, Ace payoff=20X" + (", Scale param=0.25" if sp_A == 1 else ""))
        figA = _build_fig(data_A, max_n_A, y_range, global_vmin, global_vmax)
        st.plotly_chart(figA, use_container_width=True, key="mv_frontier_A")

with colB:
    if data_B is None:
        st.info(f"Frontier data not found: {file_B.name}")
        st.caption("Run frontier.py to generate frontier data.")
    else:
        st.markdown("**Fixed:** Signal cost=£3, Ace payoff=20X" + (", Scale param=0.25" if sp_B == 1 else ""))
        figB = _build_fig(data_B, max_n_B, y_range, global_vmin, global_vmax)
        st.plotly_chart(figB, use_container_width=True, key="mv_frontier_B")

# Shared legend at bottom
if (global_vmin is not None) and (global_vmax is not None):
    legend_fig = go.Figure()
    legend_fig.add_trace(
        go.Heatmap(
            z=[[global_vmin, global_vmax]],
            showscale=True,
            colorscale=[[0, "#2b8cbe"], [1, "#08306b"]],
            zmin=global_vmin,
            zmax=global_vmax,
            colorbar=dict(
                title=dict(text="Σw² (concentration)"),
                orientation="h",
                x=0.5, xanchor="center",
                y=0.0, yanchor="bottom",
                len=1.0,
                thickness=16,
            ),
            hoverinfo="skip",
            xgap=1,
            ygap=1,
        )
    )
    legend_fig.update_layout(
        template="plotly_white",
        height=80,
        width=114,
        margin=dict(l=6, r=6, t=0, b=6),
    )
    legend_fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    legend_fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    st.plotly_chart(legend_fig, use_container_width=False, key="mv_shared_legend")

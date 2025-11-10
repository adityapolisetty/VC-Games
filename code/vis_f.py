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
st.set_page_config(page_title="Frontier Viewer", layout="wide")
st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp, .block-container { background: #ffffff !important; }
      .block-container { padding-top: 1.25rem; padding-bottom: 1.0rem; }
      .stSelectbox label, .stNumberInput label { font-size: 0.92rem; margin-bottom: .18rem; }
      .stSelectbox > div[data-baseweb="select"] { min-height: 36px; }
      .js-plotly-plot .plotly .main-svg { overflow: visible !important; }
      .js-plotly-plot .plotly .hoverlayer { overflow: visible !important; }
      .js-plotly-plot .plotly .hoverlayer .hovertext {
        max-width: 600px !important;
        min-width: 400px !important;
        white-space: normal !important;
        padding: 12px !important;
        line-height: 1.5 !important;
      }
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


@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes only
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
    fix_y_mv = st.toggle("Fix Y axis range", value=False, key="fix_y_mv")

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
                                   value=0.1,
                                   format_func=lambda v: f"{int(v*100)}:{int((1-v)*100)}",
                                   key="frontier_alpha_A")
    with rowA[1]:
        signal_cost_A = st.select_slider("Signal cost", options=[0, 3, 7, 9], value=3, format_func=lambda v: f"£{v}", key="signal_cost_A")
    with rowA[2]:
        sd_step_A = st.select_slider("SD step", options=[0.1, 1, 2, 5], value=5, format_func=lambda v: f"±{v}pp", key="sd_step_A")
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Spacer

with ctlB:
    st.markdown("### Panel B")
    topB = st.columns([2, 1])
    with topB[0]:
        frontier_sp_B = st.radio("Payoff scaling", ["Off (Ace-only)", "On (Scaled)"], horizontal=True, key="frontier_sp_B")
    with topB[1]:
        frontier_sig_B = st.selectbox("Signal type", ["Median", "Top 2"], key="frontier_sig_B")
    sp_B = 1 if "On" in frontier_sp_B else 0
    sig_B = "top2" if frontier_sig_B == "Top 2" else "median"
    rowB = st.columns([1, 1, 1])
    with rowB[0]:
        # Match ALPHA_GRID from frontier.py (11 points, 0.1 step)
        alpha_B = st.select_slider("Stage 1 allocation",
                                   options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                   value=0.1,
                                   format_func=lambda v: f"{int(v*100)}:{int((1-v)*100)}",
                                   key="frontier_alpha_B")
    with rowB[1]:
        signal_cost_B = st.select_slider("Signal cost", options=[0, 3, 7, 9], value=3, format_func=lambda v: f"£{v}", key="signal_cost_B")
    with rowB[2]:
        sd_step_B = st.select_slider("SD step", options=[0.1, 1, 2, 5], value=5, format_func=lambda v: f"±{v}pp", key="sd_step_B")
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Spacer

# Resolve files
frontier_dir = Path("frontier_output/").resolve()
raw_A = dict(signal_cost=float(signal_cost_A), scale_pay=sp_A, scale_param=(0.25 if sp_A == 1 else 0.0), ace_payout=20.0)
raw_B = dict(signal_cost=float(signal_cost_B), scale_pay=sp_B, scale_param=(0.25 if sp_B == 1 else 0.0), ace_payout=20.0)
_, key_A = _canonicalize(raw_A)
_, key_B = _canonicalize(raw_B)
tag_A = f"a{int(round(float(alpha_A)*10)):02d}"
tag_B = f"a{int(round(float(alpha_B)*10)):02d}"
file_A = frontier_dir / f"{key_A}_{sig_A}_{tag_A}.npz"
file_B = frontier_dir / f"{key_B}_{sig_B}_{tag_B}.npz"

data_A = load_frontier_npz(str(file_A))
data_B = load_frontier_npz(str(file_B))

# Helper: gather means for y-range
def _gather_means(fd):
    if fd is None:
        return []
    means = []
    mean_by_n = fd["best_means_by_n"]
    for n_sig in range(len(mean_by_n)):
        mv = mean_by_n[n_sig]
        if len(mv) > 0:
            means.append(np.asarray(mv, float))
    return means

y_range = None
if fix_y_mv and (data_A is not None) and (data_B is not None):
    all_means = [*_gather_means(data_A), *_gather_means(data_B)]
    if len(all_means) > 0:
        y_min = float(min(np.min(a) for a in all_means))
        y_max = float(max(np.max(a) for a in all_means))
        y_range = _padded_range(y_min, y_max)

# Helper: coarsen frontier data by SD binning
def _coarsen_frontier(sd_vals, mean_vals, weights, ace_hits, king_hits, queen_hits, sd_step):
    """
    Re-bin frontier data to coarser SD granularity.
    Within each bin, select the point with highest mean return.
    """
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

# Helper: build figure
def _build_fig(fd, sd_step, y_range_override=None, cmin_override=None, cmax_override=None):
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
    total_rounds = meta.get("total_rounds", 100000)  # Default to 100k if not in metadata
    scale_pay = meta.get("params", {}).get("scale_pay", 0)

    points = []
    all_sum_sq_weights = []
    all_custom_data = []  # Store metadata for all points across traces
    for n_sig in range(len(sd_by_n)):
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
            n=n_sig,
            sd=np.asarray(sd_vals, float),
            mean=np.asarray(mean_vals, float),
            ssq=np.asarray(sum_sq_weights, float),
            weights=weights,
            ace_hits=ace_hits,
            king_hits=king_hits,
            queen_hits=queen_hits
        ))
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
        weights_list = p["weights"]
        ace_hits_list = p["ace_hits"]
        king_hits_list = p["king_hits"]
        queen_hits_list = p["queen_hits"]

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

        # Build enhanced hover texts and customdata
        hover_texts = []
        custom_data = []
        for idx_k, k in enumerate(keep_idx):
            # Basic metrics
            mean_val = ys[idx_k]
            sd_val = xs[idx_k]
            sharpe = mean_val / sd_val if sd_val > 0.001 else 0.0

            # Get weights - already sorted by Stage-1 EV (position 0 = highest EV)
            w_vec = np.asarray(weights_list[k], float)
            # Show all 9 positions, split into 3 lines of 3 for readability
            line1 = " | ".join([f"Pile {i+1}: {w_vec[i]:.1%}" for i in range(3)])
            line2 = " | ".join([f"Pile {i+1}: {w_vec[i]:.1%}" for i in range(3, 6)])
            line3 = " | ".join([f"Pile {i+1}: {w_vec[i]:.1%}" for i in range(6, 9)])
            weights_str = f"{line1}<br>{line2}<br>{line3}"

            # Hit rates
            ace_hit = int(ace_hits_list[k]) if k < len(ace_hits_list) else 0
            king_hit = int(king_hits_list[k]) if k < len(king_hits_list) else 0
            queen_hit = int(queen_hits_list[k]) if k < len(queen_hits_list) else 0

            hit_rate_str = ""
            if total_rounds and total_rounds > 0:
                ace_pct = (ace_hit / total_rounds) * 100
                hit_rate_str = f"<br><br><b>Hit Rates:</b><br>Ace: {ace_pct:.1f}%"
                if scale_pay == 1:
                    king_pct = (king_hit / total_rounds) * 100
                    queen_pct = (queen_hit / total_rounds) * 100
                    hit_rate_str += f", King: {king_pct:.1f}%, Queen: {queen_pct:.1f}%"

            # Format simulations count
            sim_display = f"{total_rounds:,}" if total_rounds else "N/A"

            # Detailed hover text with all information
            hover_text = (
                f"<b>Signals: n={n_sig}</b><br>"
                f"<b>Simulations:</b> {sim_display}<br>"
                f"<b>Mean Return:</b> {mean_val:.2f}%<br>"
                f"<b>Std Dev:</b> {sd_val:.2f}%<br>"
                f"<b>Sharpe Ratio:</b> {sharpe:.2f}<br>"
                f"<b>Σw²:</b> {cs[idx_k]:.3f}"
                f"{hit_rate_str}<br>"
                f"<br><b>Weights (sorted high→low by Stage-1 EV):</b><br>"
                f"{line1}<br>{line2}<br>{line3}"
            )
            hover_texts.append(hover_text)

            # Store metadata for click events
            custom_data.append({
                'n_sig': n_sig,
                'mean': mean_val,
                'sd': sd_val,
                'sharpe': sharpe,
                'weights': w_vec.tolist(),
                'ace_hits': ace_hit,
                'king_hits': king_hit,
                'queen_hits': queen_hit,
                'total_rounds': total_rounds,
                'scale_pay': scale_pay,
                'ssq': cs[idx_k]
            })

        #Store custom data for this trace
        all_custom_data.extend(custom_data)

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", name=f"n={n_sig}",
            marker=dict(size=16, color=cs, colorscale=[[0, "#2b8cbe"], [1, "#08306b"]],
                        cmin=vmin_global, cmax=vmax_global, showscale=False, line=dict(width=0)),
            text=[str(n_sig)] * len(keep_idx), textposition="middle center", textfont=dict(size=11, color="white"),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_texts, showlegend=False, opacity=ALPHA,
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
        margin=dict(l=10, r=10, t=10, b=50),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Roboto, Arial, sans-serif",
            bordercolor="#2b8cbe",
            align="left",
            namelength=-1,
        ),
    )
    return fig, all_custom_data

# Compute global Σw² range across both datasets for consistent color mapping
def _ssq_extents(fd):
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
        figA, _ = _build_fig(data_A, sd_step_A, y_range, global_vmin, global_vmax)
        st.plotly_chart(figA, use_container_width=True, key="mv_frontier_A")

with colB:
    if data_B is None:
        st.info(f"Frontier data not found: {file_B.name}")
        st.caption("Run frontier.py to generate frontier data.")
    else:
        st.markdown(f"**Fixed:** Ace payoff=20X" + (", Scale param=0.25" if sp_B == 1 else ""))
        figB, _ = _build_fig(data_B, sd_step_B, y_range, global_vmin, global_vmax)
        st.plotly_chart(figB, use_container_width=True, key="mv_frontier_B")

# Shared legend at bottom
if (global_vmin is not None) and (global_vmax is not None):
    # Create legend with title on left, color bar on right
    st.markdown(
        f"""
        <div style='display: flex; align-items: center; gap: 20px; width: 100%; max-width: 900px; margin: 20px auto;'>
            <div style='font-weight: bold; white-space: nowrap;'>Legend: Σw² (portfolio concentration)</div>
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

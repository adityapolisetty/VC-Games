#!/usr/bin/env python3
# card_game_dynamic_v1.py — Two‑stage model
# Stage‑1 invests alpha*BUDGET (signals paid from Stage‑1 only), then reveals
# the second‑highest rank (R2) on piles invested. Stage‑2 invests the remaining
# budget only in piles invested in Stage‑1; per‑pound payoff in Stage‑2 is 0.5x.
# Signal regimes supported: median, top2. Seeding uses round_seed(base, r).

import numpy as np, argparse, pathlib, os, sys, tempfile, json, zlib
from concurrent.futures import ProcessPoolExecutor
from numpy.random import default_rng
from time import perf_counter
from math import comb
from fns import save_npz, canonicalize_params, seed_for_id

"""
Dynamic model overview (inputs, flow, outputs)

Inputs
- CLI args (see main): --seed, --rounds, --max_signals, --procs, and params
  (--signal_cost, --scale_pay, --scale_param, --ace_payout). Stage‑1 allocation
  alpha is handled in dynamic evaluators (later in file).

Flow
- Per parameter configuration, for each signal type in {median, top2} and each
  n in 0..max_signals, we simulate rounds with deterministic round_seed(se ed, r).
- Stage‑1: observe chosen n piles (per permutation), compute weights, pay signal
  costs from Stage‑1 budget and “invest” alpha*BUDGET minus info cost.
- Stage‑2: reveal R2 on piles invested in Stage‑1, update expected payoffs using
  joint posteriors P(Rmax | Stage‑1 bucket, R2), and invest the remaining budget
  only in those piles at 0.5x per‑pound payoff. Payout is realized at the end.

Outputs
- One NPZ per configuration with summary arrays over n for both regimes and the
  three rules (highest, linear, top‑5), plus histogram counts.
"""

# -----------------------
# Parameters / Defaults
# -----------------------
NUM_PILES       = 9
CARDS_PER_PILE  = 5
RANKS           = np.arange(2, 15)          # 2..14, where 14 is Ace
ACE_RANK        = 14
COPIES_PER_RANK = 4                         # one standard deck
BUDGET          = 100.0
# Note: no per-card success probability; payouts are deterministic given ranks

# --------------------------------------------
# Sweep grids (order matters for ID mapping)
# for sc in SIGNAL_COSTS:
#   for sp in SCALE_PAYS:
#     for s in SCALE_PARAMS:
#       for ap in ACE_PAYOUTS:
# ---------------------------------------------

SIGNAL_COSTS   = [0, 3, 5, 7, 9, 11]
SCALE_PAYS     = [0, 1]
SCALE_PARAMS   = [1/3, 1/4, 1/5, 1/6]
ACE_PAYOUTS    = [10, 20, 30, 40]
STAGE1_ALLOC   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


POST_NPZ_DEFAULT = "../output/post_mc.npz"

# Joint posterior (Stage‑1 bucket + R2) from precomp_joint.py
POST_NPZ_JOINT_DEFAULT = "../output_joint/post_joint.npz" 

# Histogram spec for net-return distributions (percent)
HIST_START = -100.0   # inclusive lower bound
HIST_STEP  = 1.0      # bin width in percentage points
HIST_N     = 2000     # number of bins; covers [-100, 1900]

def _load_mc_posteriors(npz_path: str):
    """Load empirical P(Rmax|signal) tables and prior from precompute NPZ.

    Only median and top‑2 regimes are required for the dynamic model. Max/min
    entries, if present, are ignored; empty placeholders are returned for
    compatibility with legacy call‑sites until they are refactored.
    Returns (rmax_median, rmax_top2, rmax_max, rmax_min, prior_rmax, meta_curves)
    where meta_curves contains (pm_x, pm_y, pt2_x, pt2_y, pmax_x, pmax_y, pmin_x, pmin_y).
    """
    p = pathlib.Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"post_npz not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        req = {"rmax_median_keys", "rmax_median_mat", "rmax_top2_keys", "rmax_top2_mat", "prior_rmax"}
        missing = [k for k in req if k not in z.files]
        if missing:
            raise ValueError(f"post_npz missing arrays: {missing}")
        m_keys = np.asarray(z["rmax_median_keys"], int)
        m_mat  = np.asarray(z["rmax_median_mat"], float)
        t_keys = np.asarray(z["rmax_top2_keys"], int)
        t_mat  = np.asarray(z["rmax_top2_mat"], float)
        prior  = np.asarray(z["prior_rmax"], float)

    # Build dicts mapping bucket -> vector
    rmax_median = {int(k): np.array(m_mat[i], float) for i, k in enumerate(m_keys)}
    rmax_top2   = {int(k): np.array(t_mat[i], float) for i, k in enumerate(t_keys)}
    # Placeholders for legacy max/min (not used in dynamic model)
    rmax_max, rmax_min = {}, {}

    # P(Ace | signal) meta curves from last column
    ace_idx = ACE_RANK - 2
    pm_x = m_keys.astype(float)
    pm_y = m_mat[:, ace_idx].astype(float) if m_mat.size else np.zeros((0,), float)
    pt2_x = t_keys.astype(float)
    pt2_y = t_mat[:, ace_idx].astype(float) if t_mat.size else np.zeros((0,), float)
    # Empty meta for max/min
    pmax_x = np.array([], float); pmax_y = np.array([], float)
    pmin_x = np.array([], float); pmin_y = np.array([], float)

    return rmax_median, rmax_top2, rmax_max, rmax_min, prior, (pm_x, pm_y, pt2_x, pt2_y, pmax_x, pmax_y, pmin_x, pmin_y)

# ------------------------------------
# Round-seed helper (32-bit Adler32)
# ------------------------------------

def round_seed(base_seed: int, r: int) -> int:
    """Derive a deterministic 32-bit seed from (base_seed, r) using Adler32."""
    s = f"{int(base_seed)}|round|{int(r)}".encode("utf-8")
    return int(np.uint32(zlib.adler32(s)))

# --------------------------------------------------
# Global-deck board deal (tested OK)
# --------------------------------------------------

def _deal_cards_global_deck(rng):
    """
    Deal one 9-pile board under a global-deck constraint:
    - Specials: exactly 1x Ace, 2x rank-13, 2x rank-12 placed first (no replenishment)
    - Fill remaining slots from ranks 2..11 without replacement (4 copies per rank)

    Input: rng = numpy Generator seeded for this round
    Returns:
      has_ace:  bool[NUM_PILES]
      hands:    list[np.ndarray(5, int)] sorted per pile
      medians:  int[NUM_PILES]
      top2sum:  int[NUM_PILES]
      max_rank: int[NUM_PILES]
    """
    # Special cards are capped and not replenished from leftovers.
    # Exactly 5 specials total: 1x Ace, 2x rank-13, 2x rank-12.
    special_cards = [ACE_RANK, 13, 13, 12, 12]

    # Shuffle special cards
    special_cards = np.array(special_cards, dtype=int)
    if special_cards.size:
        rng.shuffle(special_cards)

    # Initialize hands
    hands = [[] for _ in range(NUM_PILES)]

    # Distribute special cards to random piles
    for card in special_cards:
        available = [i for i in range(NUM_PILES) if len(hands[i]) < CARDS_PER_PILE]
        if not available:
            break  # safety: shouldn't happen with <=5 specials and 9 piles
        pile_idx = int(rng.choice(available))
        hands[pile_idx].append(int(card))

    # Build pool only from non-special ranks 2..11 (exactly 40 cards)
    pool = np.repeat(np.arange(2, 12, dtype=int), COPIES_PER_RANK)

    # Fill remaining slots without replacement from the pool
    for i in range(NUM_PILES):
        need = CARDS_PER_PILE - len(hands[i])
        if need > 0:
            if need > pool.shape[0]:
                # Should not happen given deck arithmetic; guard to avoid crashes.
                need = int(pool.shape[0])
            idx = rng.choice(pool.shape[0], size=need, replace=False)
            draw = pool[idx]
            hands[i].extend(draw.tolist())
            pool = np.delete(pool, idx)

    # Compute statistics and final has_ace from completed hands
    has_ace = np.zeros(NUM_PILES, dtype=bool)
    medians   = np.empty(NUM_PILES, dtype=int)
    top2sum   = np.empty(NUM_PILES, dtype=int)
    max_rank  = np.empty(NUM_PILES, dtype=int)
    min_rank  = np.empty(NUM_PILES, dtype=int)

    for i in range(NUM_PILES):
        arr = np.array(sorted(hands[i]), dtype=int)
        has_ace[i]   = bool(np.any(arr == ACE_RANK))
        medians[i]   = int(arr[CARDS_PER_PILE//2])
        top2sum[i]   = int(arr[-1] + arr[-2])
        max_rank[i]  = int(arr[-1])
        min_rank[i]  = int(arr[0])

    return has_ace, [np.array(sorted(h), int) for h in hands], medians, top2sum, max_rank, min_rank

# -----------------------
# Realized payout
# -----------------------

def _compute_gross_payout(hands, w, scale_pay, scale_param, ace_payout, investable_budget):
    """
    Compute gross payout without subtracting BUDGET.
    Helper for dynamic model where multiple stages are summed before computing net return.
    """
    w = np.array(w, float)
    gross = 0.0

    for i in range(NUM_PILES):
        # Only the highest card in the pile determines the payoff
        max_r = int(np.max(hands[i])) if len(hands[i]) else 2
        if scale_pay == 0:
            # Ace-only: pays if highest card is Ace
            payoff_per_pound = float(ace_payout) if max_r == ACE_RANK else 0.0
        else:
            # Scaled: single highest card pays with scale_param^(14 - highest)
            steps_down = ACE_RANK - max_r
            if steps_down > 11:
                raise ValueError(f"Found max_r={max_r} - shouldnt be possible with global deck")

            payoff_per_pound = float(ace_payout) * (float(scale_param) ** steps_down)
        gross += investable_budget * w[i] * payoff_per_pound

    return gross

def realize_payout(hands, w, scale_pay, scale_param, ace_payout,
                   investable_budget):
    """
    Invest only the remaining budget after paying for signals.
    - investable_budget = max(0, BUDGET - info_cost)
    - gross is computed on that remaining stake.
    - Returns net_return (%) relative to the original BUDGET for comparability.
    """
    gross = _compute_gross_payout(hands, w, scale_pay, scale_param, ace_payout, investable_budget)
    # express net as % of original BUDGET for consistent scaling
    net_return = 100.0 * ((gross - BUDGET) / BUDGET) if BUDGET > 0 else 0.0
    return net_return

# ===============================
# Dynamic two‑stage implementation
# ===============================

def _second_highest_rank(pile: np.ndarray) -> int:
    arr = np.sort(np.asarray(pile, int))
    return int(arr[-2]) if arr.size >= 2 else int(arr[-1])

def _weights_max(sc: np.ndarray) -> np.ndarray:
    p = np.array(sc, float); mx = np.max(p)
    winners = np.where(p == mx)[0]
    w = np.zeros_like(p, float); w[winners] = 1.0 / max(1, winners.size)
    return w

def _weights_linear(sc: np.ndarray) -> np.ndarray:
    p = np.array(sc, float); s = float(np.sum(p))
    return (p / s) if s > 0 else (np.ones_like(p)/len(p))

def _weights_top5(sc: np.ndarray) -> np.ndarray:
    p = np.array(sc, float)
    idx = np.argsort(p)[-5:]
    w = np.zeros_like(p, float); sm = float(np.sum(p[idx]))
    w[idx] = (p[idx] / sm) if sm > 0 else (1.0 / 5.0)
    return w

def _load_joint_posteriors(npz_path: str):
    """Load joint posteriors P(Rmax | Stage‑1 bucket, R2) and marginal P(Rmax | R2)."""
    p = pathlib.Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"joint post_npz not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        req = {"joint_median_keys","joint_median_mat","joint_top2_keys","joint_top2_mat","prior_rmax","r2_marginal_mat"}
        missing = [k for k in req if k not in z.files]
        if missing:
            raise ValueError(f"joint NPZ missing arrays: {missing}")
        jm_keys = np.asarray(z["joint_median_keys"], int); jm_mat = np.asarray(z["joint_median_mat"], float)
        jt_keys = np.asarray(z["joint_top2_keys"],  int); jt_mat = np.asarray(z["joint_top2_mat"],  float)
        prior   = np.asarray(z["prior_rmax"], float)
        r2_marg = np.asarray(z["r2_marginal_mat"], float)
    def _rowmap(keys: np.ndarray):
        return {int(k): int(i) for i, k in enumerate(keys.tolist())}
    return {
        "median": (jm_keys, jm_mat, _rowmap(jm_keys)),
        "top2":   (jt_keys, jt_mat, _rowmap(jt_keys)),
    }, prior, r2_marg

def run_single_round_dynamic(
    rmax_tables, joint_tables, prior_rmax, r2_marginal,
    chosen_idx, signal_type,
    hands, medians, top2sum, max_rank, min_rank,
    scale_pay, scale_param, ace_payout, signal_cost,
    stage1_alloc,
):
    ranks_all = np.arange(2, 15, dtype=int)
    if scale_pay == 0:
        h_vals = np.array([float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
        h_vals2 = np.array([0.5*float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
    else:
        h_vals = np.array([float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)
        h_vals2 = np.array([0.5*float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)

    # Stage‑1 buckets (dynamic model only supports median/top2)
    if signal_type == "median":
        buckets = np.asarray(medians, int)
    elif signal_type == "top2":
        buckets = np.asarray(top2sum, int)
    else:
        raise ValueError(f"Unsupported signal_type for dynamic model: {signal_type}")

    post_table = rmax_tables[signal_type]
    prior_vec = np.asarray(prior_rmax, float)
    chosen_set = set(int(x) for x in np.asarray(chosen_idx, int))
    scores1 = np.zeros(NUM_PILES, float)
    for i in range(NUM_PILES):
        vec = np.asarray(post_table.get(int(buckets[i]), prior_vec), float) if (i in chosen_set) else prior_vec
        scores1[i] = float(np.dot(h_vals, vec))

    # Weights
    w1_max  = _weights_max(scores1)
    w1_lin  = _weights_linear(scores1)
    w1_top5 = _weights_top5(scores1)

    # Budgets: signals only from Budget1
    alpha = min(1.0, max(0.0, float(stage1_alloc)))
    budget1 = alpha * float(BUDGET)
    info_cost = len(chosen_idx) * float(signal_cost)
    investable1 = max(0.0, budget1 - info_cost)
    budget2 = max(0.0, float(BUDGET) - budget1)

    # Stage‑2 scores on invested piles
    # Use joint P(Rmax | bucket, R2) for observed piles, marginal P(Rmax | R2) for unobserved
    keys, mat3d, rowmap = joint_tables[signal_type]
    R2 = np.array([_second_highest_rank(h) for h in hands], int)
    r2_marg = np.asarray(r2_marginal, float)
    def scores2_from(w1):
        sc = np.zeros(NUM_PILES, float)
        if investable1 <= 0:
            return sc
        idxs = np.where(w1 > 0)[0]
        for i in idxs:
            r2k = int(R2[i]) - 2
            if i in chosen_set:
                # Invested AND observed: use joint posterior P(Rmax | bucket, R2)
                b = int(buckets[i])
                if (b in rowmap) and (0 <= r2k < 13):
                    vec = np.asarray(mat3d[rowmap[b], r2k, :], float)
                else:
                    vec = prior_vec
            else:
                # Invested but NOT observed: use marginal posterior P(Rmax | R2)
                if 0 <= r2k < 13:
                    vec = np.asarray(r2_marg[r2k, :], float)
                else:
                    vec = prior_vec
            sc[i] = float(np.dot(h_vals2, vec))
        return sc

    sc2_max  = scores2_from(w1_max)
    sc2_lin  = scores2_from(w1_lin)
    sc2_top5 = scores2_from(w1_top5)

    w2_max  = _weights_max(sc2_max)
    w2_lin  = _weights_linear(sc2_lin)
    w2_top5 = _weights_top5(sc2_top5)

    # Realize at end: Stage-1 + Stage-2 (compute gross first, then subtract BUDGET once)
    gross_max = _compute_gross_payout(hands, w1_max,  scale_pay, scale_param, ace_payout,      investable1) \
              + _compute_gross_payout(hands, w2_max,  scale_pay, scale_param, 0.5*ace_payout, budget2)
    gross_lin = _compute_gross_payout(hands, w1_lin,  scale_pay, scale_param, ace_payout,      investable1) \
              + _compute_gross_payout(hands, w2_lin,  scale_pay, scale_param, 0.5*ace_payout, budget2)
    gross_top = _compute_gross_payout(hands, w1_top5, scale_pay, scale_param, ace_payout,      investable1) \
              + _compute_gross_payout(hands, w2_top5, scale_pay, scale_param, 0.5*ace_payout, budget2)

    # Subtract BUDGET only once to get net return (%)
    n_max = 100.0 * ((gross_max - BUDGET) / BUDGET) if BUDGET > 0 else 0.0
    n_lin = 100.0 * ((gross_lin - BUDGET) / BUDGET) if BUDGET > 0 else 0.0
    n_top = 100.0 * ((gross_top - BUDGET) / BUDGET) if BUDGET > 0 else 0.0

    return dict(net_return_max=float(n_max), net_return_linear=float(n_lin), net_return_top5=float(n_top))

    
# -----------------------
# Dynamic worker (two-stage)
# -----------------------
def _worker_chunk_dynamic(base_seed, round_start, rounds_chunk, signal_type, n_sig,
                          rmax_tables, joint_tables, prior_rmax, r2_marginal, params, stage1_alloc):
    """
    Evaluate a contiguous chunk of rounds for a fixed (signal_type, n_sig) using
    the dynamic two-stage model with Stage-1 allocation stage1_alloc.
    Returns three 1-D arrays (length = rounds_chunk): max / linear / top-5 net returns.
    """
    nr_max = np.empty(rounds_chunk, float)
    nr_lin = np.empty(rounds_chunk, float)
    nr_top5 = np.empty(rounds_chunk, float)
    for i in range(rounds_chunk):
        r = int(round_start) + int(i)
        rng = default_rng(round_seed(base_seed, r))
        _, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
        pi = rng.permutation(NUM_PILES)
        chosen_idx = pi[:n_sig]
        out = run_single_round_dynamic(
            rmax_tables=rmax_tables,
            joint_tables=joint_tables,
            prior_rmax=prior_rmax,
            r2_marginal=r2_marginal,
            chosen_idx=chosen_idx, signal_type=signal_type,
            hands=hands, medians=medians, top2sum=top2sum, max_rank=max_rank, min_rank=min_rank,
            scale_pay=params["scale_pay"], scale_param=params["scale_param"],
            ace_payout=params["ace_payout"], signal_cost=params["signal_cost"],
            stage1_alloc=stage1_alloc,
        )
        nr_max[i]  = out["net_return_max"]
        nr_lin[i]  = out["net_return_linear"]
        nr_top5[i] = out["net_return_top5"]
    return nr_max, nr_lin, nr_top5

# -----------------------
# Main simulate (dynamic, two-stage)
# -----------------------
def simulate_experiment_dynamic(seed_int, rounds, max_signals, procs, params, stage1_alloc=0.5):
    """
    Dynamic two-stage simulation for ONE parameter configuration.

    - Stage-1 invests alpha*BUDGET (alpha=stage1_alloc), paying for signals from Stage-1 only.
    - Stage-2 invests remaining budget only on Stage-1 invested piles at 0.5x payoff, using
      joint posteriors P(Rmax | Stage-1 bucket, R2).
    - Deterministic board/permutation per round via round_seed(seed_int, r).
    """

    # Load P(Rmax|signal) for median/top2 and prior
    rmax_median, rmax_top2, _rmax_max_unused, _rmax_min_unused, prior_mc, (pm_x, pm_y, pt2_x, pt2_y, pmax_x, pmax_y, pmin_x, pmin_y) = _load_mc_posteriors(POST_NPZ_DEFAULT)

    # Load joint posteriors, marginal posteriors, and prior
    joint_tables, prior_joint, r2_marginal = _load_joint_posteriors(POST_NPZ_JOINT_DEFAULT)
    prior_rmax = prior_joint if isinstance(prior_joint, np.ndarray) else prior_mc

    # Tables used by dynamic model
    rmax_tables = {"median": rmax_median, "top2": rmax_top2}
    signal_types = ["median", "top2"]

    # Output containers (dynamic-only regimes)
    dist    = {st: {} for st in signal_types}
    summary = {st: {} for st in signal_types}

    hists = {
        st: {
            'max':    np.zeros((int(max_signals)+1, HIST_N), dtype=np.uint32),
            'linear': np.zeros((int(max_signals)+1, HIST_N), dtype=np.uint32),
            'top5':   np.zeros((int(max_signals)+1, HIST_N), dtype=np.uint32),
        } for st in signal_types
    }

    def _hist_counts(arr: np.ndarray) -> np.ndarray:
        v = np.asarray(arr, float)
        idx = np.floor((v - HIST_START) / HIST_STEP).astype(int)
        mask = (idx >= 0) & (idx < HIST_N)
        if not np.any(mask):
            return np.zeros((HIST_N,), dtype=np.uint32)
        cnt = np.bincount(idx[mask], minlength=HIST_N)
        return cnt.astype(np.uint32, copy=False)

    total_units = len(signal_types) * (int(max_signals) + 1) * int(rounds)
    processed = 0
    step_overall = max(1, total_units // 100)

    def _print_bar(cur, total):
        cur = int(cur); total = max(1, int(total))
        pct = int(100 * cur / total)
        barw = 30; filled = int(barw * cur / total)
        bar = "=" * filled + "-" * (barw - filled)
        sys.stdout.write(f"\rProgress {cur}/{total} {pct:3d}% [{bar}]"); sys.stdout.flush()

    for st in signal_types:
        for n_sig in range(int(max_signals) + 1):
            n_max_arr  = np.empty(int(rounds), float)
            n_lin_arr  = np.empty(int(rounds), float)
            n_top5_arr = np.empty(int(rounds), float)

            if procs and int(procs) > 1 and int(rounds) > 1:
                W = int(procs); base = int(rounds) // W; rem = int(rounds) % W
                chunk_sizes = [base + (1 if i < rem else 0) for i in range(W)]
                starts = []
                s = 0
                for c in chunk_sizes:
                    if c > 0:
                        starts.append((s, c))
                        s += c
                with ProcessPoolExecutor(max_workers=len(starts)) as ex:
                    futures = [
                        ex.submit(
                            _worker_chunk_dynamic,
                            int(seed_int), int(start), int(sz), st, int(n_sig),
                            rmax_tables, joint_tables, prior_rmax, r2_marginal, params, float(stage1_alloc)
                        )
                        for (start, sz) in starts
                    ]
                    for (start, sz), fut in zip(starts, futures):
                        n_m, n_l, n_s = fut.result()
                        n_max_arr[start:start+sz]  = n_m
                        n_lin_arr[start:start+sz]  = n_l
                        n_top5_arr[start:start+sz] = n_s
                        processed += int(sz)
                        if (processed % step_overall == 0) or (processed >= total_units):
                            _print_bar(processed, total_units)
            else:
                for r in range(int(rounds)):
                    rng = default_rng(round_seed(int(seed_int), int(r)))
                    _, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
                    pi = rng.permutation(NUM_PILES)
                    chosen_idx = pi[:n_sig]
                    out = run_single_round_dynamic(
                        rmax_tables=rmax_tables,
                        joint_tables=joint_tables,
                        prior_rmax=prior_rmax,
                        r2_marginal=r2_marginal,
                        chosen_idx=chosen_idx, signal_type=st,
                        hands=hands, medians=medians, top2sum=top2sum, max_rank=max_rank, min_rank=min_rank,
                        scale_pay=params["scale_pay"], scale_param=params["scale_param"],
                        ace_payout=params["ace_payout"], signal_cost=params["signal_cost"],
                        stage1_alloc=stage1_alloc,
                    )
                    n_max_arr[r]  = out["net_return_max"]
                    n_lin_arr[r]  = out["net_return_linear"]
                    n_top5_arr[r] = out["net_return_top5"]
                    processed += 1
                    if (processed % step_overall == 0) or (processed >= total_units):
                        _print_bar(processed, total_units)

            dist[st][n_sig] = dict(
                net_return_max=n_max_arr, net_return_linear=n_lin_arr, net_return_top5=n_top5_arr
            )
            summary[st][n_sig] = dict(
                mean_net_return_max=float(np.mean(n_max_arr)), mean_net_return_linear=float(np.mean(n_lin_arr)), mean_net_return_top5=float(np.mean(n_top5_arr)),
                sd_net_return_max=float(np.std(n_max_arr, ddof=1)), sd_net_return_linear=float(np.std(n_lin_arr, ddof=1)), sd_net_return_top5=float(np.std(n_top5_arr, ddof=1)),
                med_net_return_max=float(np.median(n_max_arr)), med_net_return_linear=float(np.median(n_lin_arr)), med_net_return_top5=float(np.median(n_top5_arr))
            )
            hists[st]['max'][int(n_sig),   :] = _hist_counts(n_max_arr)
            hists[st]['linear'][int(n_sig),:] = _hist_counts(n_lin_arr)
            hists[st]['top5'][int(n_sig),  :] = _hist_counts(n_top5_arr)

    try:
        print()
    except Exception:
        pass

    meta = dict(
        mode="dynamic", params=dict(params), stage1_alloc=float(stage1_alloc),
        post_median_x=pm_x, post_median_y=pm_y,
        post_top2_x=pt2_x, post_top2_y=pt2_y,
        hist_start=float(HIST_START), hist_step=float(HIST_STEP), hist_n=int(HIST_N)
    )
    return dist, summary, meta, hists


# -----------------------
# Sweep (multi-combo)
# -----------------------
def run_sweep(base_seed, rounds, max_signals, procs_inner, out_dir,
              sweep_index=None, sweep_stride=1, skip_existing=False):
    """
    Build full parameter grid including STAGE1_ALLOC, optionally slice it for array jobs,
    run each combo, save NPZ to out_dir, and write out_dir/index.json.

    If skip_existing=True, skips parameter combinations that already have output files.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grids
    SIGNAL_COSTS = [0, 3, 5, 7, 9, 11]
    SCALE_PARAMS = [1/3, 1/4, 1/5, 1/6]
    SCALE_PAYS   = [0, 1]
    ACE_PAYOUTS  = [10, 20, 30, 40]
    STAGE1_ALLOC = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Build all combos (sweep alpha from STAGE1_ALLOC grid)
    combos = []
    seen_keys = set()  # Track unique (key_id, alpha) pairs to avoid redundant configs
    for alpha in STAGE1_ALLOC:
        for sc in SIGNAL_COSTS:
            for sp in SCALE_PAYS:
                for s in SCALE_PARAMS:
                    for ap in ACE_PAYOUTS:
                        raw = dict(
                            signal_cost=sc,
                            scale_pay=sp,
                            scale_param=s,
                            ace_payout=ap,
                        )
                        norm, key_tuple, key_id = canonicalize_params(raw)
                        # Same base seed across combos; round_seed(base_seed, r) keeps boards identical across grid
                        seed_i = int(base_seed)
                        a_tag = f"a{int(round(float(alpha)*10)):02d}"
                        outfile = out_dir / f"{key_id}_{a_tag}.npz"

                        # Skip redundant configs (e.g., scale_pay=0 makes scale_param irrelevant)
                        combo_key = (key_id, alpha)
                        if combo_key not in seen_keys:
                            seen_keys.add(combo_key)
                            combos.append((raw, norm, key_tuple, key_id, seed_i, outfile, alpha))

    total_unique = len(combos)
    print(f"Generated {total_unique} unique parameter combinations (after deduplication)")

    # slicing param grid for array jobs
    if sweep_index is not None:
        if sweep_stride <= 0:
            raise ValueError("sweep_stride must be >= 1")
        if not (0 <= sweep_index < sweep_stride):
            raise ValueError("sweep_index must be in [0, sweep_stride)")
        combos = [item for i, item in enumerate(combos) if (i % sweep_stride) == sweep_index]

    # Skip combinations that already have output files (if requested)
    if skip_existing:
        total_combos = len(combos)
        print(f"Checking {total_combos} combinations for existing files in {out_dir}...")
        combos_filtered = []
        skipped_files = []
        for item in combos:
            if item[5].exists():
                skipped_files.append(str(item[5].name))
            else:
                combos_filtered.append(item)
        combos = combos_filtered
        skipped_count = total_combos - len(combos)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} already-computed combinations (found existing output files)")
            if skipped_count <= 5:
                for fname in skipped_files[:5]:
                    print(f"  - {fname}")
        else:
            print(f"No existing files found; will compute all {total_combos} combinations")

    index = {}

    def _run_one(item):
        raw, norm, key_tuple, key_id, seed_i, outfile, alpha = item
        dist, summary, meta, hists = simulate_experiment_dynamic(
            seed_int=seed_i,
            rounds=rounds,
            max_signals=max_signals,
            procs=procs_inner,
            params=norm,
            stage1_alloc=float(alpha),
        )
        save_npz(
            outfile,
            argparse.Namespace(rounds=rounds, max_signals=max_signals),
            dist, summary, meta, norm, raw, key_tuple, key_id, hists
        )
        return key_id, str(outfile)

    # Run combos serially (inter-param parallelism removed)
    for item in combos:
        kid, path = _run_one(item)
        index[kid] = path

    # Write index
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    return index

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",        type=int, required=True)
    ap.add_argument("--rounds",      type=int, required=True)
    ap.add_argument("--max_signals", type=int, default=9)
    ap.add_argument("--procs",       type=int, default=1, help="intra-run parallelism")
    ap.add_argument("--stage1_alloc", type=float, default=0.5, help="Stage-1 allocation alpha in [0,1]")

    # Single-run params (optional when not sweeping)
    ap.add_argument("--signal_cost", type=float, default=7.0)
    ap.add_argument("--scale_pay",   type=int,   default=0, choices=[0, 1])
    ap.add_argument("--scale_param", type=float, default=0.25)
    ap.add_argument("--ace_payout",  type=float, default=20.0)

    # Output
    ap.add_argument("--out",        type=str, default=None,
                    help="single-run output path; default=output_joint/<canonical_id>_aXX.npz")

    # Sweep
    ap.add_argument("--sweep",       action="store_true")
    ap.add_argument("--sweep_out",   type=str, default="output_joint",
                    help="directory for sweep outputs (files saved directly in this dir)")
    ap.add_argument("--sweep_index", type=int, default=None,
                    help="process only combos with idx %% sweep_stride == sweep_index")
    ap.add_argument("--sweep_stride", type=int, default=1,
                    help="stride used with --sweep_index")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip parameter combinations that already have output files")

    # empirical posteriors NPZ is required and read from POST_NPZ_DEFAULT; run precompute script to generate it

    args = ap.parse_args()

    if args.sweep:
        print(f"Starting sweep with skip_existing={args.skip_existing}")
        t0 = perf_counter()
        index = run_sweep(
            base_seed=int(args.seed),
            rounds=int(args.rounds),
            max_signals=int(args.max_signals),
            procs_inner=int(args.procs),
            out_dir=args.sweep_out,
            sweep_index=args.sweep_index,
            sweep_stride=args.sweep_stride,
            skip_existing=bool(args.skip_existing),
        )
        t1 = perf_counter()
        print(f"wrote {len(index)} files to {args.sweep_out} in {t1 - t0:.2f}s")
        return

    # single run
    raw_params = dict(
        signal_cost=float(args.signal_cost),
        scale_pay=int(args.scale_pay),
        scale_param=float(args.scale_param),
        ace_payout=float(args.ace_payout),
    )
    norm_params, key_tuple, key_id = canonicalize_params(raw_params)
    if args.out:
        out_path = pathlib.Path(args.out)
    else:
        a_tag = f"a{int(round(float(args.stage1_alloc)*10)):02d}"
        out_path = pathlib.Path("output_joint") / f"{key_id}_{a_tag}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)


    t0 = perf_counter()
    dist, summary, meta, hists = simulate_experiment_dynamic(
        seed_int=int(args.seed),
        rounds=int(args.rounds),
        max_signals=int(args.max_signals),
        procs=int(args.procs),
        params=norm_params,
        stage1_alloc=float(args.stage1_alloc),
    )
    save_npz(out_path, args, dist, summary, meta, norm_params, raw_params, key_tuple, key_id, hists)
    t1 = perf_counter()
    print(f"wrote {out_path} | total={t1 - t0:.2f}s")

if __name__ == "__main__":
    main()

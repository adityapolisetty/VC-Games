#!/usr/bin/env python3
# card_game_v7.py
# Single-stage model: duplicates allowed within a pile; global 52-deck constraint.
# Uses P(Rmax | signal) posteriors and highest-card-only payout when scale_pay=1.
# Stable canonical IDs; sweep writes one NPZ per parameter combo + index.json.

import numpy as np, argparse, pathlib, os, sys, tempfile, json, zlib
from concurrent.futures import ProcessPoolExecutor
from numpy.random import default_rng
from time import perf_counter
from math import comb
from fns import save_npz, canonicalize_params, seed_for_id

"""
Module overview (inputs, flow, outputs)

Inputs
- CLI args (see main): --seed, --rounds, --max_signals, --procs, and single-run params
  (--signal_cost, --scale_pay, --scale_param, --ace_payout). For sweeps, use
  --sweep/--sweep_out and slice the grid with --sweep_index/--sweep_stride.

Flow
- One parameter configuration per call to simulate_experiment.
- Parallelize across rounds only. Each round r uses round_seed(seed, r) to get
  a deterministic RNG. We deal one board and one pile order per round, then build
  nested signal sets S[n] = first n piles. No RNG is used after that point.
- For each signal type (median/top2) and each n, we compute net returns for three
  weight rules (highest, linear, top-5) and aggregate mean/sd over rounds.

Outputs
- One NPZ per configuration containing summary arrays for both signal types and
  all n (mean/sd of net returns for the three rules), plus P(Ace|signal) curves
  derived from P(Rmax|signal). In sweep mode, run_sweep also writes index.json.
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

# Canonical fill-ins when a switch is OFF
CANON_SCALE_PARAM_DEFAULT = 0   # used when scale_pay == 0

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



POST_NPZ_DEFAULT = "../output/post_mc.npz"

# Histogram spec for net-return distributions (percent)
HIST_START = -100.0   # inclusive lower bound
HIST_STEP  = 1.0      # bin width in percentage points
HIST_N     = 2000     # number of bins; covers [-100, 1900]

def _load_mc_posteriors(npz_path: str):
    """Load empirical P(Rmax|signal) tables and prior from precompute NPZ.

    Returns (rmax_median, rmax_top2, rmax_max, rmax_min, prior_rmax, meta_curves)
    where meta_curves contains (pm_x, pm_y, pt2_x, pt2_y, pmax_x, pmax_y, pmin_x, pmin_y).
    """
    p = pathlib.Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"post_npz not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        req = {
            "rmax_median_keys", "rmax_median_mat",
            "rmax_top2_keys",  "rmax_top2_mat",
            "rmax_max_keys",   "rmax_max_mat",
            "rmax_min_keys",   "rmax_min_mat",
            "prior_rmax",
        }
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
    # load max/min mats for meta curves and scoring
    # re-open NPZ to get arrays (already loaded above but keep structure simple)
    with np.load(p, allow_pickle=False) as z:
        x_keys = np.asarray(z["rmax_max_keys"], int); x_mat = np.asarray(z["rmax_max_mat"], float)
        n_keys = np.asarray(z["rmax_min_keys"], int); n_mat = np.asarray(z["rmax_min_mat"], float)
    rmax_max = {int(k): np.array(x_mat[i], float) for i, k in enumerate(x_keys)}
    rmax_min = {int(k): np.array(n_mat[i], float) for i, k in enumerate(n_keys)}

    # P(Ace | signal) meta curves from last column
    ace_idx = ACE_RANK - 2
    pm_x = m_keys.astype(float)
    pm_y = m_mat[:, ace_idx].astype(float) if m_mat.size else np.zeros((0,), float)
    pt2_x = t_keys.astype(float)
    pt2_y = t_mat[:, ace_idx].astype(float) if t_mat.size else np.zeros((0,), float)
    pmax_x = x_keys.astype(float)
    pmax_y = x_mat[:, ace_idx].astype(float) if x_mat.size else np.zeros((0,), float)
    pmin_x = n_keys.astype(float)
    pmin_y = n_mat[:, ace_idx].astype(float) if n_mat.size else np.zeros((0,), float)

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

def realize_payout(hands, w, scale_pay, scale_param, ace_payout,
                   investable_budget):
    """
    Invest only the remaining budget after paying for signals.
    - investable_budget = max(0, BUDGET - info_cost)
    - gross is computed on that remaining stake.
    - Returns net_return (%) relative to the original BUDGET for comparability.
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

    # express net as % of original BUDGET for consistent scaling
    net_return = 100.0 * ((gross - BUDGET) / BUDGET) if BUDGET > 0 else 0.0
    return net_return

# -----------------------
# One round
# -----------------------
def run_single_round(rmax_median, rmax_top2, rmax_max, rmax_min, prior_rmax,
                     chosen_idx, signal_type,
                     hands, medians, top2sum, max_rank, min_rank,
                     scale_pay, scale_param, ace_payout, signal_cost):
    """
    Compute net returns for ONE round, for ONE signal type, and ONE n.

    Inputs
    - rmax_median / rmax_top2: dict mapping signal value -> P(Rmax=r) vector (r=2..14 -> index r-2)
    - prior_rmax: P(Rmax=r) vector used for unobserved piles
    - chosen_idx: 1-D indices of piles we observe this round (prefix of the round's permutation)
    - signal_type: "median" or "top2" or "min" or "max" (which posterior table to use)
    - hands, medians, top2sum: per-pile board data for this round (no RNG here)
    - scale_pay, scale_param, ace_payout, signal_cost: configuration parameters

    What it does
    - Builds an expected-payoff score per pile using P(Rmax | observation) for observed piles,
      and the prior for unobserved piles.
    - Converts scores into three weight rules: highest, linear, top-5.
    - Charges info cost len(chosen_idx) * signal_cost, invests the remainder according to weights,
      and returns net returns (%) for each rule.

    No RNG is used inside this function. It is a pure computation given the board and chosen_idx.
    """

    # Build expected-payoff scores per pile
    ranks_all = np.arange(2, 15, dtype=int)
    if scale_pay == 0:
        h_vals = np.array([float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
    else:
        h_vals = np.array([float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)

    table_map = {"median": rmax_median, "top2": rmax_top2, "max": rmax_max, "min": rmax_min}
    post_table = table_map[signal_type]

    scores = np.zeros(NUM_PILES, float)
    chosen_set = set(int(x) for x in np.asarray(chosen_idx, int))
    for i in range(NUM_PILES):
        if i in chosen_set:
            if signal_type == "median":
                obs = int(medians[i])
            elif signal_type == "top2":
                obs = int(top2sum[i])
            elif signal_type == "max":
                obs = int(max_rank[i])
            else:
                obs = int(min_rank[i])
            rmax_vec = post_table[obs]
        else:
            rmax_vec = prior_rmax
        vec = np.asarray(rmax_vec, float)
        if not np.all(np.isfinite(vec)):
            raise ValueError(
                f"Non-finite P(Rmax|obs) for pile={i}, observed={i in chosen_set}, "
                f"signal_type={signal_type}, obs={obs if i in chosen_set else 'prior'}, vec={vec}"
            )
        score_i = float(np.dot(h_vals, vec))
        if not np.isfinite(score_i):
            raise ValueError(
                f"Non-finite score for pile={i}, observed={i in chosen_set}, "
                f"signal_type={signal_type}, obs={obs if i in chosen_set else 'prior'}, "
                f"h_vals={h_vals}, vec={vec}"
            )
        scores[i] = score_i

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            f"Non-finite scores array for signal_type={signal_type}, chosen_idx={list(map(int, chosen_idx))}, scores={scores}"
        )

    def weights_max(sc):
        p = np.array(sc, float); mx = np.max(p)
        highest_pos_prob = np.where(p == mx)[0]
        w = np.zeros_like(p, float); w[highest_pos_prob] = 1.0 / len(highest_pos_prob)
        return w

    def weights_linear(sc):
        p = np.array(sc, float); s = float(np.sum(p))
        return (p / s) if s > 0 else (np.ones_like(p)/len(p))

    def weights_top5(sc):
        p = np.array(sc, float)
        top5_idx = np.argsort(p)[-5:]
        w = np.zeros_like(p, float)
        top5_values = p[top5_idx]
        top5_sum = float(np.sum(top5_values))
        if top5_sum > 0:
            w[top5_idx] = top5_values / top5_sum
        else:
            w[top5_idx] = 1.0 / 5.0
        return w

    # Sanity: ensure we have at least one winner before computing highest-weights
    mx_tmp = np.max(scores)
    winners_tmp = np.where(scores == mx_tmp)[0]
    if winners_tmp.size == 0:
        raise ValueError(
            f"No winners in highest-weights: signal_type={signal_type}, "
            f"chosen_idx={list(map(int, chosen_idx))}, mx={mx_tmp}, scores={scores}"
        )

    w_max = weights_max(scores)
    w_lin = weights_linear(scores)
    w_top5 = weights_top5(scores)

    # pay for information first; invest the remainder only
    info_cost = len(chosen_idx) * float(signal_cost)
    investable_budget = max(0.0, BUDGET - info_cost)

    n_max = realize_payout(hands, w_max, scale_pay, scale_param, ace_payout, investable_budget)
    n_lin = realize_payout(hands, w_lin, scale_pay, scale_param, ace_payout, investable_budget)
    n_top5 = realize_payout(hands, w_top5, scale_pay, scale_param, ace_payout, investable_budget)

    return dict(
        net_return_max=float(n_max),
        net_return_linear=float(n_lin),
        net_return_top5=float(n_top5),
    )

# -----------------------
# Worker
# -----------------------
def _worker_chunk(base_seed, round_start, rounds_chunk, signal_type, n_sig, rmax_median, rmax_top2, rmax_max, rmax_min, prior_rmax, params):
    """
    Evaluate a contiguous chunk of rounds for a fixed (signal_type, n_sig).

    For each round r in [round_start, round_start + rounds_chunk):
    - Seed RNG deterministically from (base_seed, r)
    - Deal one board and compute per-pile signals
    - Draw a single permutation of piles and build chosen_idx = prefix of length n_sig
    - Call run_single_round (pure) to get net returns for the three weight rules

    Returns three 1-D arrays (length = rounds_chunk): max / linear / top-5 net returns.
    """
    nr_max = np.empty(rounds_chunk, float)
    nr_lin = np.empty(rounds_chunk, float)
    nr_top5 = np.empty(rounds_chunk, float)
    for i in range(rounds_chunk):
        r = int(round_start) + int(i)
        rng = default_rng(round_seed(base_seed, r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
        pi = rng.permutation(NUM_PILES)
        chosen_idx = pi[:n_sig]
        out = run_single_round(
            rmax_median=rmax_median, rmax_top2=rmax_top2, rmax_max=rmax_max, rmax_min=rmax_min, prior_rmax=prior_rmax,
            chosen_idx=chosen_idx, signal_type=signal_type,
            hands=hands, medians=medians, top2sum=top2sum, max_rank=max_rank, min_rank=min_rank,
            scale_pay=params["scale_pay"], scale_param=params["scale_param"],
            ace_payout=params["ace_payout"], signal_cost=params["signal_cost"],
        )
        nr_max[i] = out["net_return_max"]; nr_lin[i] = out["net_return_linear"]; nr_top5[i] = out["net_return_top5"]
    return nr_max, nr_lin, nr_top5

# -----------------------
# Main simulate (single configuration)
# -----------------------
def simulate_experiment(seed_int, rounds, max_signals, procs, params):
    """
    Orchestrate the full simulation for ONE parameter configuration.

    High-level flow
    - Precompute signal->P(Rmax) tables once (for median and top-2) and the prior P(Rmax).
    - For each signal_type in {median, top2} and each n in 0..max_signals:
        * Allocate arrays of length = rounds for the three weight rules.
        * Parallelize across rounds only (if procs>1): split round indices into contiguous
          chunks and launch _worker_chunk for each chunk. Otherwise, run serially.
        * Assemble results in round order so outputs are invariant to scheduling.
    - Compute mean and standard deviation over rounds and package into the summary.
    - Derive P(Ace|signal) meta curves from the Rmax tables and include in NPZ metadata.

    Determinism
    - One RNG seed per round: round_seed(seed_int, r)
    - No RNG inside scoring/payout; only board deal and the single pile permutation use RNG.
    """

    # Empirical P(Rmax|signal) and prior are mandatory; load from default path
    rmax_median, rmax_top2, rmax_max, rmax_min, prior_rmax, (pm_x, pm_y, pt2_x, pt2_y, pmax_x, pmax_y, pmin_x, pmin_y) = _load_mc_posteriors(POST_NPZ_DEFAULT)

    signal_types = ["median", "top2", "max", "min"]

    dist    = {st: {} for st in signal_types}
    summary = {st: {} for st in signal_types}
    # Fixed-bin histogram containers for net-return distributions
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

    total_units = len(signal_types) * (max_signals + 1) * int(rounds)
    processed = 0
    step_overall = max(1, total_units // 100)

    def _print_bar(cur, total):
        cur = int(cur); total = max(1, int(total))
        pct = int(100 * cur / total)
        barw = 30; filled = int(barw * cur / total)
        bar = "=" * filled + "-" * (barw - filled)
        sys.stdout.write(f"\rProgress {cur}/{total} {pct:3d}% [{bar}]"); sys.stdout.flush()

    for st in signal_types:
        for n_sig in range(max_signals+1):

            # Preallocate arrays in round order
            n_max_arr = np.empty(int(rounds), float)
            n_lin_arr = np.empty(int(rounds), float)
            n_top5_arr = np.empty(int(rounds), float)
            if procs and int(procs) > 1 and rounds > 1:
                W = int(procs); base = rounds // W; rem = rounds % W
                chunk_sizes = [base + (1 if i < rem else 0) for i in range(W)]
                starts = []
                s = 0
                for c in chunk_sizes:
                    if c > 0:
                        starts.append((s, c))
                        s += c
                with ProcessPoolExecutor(max_workers=len(starts)) as ex:
                    futures = [ex.submit(_worker_chunk, int(seed_int), int(start), int(sz), st, int(n_sig), rmax_median, rmax_top2, rmax_max, rmax_min, prior_rmax, params)
                               for (start, sz) in starts]
                    for (start, sz), fut in zip(starts, futures):
                        n_m, n_l, n_s = fut.result()
                        n_max_arr[start:start+sz] = n_m
                        n_lin_arr[start:start+sz] = n_l
                        n_top5_arr[start:start+sz] = n_s
                        processed += int(sz)
                        if (processed % step_overall == 0) or (processed >= total_units):
                            _print_bar(processed, total_units)
            else:
                for r in range(int(rounds)):
                    rng_rounds = default_rng(round_seed(seed_int, r))
                    has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng_rounds)
                    pi = rng_rounds.permutation(NUM_PILES)
                    chosen_idx = pi[:n_sig]
                    out = run_single_round(
                        rmax_median=rmax_median, rmax_top2=rmax_top2, rmax_max=rmax_max, rmax_min=rmax_min, prior_rmax=prior_rmax,
                        chosen_idx=chosen_idx, signal_type=st,
                        hands=hands, medians=medians, top2sum=top2sum, max_rank=max_rank, min_rank=min_rank,
                        scale_pay=params["scale_pay"], scale_param=params["scale_param"],
                        ace_payout=params["ace_payout"], signal_cost=params["signal_cost"],
                    )
                    n_max_arr[r] = out["net_return_max"]; n_lin_arr[r] = out["net_return_linear"]; n_top5_arr[r] = out["net_return_top5"]
                    processed += 1
                    if (processed % step_overall == 0) or (processed >= total_units):
                        _print_bar(processed, total_units)

            # Validate sample size for standard deviation calculation
            n_samples = len(n_max_arr)
            if n_samples < 2:
                raise ValueError(
                    f"Cannot compute standard deviation with n={n_samples} samples. "
                    f"Signal type '{st}', n_signals={n_sig}. "
                    f"Increase --rounds or reduce --procs to ensure at least 2 samples per worker."
                )

            dist[st][n_sig] = dict(
                net_return_max=n_max_arr, net_return_linear=n_lin_arr, net_return_top5=n_top5_arr
            )
            summary[st][n_sig] = dict(
                mean_net_return_max=float(np.mean(n_max_arr)), mean_net_return_linear=float(np.mean(n_lin_arr)), mean_net_return_top5=float(np.mean(n_top5_arr)),
                sd_net_return_max=float(np.std(n_max_arr, ddof=1)), sd_net_return_linear=float(np.std(n_lin_arr, ddof=1)), sd_net_return_top5=float(np.std(n_top5_arr, ddof=1)),
                med_net_return_max=float(np.median(n_max_arr)), med_net_return_linear=float(np.median(n_lin_arr)), med_net_return_top5=float(np.median(n_top5_arr))
            )

            # Histogram counts for this (signal_type, n_sig)
            hists[st]['max'][int(n_sig),   :] = _hist_counts(n_max_arr)
            hists[st]['linear'][int(n_sig),:] = _hist_counts(n_lin_arr)
            hists[st]['top5'][int(n_sig),  :] = _hist_counts(n_top5_arr)
    try: print()
    except Exception: pass

    meta = dict(mode="dup", params=dict(params),
                post_median_x=pm_x, post_median_y=pm_y,
                post_top2_x=pt2_x, post_top2_y=pt2_y,
                post_max_x=pmax_x, post_max_y=pmax_y,
                post_min_x=pmin_x, post_min_y=pmin_y,
                hist_start=float(HIST_START), hist_step=float(HIST_STEP), hist_n=int(HIST_N))
    return dist, summary, meta, hists


# -----------------------
# Sweep (multi-combo)
# -----------------------
def run_sweep(base_seed, rounds, max_signals, procs_inner, out_dir,
              sweep_index=None, sweep_stride=1):
    """
    Build full parameter grid, optionally slice it for array jobs,
    run each combo, save NPZ to out_dir, and write out_dir/index.json.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grids
    SIGNAL_COSTS = [0, 3, 5, 7, 9, 11]
    SCALE_PARAMS = [1/3, 1/4, 1/5, 1/6]
    SCALE_PAYS   = [0, 1]
    ACE_PAYOUTS  = [10, 20, 30, 40]

    # Build all combos
    combos = []
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
                    # Use the same base_seed across all parameter combos so that
                    # for a given round r, round_seed(base_seed, r) yields the
                    # same board/permutation across the grid. This enables
                    # like-for-like comparisons across params and signal types.
                    seed_i = int(base_seed)
                    outfile = out_dir / f"{key_id}.npz"
                    combos.append((raw, norm, key_tuple, key_id, seed_i, outfile))

    # slicing param grid for array jobs 
    if sweep_index is not None:
        if sweep_stride <= 0:
            raise ValueError("sweep_stride must be >= 1")
        if not (0 <= sweep_index < sweep_stride):
            raise ValueError("sweep_index must be in [0, sweep_stride)")
        combos = [item for i, item in enumerate(combos) if (i % sweep_stride) == sweep_index]

    index = {}

    def _run_one(item):
        raw, norm, key_tuple, key_id, seed_i, outfile = item
        dist, summary, meta, hists = simulate_experiment(
            seed_int=seed_i,
            rounds=rounds,
            max_signals=max_signals,
            procs=procs_inner,
            params=norm,
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

    # Single-run params (optional when not sweeping)
    ap.add_argument("--signal_cost", type=float, default=7.0)
    ap.add_argument("--scale_pay",   type=int,   default=0, choices=[0, 1])
    ap.add_argument("--scale_param", type=float, default=0.25)
    ap.add_argument("--ace_payout",  type=float, default=20.0)

    # Output
    ap.add_argument("--out",        type=str, default=None,
                    help="single-run output path; default=output/<canonical_id>.npz")

    # Sweep
    ap.add_argument("--sweep",       action="store_true")
    ap.add_argument("--sweep_out",   type=str, default="output",
                    help="directory for sweep outputs (files saved directly in this dir)")
    # removed: --sweep_jobs; inter-run parallelism handled externally if needed
    ap.add_argument("--sweep_index", type=int, default=None,
                    help="process only combos with idx %% sweep_stride == sweep_index")
    ap.add_argument("--sweep_stride", type=int, default=1,
                    help="stride used with --sweep_index")
    # empirical posteriors NPZ is required and read from POST_NPZ_DEFAULT; run precompute script to generate it

    args = ap.parse_args()

    if args.sweep:
        t0 = perf_counter()
        index = run_sweep(
            base_seed=int(args.seed),
            rounds=int(args.rounds),
            max_signals=int(args.max_signals),
            procs_inner=int(args.procs),
            out_dir=args.sweep_out,
            sweep_index=args.sweep_index,
            sweep_stride=args.sweep_stride,
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
    out_path = pathlib.Path(args.out) if args.out else pathlib.Path("output") / f"{key_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)


    t0 = perf_counter()
    dist, summary, meta, hists = simulate_experiment(
        seed_int=int(args.seed),
        rounds=int(args.rounds),
        max_signals=int(args.max_signals),
        procs=int(args.procs),
        params=norm_params,
    )
    save_npz(out_path, args, dist, summary, meta, norm_params, raw_params, key_tuple, key_id, hists)
    t1 = perf_counter()
    print(f"wrote {out_path} | total={t1 - t0:.2f}s")

if __name__ == "__main__":
    main()

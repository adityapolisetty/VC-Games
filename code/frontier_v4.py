#!/usr/bin/env python3
"""
frontier_v4.py — Memory-optimized full enumeration with shared weight matrices

Key optimization from v3:
- Build Wm2_all ONCE in parent process (not 640× in workers)
- Share read-only across workers via fork semantics
- Removes unused Stage 2 expectation computation
- Uses multiprocessing.Pool instead of ProcessPoolExecutor for better memory sharing

Memory improvement: 640 copies → 1 shared copy = ~640GB → ~10GB
Initialization: 10×5min → 1×5min = 50min → 5min
"""

import argparse
import json
import pathlib
from typing import Dict, List, Tuple
import multiprocessing as mp

import numpy as np
from numpy.random import default_rng

# Import shared functions and constants
from sim_res import round_seed, _deal_cards_global_deck
from precomp import _second_highest_rank
from fns import canonicalize_params, NUM_PILES, CARDS_PER_PILE, ACE_RANK, BUDGET

# -----------------------
# Constants / defaults
# -----------------------

# Posterior NPZ (unified file with both marginal and joint posteriors)
POST_NPZ_DEFAULT = "../precomp_output/post_mc.npz"

# Fixed params per request
ACE_PAYOUT = 20.0
SCALE_PARAM_ON = 0.25
ALPHA_GRID = np.linspace(0.25, 1.0, 4)
UNITS = 9
SD_STEP = 0.01  # percentage points (back to 1pp for precision)

# -----------------------
# GLOBAL SHARED DATA (inherited by workers via fork)
# -----------------------
_SHARED_WM1 = None
_SHARED_WM2_ALL = None
_SHARED_STRATEGY_MAP = None
_SHARED_STAGE1_INDICES = None


def _initialize_shared_weights():
    """
    Build weight matrices ONCE in parent process.
    Workers inherit via fork (copy-on-write, no duplication).

    Returns counts for validation.
    """
    global _SHARED_WM1, _SHARED_WM2_ALL, _SHARED_STRATEGY_MAP, _SHARED_STAGE1_INDICES

    # Stage 1: Full enumeration over all 9 piles
    _SHARED_WM1 = _weight_splits(UNITS, NUM_PILES)
    Ns1 = _SHARED_WM1.shape[0]

    # Stage 2: Build all weight combinations for each Stage 1's support
    strategy_map = []
    Wm2_all_list = []

    for i1 in range(Ns1):
        support_mask = (_SHARED_WM1[i1] > 0)
        stage2_weights = _build_stage2_weights_for_support(support_mask, UNITS)

        for w2 in stage2_weights:
            strategy_map.append((i1, len(Wm2_all_list)))
            Wm2_all_list.append(w2)

    _SHARED_WM2_ALL = np.array(Wm2_all_list, dtype=float)
    _SHARED_STRATEGY_MAP = strategy_map

    # Pre-compute Stage 1 index array for vectorized expansion
    _SHARED_STAGE1_INDICES = np.array([i1 for i1, _ in strategy_map], dtype=int)

    Ns_total = len(strategy_map)
    return Ns1, Ns_total


# -----------------------
# Helpers (synced with dynamic code)
# -----------------------

def _load_posteriors(npz_path: str):
    """
    Load all posteriors from unified NPZ file (both marginal and joint).

    Returns:
        rmax_tables: dict with 'median' and 'top2' marginal posteriors P(Rmax|signal)
        joint_tables: dict with 'median' and 'top2' joint posteriors P(Rmax|signal,R2)
        prior_rmax: P(Rmax) prior distribution
        r2_marginal: P(Rmax|R2) marginal distribution
    """
    p = pathlib.Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"post_npz not found: {p}")

    with np.load(p, allow_pickle=False) as z:
        # Check for required arrays from unified NPZ
        req = {
            "rmax_median_keys", "rmax_median_mat", "rmax_top2_keys", "rmax_top2_mat",
            "joint_median_keys", "joint_median_mat", "joint_top2_keys", "joint_top2_mat",
            "prior_rmax", "r2_marginal_mat"
        }
        missing = [k for k in req if k not in z.files]
        if missing:
            raise ValueError(f"post_npz missing arrays: {missing}")

        # Load marginal posteriors P(Rmax|signal)
        m_keys = np.asarray(z["rmax_median_keys"], int)
        m_mat = np.asarray(z["rmax_median_mat"], float)
        t_keys = np.asarray(z["rmax_top2_keys"], int)
        t_mat = np.asarray(z["rmax_top2_mat"], float)

        # Load joint posteriors P(Rmax|signal,R2)
        jm_keys = np.asarray(z["joint_median_keys"], int)
        jm_mat = np.asarray(z["joint_median_mat"], float)
        jt_keys = np.asarray(z["joint_top2_keys"], int)
        jt_mat = np.asarray(z["joint_top2_mat"], float)

        # Load common arrays
        prior = np.asarray(z["prior_rmax"], float)
        r2_marg = np.asarray(z["r2_marginal_mat"], float)

    # Build marginal posterior lookup dicts
    rmax_median = {int(k): np.array(m_mat[i], float) for i, k in enumerate(m_keys)}
    rmax_top2 = {int(k): np.array(t_mat[i], float) for i, k in enumerate(t_keys)}
    rmax_tables = {"median": rmax_median, "top2": rmax_top2}

    # Build joint posterior lookup with row mapping
    def _rowmap(keys: np.ndarray):
        return {int(k): int(i) for i, k in enumerate(keys.tolist())}

    joint_tables = {
        "median": (jm_keys, jm_mat, _rowmap(jm_keys)),
        "top2": (jt_keys, jt_mat, _rowmap(jt_keys)),
    }

    return rmax_tables, joint_tables, prior, r2_marg


def _hvals(scale_pay: int, scale_param: float, ace_payout: float, half: bool = False) -> np.ndarray:
    ranks_all = np.arange(2, 15, dtype=int)
    if scale_pay == 0:
        v = np.array([float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
    else:
        v = np.array([float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)
    if half:
        v = 0.5 * v
    return v


def _per_dollar_realized(max_rank: np.ndarray, scale_pay: int, scale_param: float, ace_payout: float) -> np.ndarray:
    r = np.asarray(max_rank, int)
    if scale_pay == 0:
        return np.where(r == ACE_RANK, float(ace_payout), 0.0).astype(float)
    steps = (ACE_RANK - r).clip(min=0)
    return (float(ace_payout) * (float(scale_param) ** steps)).astype(float)


def _weight_splits(units: int, m: int) -> np.ndarray:
    """
    Enumerate nonnegative integer compositions of `units` into `m` parts.
    Returns array of shape (n_combinations, m) with fractional weights.
    """
    out: List[List[int]] = []
    def rec(pos: int, remaining: int, cur: List[int]):
        if pos == m - 1:
            cur.append(remaining)
            out.append(cur.copy())
            cur.pop()
            return
        for t in range(remaining + 1):
            cur.append(t)
            rec(pos + 1, remaining - t, cur)
            cur.pop()
    rec(0, units, [])
    return (np.asarray(out, float) / float(units)).reshape((-1, m))


def _build_stage2_weights_for_support(support_mask: np.ndarray, units: int) -> np.ndarray:
    """
    For a Stage 1 strategy with given support (boolean mask), enumerate all
    possible Stage 2 weight allocations over the supported piles.

    Args:
        support_mask: Boolean array of shape (NUM_PILES,) indicating which piles have Stage 1 investment
        units: Weight discretization units

    Returns:
        Array of shape (n_weights, NUM_PILES) with all possible Stage 2 weight vectors
    """
    support_indices = np.flatnonzero(support_mask)
    m = len(support_indices)

    if m == 0:
        # No support - return single zero vector
        return np.zeros((1, NUM_PILES), float)

    # Generate all weight splits over m supported piles
    weights_on_support = _weight_splits(units, m)  # Shape: (n_combinations, m)

    # Map back to full NUM_PILES space
    n_combinations = weights_on_support.shape[0]
    full_weights = np.zeros((n_combinations, NUM_PILES), float)
    full_weights[:, support_indices] = weights_on_support

    return full_weights


def _concat_stats(stats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    # Extract statistics for expanded strategy space
    g1 = stats["sum_g1"]
    g2 = stats["sum_g2"]
    g1sq = stats["sum_g1_sq"]
    g2sq = stats["sum_g2_sq"]
    g12 = stats["sum_g12"]
    ace_hits = stats["ace_hits"]
    king_hits = stats["king_hits"]
    queen_hits = stats["queen_hits"]
    cnt = int(stats["count"])

    return g1, g2, g1sq, g2sq, g12, cnt, ace_hits, king_hits, queen_hits


def _worker_chunk_il(args):
    """
    Worker for IL frontier computation using SHARED weight matrices.

    Memory optimization: Reads from global _SHARED_WM1, _SHARED_WM2_ALL
    inherited via fork (copy-on-write, no duplication).
    """
    base_seed, round_start, rounds_chunk, signal_type, n_sig, sp, scale_param, rmax_tables, joint_tables, prior_rmax, r2_marginal = args

    # Access shared data (no copying!)
    Wm1 = _SHARED_WM1
    Wm2_all = _SHARED_WM2_ALL
    strategy_map = _SHARED_STRATEGY_MAP
    stage1_indices = _SHARED_STAGE1_INDICES

    Ns_total = len(strategy_map)

    stats = dict(
        sum_g1=np.zeros(Ns_total, float),
        sum_g2=np.zeros(Ns_total, float),
        sum_g1_sq=np.zeros(Ns_total, float),
        sum_g2_sq=np.zeros(Ns_total, float),
        sum_g12=np.zeros(Ns_total, float),
        ace_hits=np.zeros(Ns_total, int),
        king_hits=np.zeros(Ns_total, int),
        queen_hits=np.zeros(Ns_total, int),
        count=0,
    )

    h1 = _hvals(sp, scale_param, ACE_PAYOUT, half=False)
    h2 = _hvals(sp, scale_param, ACE_PAYOUT, half=True)
    post_table = rmax_tables[signal_type]

    for i in range(rounds_chunk):
        r = int(round_start) + int(i)
        # DETERMINISM: same base_seed and round r → same RNG state → same board
        rng = default_rng(round_seed(int(base_seed), r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
        # Compute has_king and has_queen for premium card tracking
        has_king = np.array([13 in np.asarray(h, int) for h in hands], dtype=bool)
        has_queen = np.array([12 in np.asarray(h, int) for h in hands], dtype=bool)
        pi = rng.permutation(NUM_PILES)
        chosen_idx = pi[:n_sig]
        chosen_set = set(int(x) for x in np.asarray(chosen_idx, int))
        buckets = np.asarray(medians if signal_type == "median" else top2sum, int)
        prior_vec = np.asarray(prior_rmax, float)
        s1 = np.zeros(NUM_PILES, float)
        for j in range(NUM_PILES):
            vec = np.asarray(post_table.get(int(buckets[j]), prior_vec), float) if (j in chosen_set) else prior_vec
            s1[j] = float(np.dot(h1, vec))
        order1 = np.argsort(-s1)
        p_real = _per_dollar_realized(np.asarray(max_rank, int), sp, scale_param, ACE_PAYOUT)
        p_real_stage2 = 0.5 * p_real  # Stage 2 receives 0.5x payoff per pound

        # Rank piles by Stage 1 expected value
        top_idx = order1  # All 9 companies in ranked order
        p_m = p_real[top_idx]
        p_m_stage2 = p_real_stage2[top_idx]

        # Compute Stage 1 returns for all Stage 1 strategies (vectorized)
        g1_all = Wm1 @ p_m  # Shape: (Ns1,)

        # Compute Stage 2 returns for all Stage 2 weight vectors (vectorized)
        g2_vec = Wm2_all @ p_m_stage2  # Shape: (Ns_total,)

        # Expand g1 using pre-computed index array (vectorized!)
        g1_expanded = g1_all[stage1_indices]

        stats["sum_g1"] += g1_expanded
        stats["sum_g2"] += g2_vec
        stats["sum_g1_sq"] += g1_expanded * g1_expanded
        stats["sum_g2_sq"] += g2_vec * g2_vec
        stats["sum_g12"] += g1_expanded * g2_vec

        # Track premium card hits: which Stage 1 strategies invested in piles containing aces/kings/queens
        has_ace_ranked = has_ace[top_idx]
        has_king_ranked = has_king[top_idx]
        has_queen_ranked = has_queen[top_idx]
        ace_hit_mask_s1 = np.any((Wm1 > 0) & has_ace_ranked, axis=1)  # Shape: (Ns1,)
        king_hit_mask_s1 = np.any((Wm1 > 0) & has_king_ranked, axis=1)
        queen_hit_mask_s1 = np.any((Wm1 > 0) & has_queen_ranked, axis=1)

        # Expand using vectorized indexing
        stats["ace_hits"] += ace_hit_mask_s1[stage1_indices].astype(int)
        stats["king_hits"] += king_hit_mask_s1[stage1_indices].astype(int)
        stats["queen_hits"] += queen_hit_mask_s1[stage1_indices].astype(int)

        stats["count"] += 1

    return stats


def _collect_stats_il(seed: int, rounds: int, procs: int, signal_type: str, n_sig: int, sp: int, scale_param: float,
                      rmax_tables, joint_tables, prior_rmax, r2_marginal):
    """
    Collect stats using multiprocessing.Pool for better memory sharing.
    Workers inherit shared weight matrices via fork (no copying!).
    """
    if procs and int(procs) > 1 and int(rounds) > 1:
        W = int(procs)
        base = int(rounds) // W
        rem = int(rounds) % W
        chunk_sizes = [base + (1 if i < rem else 0) for i in range(W)]
        starts = []
        s = 0
        for c in chunk_sizes:
            if c > 0:
                starts.append((s, c))
                s += c

        # Build argument list
        args_list = [
            (int(seed), int(start), int(sz), signal_type, int(n_sig), int(sp), float(scale_param),
             rmax_tables, joint_tables, prior_rmax, r2_marginal)
            for (start, sz) in starts
        ]

        # Use Pool with fork context (Unix default, enables memory sharing)
        with mp.Pool(processes=len(starts)) as pool:
            results = pool.map(_worker_chunk_il, args_list)

        # Aggregate results
        out_stats = None
        for st in results:
            if out_stats is None:
                out_stats = st
            else:
                for k in ("sum_g1","sum_g2","sum_g1_sq","sum_g2_sq","sum_g12","ace_hits","king_hits","queen_hits"):
                    out_stats[k] += st[k]
                out_stats["count"] += st["count"]

        return out_stats
    else:
        args = (int(seed), 0, int(rounds), signal_type, int(n_sig), int(sp), float(scale_param),
                rmax_tables, joint_tables, prior_rmax, r2_marginal)
        return _worker_chunk_il(args)


def _save_bins_npz(out_path: pathlib.Path, sd_levels_by_n, best_means_by_n, best_weights_s1_by_n, best_weights_s2_by_n, best_ace_hits_by_n, best_king_hits_by_n, best_queen_hits_by_n, meta: dict):
    payload = dict(
        sd_step=float(SD_STEP),
        sd_levels_by_n=np.array(sd_levels_by_n, dtype=object),
        best_means_by_n=np.array(best_means_by_n, dtype=object),
        best_weights_s1_by_n=np.array(best_weights_s1_by_n, dtype=object),
        best_weights_s2_by_n=np.array(best_weights_s2_by_n, dtype=object),
        best_ace_hits_by_n=np.array(best_ace_hits_by_n, dtype=object),
        best_king_hits_by_n=np.array(best_king_hits_by_n, dtype=object),
        best_queen_hits_by_n=np.array(best_queen_hits_by_n, dtype=object),
        meta=json.dumps(meta),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile, os
    fd, tmp_path = tempfile.mkstemp(prefix=out_path.stem + '.', suffix='.npz', dir=str(out_path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, str(out_path))
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def simulate_and_save_frontier(seed_int, rounds, max_signals, procs, params, stage1_alloc, out_dir: pathlib.Path, signal_type: str):
    import time

    # Load all posteriors from unified NPZ file
    print(f"[v4] Loading posteriors...")
    rmax_tables, joint_tables, prior_rmax, r2_marginal = _load_posteriors(POST_NPZ_DEFAULT)
    sp = int(params["scale_pay"])
    scale_param = float(params["scale_param"]) if sp == 1 else 0.0
    st = str(signal_type)

    # For each n, collect stats across rounds
    stats_by_n = {}
    print(f"[v4] Processing {max_signals + 1} n_sig values (0-{max_signals}) with {rounds} rounds, {procs} workers")
    print(f"[v4] File: alpha={stage1_alloc:.2f}, scale_pay={sp}, signal_type={st}")
    print("")

    for n_sig in range(int(max_signals) + 1):
        # Validate: skip if stage1_alloc=0 (no budget) or cannot afford n_sig signals
        budget1 = float(stage1_alloc) * BUDGET
        signal_cost = float(params.get('signal_cost', 3.0))
        signal_cost_total = float(n_sig) * signal_cost
        if stage1_alloc <= 0 or budget1 < signal_cost_total:
            # No Stage 1 budget or cannot afford signals - skip simulation entirely
            print(f"[v4] n_sig={n_sig}: SKIPPED (cannot afford {n_sig} signals with alpha={stage1_alloc:.2f})")
            stats_by_n[n_sig] = None
            continue

        print(f"[v4] n_sig={n_sig}/{max_signals}: Starting {rounds} rounds with {procs} workers...")
        t_start = time.time()

        result = _collect_stats_il(
            seed=int(seed_int), rounds=int(rounds), procs=int(procs), signal_type=st, n_sig=int(n_sig), sp=sp, scale_param=scale_param,
            rmax_tables=rmax_tables, joint_tables=joint_tables, prior_rmax=prior_rmax, r2_marginal=r2_marginal,
        )
        stats_by_n[n_sig] = result

        t_elapsed = time.time() - t_start
        print(f"[v4] n_sig={n_sig}/{max_signals}: COMPLETED in {t_elapsed:.1f}s ({t_elapsed/60:.2f} min)")

        # Estimate remaining time
        remaining_n = max_signals - n_sig
        if remaining_n > 0:
            est_remaining = t_elapsed * remaining_n / 60
            print(f"[v4] Estimated time remaining: {est_remaining:.1f} minutes for {remaining_n} more n_sig values")
        print("")

    # For this alpha, compute bins per n and save
    sd_levels_by_n = []
    best_means_by_n = []
    best_weights_s1_by_n = []  # Stage 1 weights
    best_weights_s2_by_n = []  # Stage 2 weights
    best_ace_hits_by_n = []
    best_king_hits_by_n = []
    best_queen_hits_by_n = []

    for n_sig in range(int(max_signals) + 1):
        stats = stats_by_n[n_sig]
        # Skip configurations where signals were unaffordable
        if stats is None:
            sd_levels_by_n.append(np.array([], float))
            best_means_by_n.append(np.array([], float))
            best_weights_s1_by_n.append(np.zeros((0, NUM_PILES), float))
            best_weights_s2_by_n.append(np.zeros((0, NUM_PILES), float))
            best_ace_hits_by_n.append(np.array([], int))
            best_king_hits_by_n.append(np.array([], int))
            best_queen_hits_by_n.append(np.array([], int))
            continue
        g1, g2, g1sq, g2sq, g12, cnt, ace_hits, king_hits, queen_hits = _concat_stats(stats)
        if cnt <= 1 or g1.size == 0:
            sd_levels_by_n.append(np.array([], float))
            best_means_by_n.append(np.array([], float))
            best_weights_s1_by_n.append(np.zeros((0, NUM_PILES), float))
            best_weights_s2_by_n.append(np.zeros((0, NUM_PILES), float))
            best_ace_hits_by_n.append(np.array([], int))
            best_king_hits_by_n.append(np.array([], int))
            best_queen_hits_by_n.append(np.array([], int))
            continue
        budget1 = float(stage1_alloc) * BUDGET
        signal_cost = float(params.get('signal_cost', 3.0))
        investable1 = max(0.0, budget1 - float(n_sig) * signal_cost)
        budget2 = max(0.0, BUDGET - budget1)
        c1 = investable1 / BUDGET
        # Stage 2 can only invest in piles that Stage 1 invested in
        if investable1 <= 0:
            c2 = 0.0
        else:
            c2 = budget2 / BUDGET  # Full budget2 (0.5x payoff already in g2)
        mean_g1 = g1 / cnt
        mean_g2 = g2 / cnt
        var_g1 = g1sq / cnt - mean_g1 ** 2
        var_g2 = g2sq / cnt - mean_g2 ** 2
        cov_g12 = g12 / cnt - (mean_g1 * mean_g2)
        # Signal cost is a permanent sunk cost (not recoverable cash)
        signal_cost_total = float(n_sig) * signal_cost
        signal_cost_fraction = signal_cost_total / BUDGET
        mean_net = 100.0 * (c1 * (mean_g1 - 1.0) + c2 * (mean_g2 - 1.0) - signal_cost_fraction)
        var_net = (100.0 ** 2) * ((c1 ** 2) * np.clip(var_g1, 0, np.inf) + (c2 ** 2) * np.clip(var_g2, 0, np.inf) + 2.0 * c1 * c2 * cov_g12)
        sd_net = np.sqrt(np.clip(var_net, 0.0, np.inf))

        # Bin by SD levels
        bins = np.floor(sd_net / SD_STEP).astype(int)
        # For each bin, pick index with highest mean
        max_bin = int(np.max(bins)) if bins.size else -1
        sd_levels = []
        best_means = []
        best_weights_s1 = []
        best_weights_s2 = []
        best_ace_hits = []
        best_king_hits = []
        best_queen_hits = []
        if max_bin >= 0:
            for b in range(max_bin + 1):
                mask = (bins == b)
                if not np.any(mask):
                    continue
                idx = np.argmax(mean_net[mask])
                sel = np.flatnonzero(mask)[idx]
                # Extract strategy metadata: (i1, i2)
                i1, i2 = _SHARED_STRATEGY_MAP[sel]
                sd_levels.append(float(b) * SD_STEP)
                best_means.append(float(mean_net[sel]))
                best_weights_s1.append(_SHARED_WM1[i1])  # Store Stage 1 weights
                best_weights_s2.append(_SHARED_WM2_ALL[i2])  # Store Stage 2 weights
                best_ace_hits.append(int(ace_hits[sel]))
                best_king_hits.append(int(king_hits[sel]))
                best_queen_hits.append(int(queen_hits[sel]))
        sd_levels_by_n.append(np.array(sd_levels, float))
        best_means_by_n.append(np.array(best_means, float))
        best_weights_s1_by_n.append(np.array(best_weights_s1, float))
        best_weights_s2_by_n.append(np.array(best_weights_s2, float))
        best_ace_hits_by_n.append(np.array(best_ace_hits, int))
        best_king_hits_by_n.append(np.array(best_king_hits, int))
        best_queen_hits_by_n.append(np.array(best_queen_hits, int))

    # Save per (params, alpha, st)
    print(f"[v4] Computing bins and saving results...")
    norm_params, key_tuple, key_id = canonicalize_params(params)
    a_tag = f"a{int(round(float(stage1_alloc)*100)):03d}"
    out_path = out_dir / f"{key_id}_{st}_{a_tag}.npz"
    meta = dict(
        mode="frontier_il_v4",
        stage1_alloc=float(stage1_alloc),
        signal_type=st,
        params=norm_params,
        total_rounds=int(rounds),
        units=int(UNITS),
        full_stage2_enumeration=True,
        memory_optimized=True,
    )
    _save_bins_npz(out_path, sd_levels_by_n, best_means_by_n, best_weights_s1_by_n, best_weights_s2_by_n, best_ace_hits_by_n, best_king_hits_by_n, best_queen_hits_by_n, meta)
    print(f"[v4] Saved: {out_path.name}")
    print(f"[v4] File complete!")


def run_sweep(base_seed, rounds, max_signals, procs_inner, out_dir,
              sweep_index=None, sweep_stride=1, skip_existing=False):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reduced parameter grid for v4
    SIGNAL_COSTS = [3]  # Single value
    SCALE_PARAMS = [0.25]  # Single value
    SCALE_PAYS = [1]  # Keep both for comparison
    ACE_PAYOUTS = [20.0]  # Single value

    combos = []
    for alpha in ALPHA_GRID:
        for sc in SIGNAL_COSTS:
            for sp in SCALE_PAYS:
                for s in SCALE_PARAMS:
                    for ap in ACE_PAYOUTS:
                        raw = dict(signal_cost=float(sc), scale_pay=sp, scale_param=(s if sp == 1 else 0.0), ace_payout=ap)
                        norm, key_tuple, key_id = canonicalize_params(raw)
                        for st in ("median", "top2"):
                            a_tag = f"a{int(round(float(alpha)*100)):03d}"
                            outfile = out_dir / f"{key_id}_{st}_{a_tag}.npz"
                            combos.append((raw, alpha, st, outfile))

    # Slice combos by stride/index
    if sweep_index is not None:
        stride = max(1, int(sweep_stride))
        idx = int(sweep_index)
        combos = [c for i, c in enumerate(combos) if (i % stride) == idx]

    # Skip existing
    if skip_existing:
        combos = [c for c in combos if not c[3].exists()]

    # BUILD WEIGHT MATRICES ONCE (shared across all n_sig values!)
    import time
    print(f"[v4] ========================================")
    print(f"[v4] Building shared weight matrices (UNITS={UNITS}, NUM_PILES={NUM_PILES})...")
    t_init_start = time.time()
    Ns1, Ns_total = _initialize_shared_weights()
    t_init_elapsed = time.time() - t_init_start
    print(f"[v4] Initialization COMPLETE in {t_init_elapsed:.1f}s ({t_init_elapsed/60:.2f} min)")
    print(f"[v4] Stage 1 strategies: {Ns1:,}")
    print(f"[v4] Total strategy pairs: {Ns_total:,}")
    print(f"[v4] Memory footprint: ~{Ns_total * NUM_PILES * 8 / 1e9:.2f} GB for Wm2_all")
    print(f"[v4] Workers will inherit via fork (no duplication!)")
    print(f"[v4] ========================================\n")

    # Execute each combo independently (per-signal_type) to avoid overwrites across array jobs
    for raw, alpha, st, outfile in combos:
        simulate_and_save_frontier(
            seed_int=int(base_seed), rounds=int(rounds), max_signals=int(max_signals), procs=int(procs_inner),
            params=raw, stage1_alloc=float(alpha), out_dir=out_dir, signal_type=st,
        )


def main():
    """
    Information-limited frontier computation with FULL Stage 2 enumeration.

    v4: Memory-optimized with shared weight matrices across workers.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="Base seed for deterministic rounds")
    ap.add_argument("--rounds", type=int, required=True, help="Number of simulation rounds")
    ap.add_argument("--max_signals", type=int, default=9, help="Max signals to sweep")
    ap.add_argument("--procs", type=int, default=1, help="Intra-run parallelism (workers per job)")

    # Sweep options (primary usage mode)
    ap.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    ap.add_argument("--sweep_out", type=str, default="frontier_output_v4", help="Output directory for sweep")
    ap.add_argument("--sweep_index", type=int, default=None, help="Array job index")
    ap.add_argument("--sweep_stride", type=int, default=1, help="Array job stride")
    ap.add_argument("--skip_existing", action="store_true", help="Skip existing output files")

    args = ap.parse_args()

    if not args.sweep:
        raise ValueError("Single-run mode not supported. Use --sweep for frontier computation.")

    run_sweep(
        base_seed=int(args.seed), rounds=int(args.rounds), max_signals=int(args.max_signals), procs_inner=int(args.procs),
        out_dir=args.sweep_out, sweep_index=args.sweep_index, sweep_stride=args.sweep_stride, skip_existing=bool(args.skip_existing),
    )


if __name__ == "__main__":
    main()

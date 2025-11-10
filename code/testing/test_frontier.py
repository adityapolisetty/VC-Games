#!/usr/bin/env python3
"""
Local test runner for frontier.py logic.

Shows detailed NPZ-level information for 2 specific strategies:
1. Fully concentrated: [1, 0, 0, 0, 0, 0, 0, 0, 0]
2. Fully diversified: [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]

Runs 1 round and shows statistics for all n_sig values (0 to 9).
"""

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
from numpy.random import default_rng

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import shared functions and constants
from sim_res import round_seed, _deal_cards_global_deck
from precomp import _second_highest_rank
from fns import canonicalize_params, NUM_PILES, CARDS_PER_PILE, ACE_RANK, BUDGET

# -----------------------
# Constants / defaults
# -----------------------

# Posterior NPZ (unified file with both marginal and joint posteriors)
POST_NPZ_DEFAULT = "../../precomp_output/post_mc.npz"

# Fixed params per request
ACE_PAYOUT = 20.0
SCALE_PARAM_ON = 0.25
ALPHA_GRID = np.linspace(0, 1.0, 21) 
UNITS = 9
SD_STEP = 0.1  # percentage points


# -----------------------
# Helpers (synced with dynamic code)
# -----------------------
# Functions imported from sim_res: round_seed, _deal_cards_global_deck, _second_highest_rank


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
    # enumerate nonnegative integer compositions of `units` into `m` parts
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


def _concat_stats(stats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Extract statistics directly (m=NUM_PILES=9 hardcoded, no concatenation needed)
    g1 = stats["sum_g1"]
    g2 = stats["sum_g2"]
    g1sq = stats["sum_g1_sq"]
    g2sq = stats["sum_g2_sq"]
    g12 = stats["sum_g12"]
    ace_hits = stats["ace_hits"]
    king_hits = stats["king_hits"]
    queen_hits = stats["queen_hits"]
    cnt = int(stats["count"])
    # Weight template is already in (43758, NUM_PILES) format
    rank9 = stats.get("Wm_template", _weight_splits(UNITS, NUM_PILES))
    return g1, g2, g1sq, g2sq, g12, cnt, rank9, ace_hits, king_hits, queen_hits


def _worker_chunk_il(base_seed, round_start, rounds_chunk, signal_type, n_sig, sp, scale_param, rmax_tables, joint_tables, prior_rmax, r2_marginal, debug_excel=False):
    """
    Worker for IL frontier computation with deterministic round seeding.

    CRITICAL: round_seed(base_seed, r) generates the SAME RNG state for round r
    across ALL parameter combinations. This ensures:
    - Same board (hands, medians, top2sum, max_rank, R2)
    - Same permutation (pi)
    - Different chosen_idx based on n_sig (first n_sig from same permutation)
    - Different rankings based on which piles were observed
    """
    # Generate ALL weight strategies (same as original)
    Wm_full = _weight_splits(UNITS, NUM_PILES)

    # Filter to only 2 strategies: concentrated and diversified
    concentrated = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    diversified = np.array([1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9])

    # Find indices of these strategies in the full weight matrix
    concentrated_idx = None
    diversified_idx = None
    for i in range(Wm_full.shape[0]):
        if np.allclose(Wm_full[i], concentrated):
            concentrated_idx = i
        if np.allclose(Wm_full[i], diversified):
            diversified_idx = i

    # Use only these 2 strategies
    strategy_indices = [concentrated_idx, diversified_idx]
    Wm = Wm_full[strategy_indices]
    Ns = Wm.shape[0]

    stats = dict(
        sum_g1=np.zeros(Ns, float),
        sum_g2=np.zeros(Ns, float),
        sum_g1_sq=np.zeros(Ns, float),
        sum_g2_sq=np.zeros(Ns, float),
        sum_g12=np.zeros(Ns, float),
        ace_hits=np.zeros(Ns, int),  # Count rounds where strategy invested in ace-containing piles
        king_hits=np.zeros(Ns, int),  # Count rounds where strategy invested in king-containing piles
        queen_hits=np.zeros(Ns, int),  # Count rounds where strategy invested in queen-containing piles
        count=0,
        Wm_template=Wm,
    )

    h1 = _hvals(sp, scale_param, ACE_PAYOUT, half=False)
    h2 = _hvals(sp, scale_param, ACE_PAYOUT, half=True)
    post_table = rmax_tables[signal_type]
    keys, mat3d, rowmap = joint_tables[signal_type]

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
        R2 = np.array([_second_highest_rank(h) for h in hands], int)
        p_real = _per_dollar_realized(np.asarray(max_rank, int), sp, scale_param, ACE_PAYOUT)
        p_real_stage2 = 0.5 * p_real  # Stage 2 receives 0.5x payoff per pound

        # Investment in all NUM_PILES companies with various weight strategies
        top_idx = order1  # All 9 companies in ranked order
        p_m = p_real[top_idx]
        p_m_stage2 = p_real_stage2[top_idx]
        # Stage-2 expected ordering within support
        s2_m = np.zeros(NUM_PILES, float)
        for jj, k in enumerate(top_idx):
            r2k = int(R2[k]) - 2
            if k in chosen_set:
                b = int(buckets[k])
                if (b in rowmap) and (0 <= r2k < 13):
                    vec = np.asarray(mat3d[rowmap[b], r2k, :], float)
                else:
                    vec = prior_vec
            else:
                vec = np.asarray(r2_marginal[r2k, :], float) if (0 <= r2k < 13) else prior_vec
            s2_m[jj] = float(np.dot(h2, vec))
        perm2 = np.argsort(-s2_m)
        g1 = Wm @ p_m
        g2 = Wm[:, perm2] @ p_m_stage2[perm2]  # Use Stage 2 payouts (0.5x) reordered by Stage 2 ranking
        stats["sum_g1"] += g1
        stats["sum_g2"] += g2
        stats["sum_g1_sq"] += g1 * g1
        stats["sum_g2_sq"] += g2 * g2
        stats["sum_g12"] += g1 * g2

        # Track premium card hits: which strategies invested in piles containing aces/kings/queens
        has_ace_ranked = has_ace[top_idx]  # Boolean array for ranked positions
        has_king_ranked = has_king[top_idx]
        has_queen_ranked = has_queen[top_idx]
        ace_hit_mask = np.any((Wm > 0) & has_ace_ranked, axis=1)  # True if strategy invested in any ace pile
        king_hit_mask = np.any((Wm > 0) & has_king_ranked, axis=1)
        queen_hit_mask = np.any((Wm > 0) & has_queen_ranked, axis=1)
        stats["ace_hits"] += ace_hit_mask.astype(int)
        stats["king_hits"] += king_hit_mask.astype(int)
        stats["queen_hits"] += queen_hit_mask.astype(int)

        stats["count"] += 1

    return stats


def _collect_stats_il(seed: int, rounds: int, procs: int, signal_type: str, n_sig: int, sp: int, scale_param: float,
                      rmax_tables, joint_tables, prior_rmax, r2_marginal, debug_excel=False):
    if procs and int(procs) > 1 and int(rounds) > 1:
        W = int(procs); base = int(rounds) // W; rem = int(rounds) % W
        chunk_sizes = [base + (1 if i < rem else 0) for i in range(W)]
        starts = []
        s = 0
        for c in chunk_sizes:
            if c > 0:
                starts.append((s, c))
                s += c
        out_stats = None
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=len(starts)) as ex:
            futs = [
                ex.submit(
                    _worker_chunk_il,
                    int(seed), int(start), int(sz), signal_type, int(n_sig), int(sp), float(scale_param),
                    rmax_tables, joint_tables, prior_rmax, r2_marginal, debug_excel,
                ) for (start, sz) in starts
            ]
            for fut in futs:
                st = fut.result()
                if out_stats is None:
                    out_stats = st
                else:
                    for k in ("sum_g1","sum_g2","sum_g1_sq","sum_g2_sq","sum_g12","ace_hits","king_hits","queen_hits"):
                        out_stats[k] += st[k]
                    out_stats["count"] += st["count"]

        return out_stats
    else:
        return _worker_chunk_il(int(seed), 0, int(rounds), signal_type, int(n_sig), int(sp), float(scale_param), rmax_tables, joint_tables, prior_rmax, r2_marginal, debug_excel)
    


def _save_bins_npz(out_path: pathlib.Path, sd_levels_by_n, best_means_by_n, best_weights_by_n, best_ace_hits_by_n, best_king_hits_by_n, best_queen_hits_by_n, meta: dict):
    payload = dict(
        sd_step=float(SD_STEP),
        sd_levels_by_n=np.array(sd_levels_by_n, dtype=object),
        best_means_by_n=np.array(best_means_by_n, dtype=object),
        best_weights_by_n=np.array(best_weights_by_n, dtype=object),
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


def simulate_and_save_frontier(seed_int, rounds, max_signals, procs, params, stage1_alloc, out_dir: pathlib.Path, signal_type: str, debug_excel=False):
    # Load all posteriors from unified NPZ file
    rmax_tables, joint_tables, prior_rmax, r2_marginal = _load_posteriors(POST_NPZ_DEFAULT)
    sp = int(params["scale_pay"])
    scale_param = float(params["scale_param"]) if sp == 1 else 0.0
    st = str(signal_type)
    # For each n, collect stats across rounds
    stats_by_n = {}
    debug_data_by_n = {}  # Collect debug data per n_sig
    for n_sig in range(int(max_signals) + 1):
        # Validate: skip if cannot afford n_sig signals with this stage1_alloc
        budget1 = float(stage1_alloc) * BUDGET
        signal_cost = float(params.get('signal_cost', 3.0))
        signal_cost_total = float(n_sig) * signal_cost
        if budget1 < signal_cost_total:
            # Cannot afford signals - skip simulation entirely
            stats_by_n[n_sig] = None
            debug_data_by_n[n_sig] = None
            continue

        result = _collect_stats_il(
            seed=int(seed_int), rounds=int(rounds), procs=int(procs), signal_type=st, n_sig=int(n_sig), sp=sp, scale_param=scale_param,
            rmax_tables=rmax_tables, joint_tables=joint_tables, prior_rmax=prior_rmax, r2_marginal=r2_marginal, debug_excel=debug_excel,
        )
        if debug_excel:
            stats_by_n[n_sig], debug_data_by_n[n_sig] = result
        else:
            stats_by_n[n_sig] = result
            debug_data_by_n[n_sig] = None

    # For this alpha, compute bins per n and save
    sd_levels_by_n = []
    best_means_by_n = []
    best_weights_by_n = []
    best_ace_hits_by_n = []
    best_king_hits_by_n = []
    best_queen_hits_by_n = []

    for n_sig in range(int(max_signals) + 1):
        stats = stats_by_n[n_sig]

        # Skip configurations where signals were unaffordable
        if stats is None:
            sd_levels_by_n.append(np.array([], float))
            best_means_by_n.append(np.array([], float))
            best_weights_by_n.append(np.zeros((0, NUM_PILES), float))
            best_ace_hits_by_n.append(np.array([], int))
            best_king_hits_by_n.append(np.array([], int))
            best_queen_hits_by_n.append(np.array([], int))
            continue
        g1, g2, g1sq, g2sq, g12, cnt, rank9, ace_hits, king_hits, queen_hits = _concat_stats(stats)
        if cnt <= 1 or g1.size == 0:
            sd_levels_by_n.append(np.array([], float))
            best_means_by_n.append(np.array([], float))
            best_weights_by_n.append(np.zeros((0, NUM_PILES), float))
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
        best_weights = []
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
                sd_levels.append(float(b) * SD_STEP)
                best_means.append(float(mean_net[sel]))
                best_weights.append(rank9[sel])
                best_ace_hits.append(int(ace_hits[sel]))
                best_king_hits.append(int(king_hits[sel]))
                best_queen_hits.append(int(queen_hits[sel]))
        sd_levels_by_n.append(np.array(sd_levels, float))
        best_means_by_n.append(np.array(best_means, float))
        best_weights_by_n.append(np.array(best_weights, float))
        best_ace_hits_by_n.append(np.array(best_ace_hits, int))
        best_king_hits_by_n.append(np.array(best_king_hits, int))
        best_queen_hits_by_n.append(np.array(best_queen_hits, int))

    # Save per (params, alpha, st)
    norm_params, key_tuple, key_id = canonicalize_params(params)
    a_tag = f"a{int(round(float(stage1_alloc)*10)):02d}"
    out_path = out_dir / f"{key_id}_{st}_{a_tag}.npz"
    meta = dict(
        mode="frontier_il",
        stage1_alloc=float(stage1_alloc),
        signal_type=st,
        params=norm_params,
        total_rounds=int(rounds),
    )
    _save_bins_npz(out_path, sd_levels_by_n, best_means_by_n, best_weights_by_n, best_ace_hits_by_n, best_king_hits_by_n, best_queen_hits_by_n, meta)


def run_sweep(base_seed, rounds, max_signals, procs_inner, out_dir,
              sweep_index=None, sweep_stride=1, skip_existing=False, debug_excel=False):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grids per request
    SIGNAL_COSTS = [0, 3, 7, 9]
    SCALE_PARAMS = [0.25]
    SCALE_PAYS = [0, 1]
    ACE_PAYOUTS = [20.0]

    combos = []
    for alpha in ALPHA_GRID:
        for sc in SIGNAL_COSTS:
            for sp in SCALE_PAYS:
                for s in SCALE_PARAMS:
                    for ap in ACE_PAYOUTS:
                        raw = dict(signal_cost=float(sc), scale_pay=sp, scale_param=(s if sp == 1 else 0.0), ace_payout=ap)
                        norm, key_tuple, key_id = canonicalize_params(raw)
                        for st in ("median", "top2"):
                            a_tag = f"a{int(round(float(alpha)*100)):03d}"  # 000, 005, 010, ..., 100
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

    # Execute each combo independently (per-signal_type) to avoid overwrites across array jobs
    for raw, alpha, st, outfile in combos:
        simulate_and_save_frontier(
            seed_int=int(base_seed), rounds=int(rounds), max_signals=int(max_signals), procs=int(procs_inner),
            params=raw, stage1_alloc=float(alpha), out_dir=out_dir, signal_type=st, debug_excel=debug_excel,
        )


def run_local_test(seed, rounds, signal_cost, stage1_alloc, signal_type, scale_pay, post_npz_path):
    """
    Run simulation for 2 strategies across all n_sig using original frontier.py logic.
    """
    print("="*80)
    print("LOCAL FRONTIER TEST - 2 STRATEGIES, ALL n_sig")
    print("="*80)

    # Load posteriors
    rmax_tables, joint_tables, prior_rmax, r2_marginal = _load_posteriors(post_npz_path)

    # Setup parameters (matching original)
    params = {
        'signal_cost': signal_cost,
        'scale_pay': scale_pay,
        'scale_param': SCALE_PARAM_ON if scale_pay == 1 else 0.0,
        'ace_payout': ACE_PAYOUT,
    }

    scale_param = params['scale_param']

    print(f"\nParameters:")
    print(f"  seed = {seed}")
    print(f"  rounds = {rounds}")
    print(f"  signal_type = {signal_type}")
    print(f"  signal_cost = £{signal_cost}")
    print(f"  stage1_alloc = {stage1_alloc*100:.0f}%")
    print(f"  scale_pay = {scale_pay}")
    print(f"  scale_param = {scale_param}")
    print(f"  ace_payout = {ACE_PAYOUT}")

    print(f"\nStrategies:")
    print(f"  Strategy 0: Concentrated [1, 0, 0, 0, 0, 0, 0, 0, 0]")
    print(f"  Strategy 1: Diversified  [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]")

    # If rounds=1, show the actual hand for reference
    if rounds == 1:
        print(f"\n" + "="*80)
        print("HAND (Round 0) - For Reference")
        print("="*80)
        from numpy.random import default_rng
        rng = default_rng(round_seed(seed, 0))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
        R2_vals = np.array([_second_highest_rank(h) for h in hands], int)
        p_real_ref = _per_dollar_realized(np.array(max_rank, int), scale_pay, scale_param, ACE_PAYOUT)

        for i in range(NUM_PILES):
            ace_marker = "★ ACE" if has_ace[i] else ""
            print(f"  Pile {i}: {hands[i]} | max={max_rank[i]}, R2={R2_vals[i]}, median={medians[i]}, top2={top2sum[i]}, payoff=£{p_real_ref[i]:.1f} {ace_marker}")

    print(f"\n" + "="*80)
    print(f"RESULTS BY n_sig ({signal_type.upper()} SIGNALS)")
    print("="*80)

    results = []

    for n_sig in range(10):  # 0 to 9
        print(f"\n{'─'*80}")
        print(f"n_sig = {n_sig}")
        print(f"{'─'*80}")

        # Budget check
        budget1 = stage1_alloc * BUDGET
        signal_cost_total = n_sig * signal_cost

        if budget1 < signal_cost_total:
            print(f"  ⚠️  Cannot afford {n_sig} signals (need £{signal_cost_total}, have £{budget1})")
            results.append(None)
            continue

        # Run simulation using original _collect_stats_il with parallel processing
        import os
        procs = os.cpu_count() or 1  # Use all available CPUs

        stats = _collect_stats_il(
            seed=seed,
            rounds=rounds,
            procs=procs,
            signal_type=signal_type,
            n_sig=n_sig,
            sp=scale_pay,
            scale_param=scale_param,
            rmax_tables=rmax_tables,
            joint_tables=joint_tables,
            prior_rmax=prior_rmax,
            r2_marginal=r2_marginal,
            debug_excel=False
        )

        # Extract stats using original _concat_stats
        g1, g2, g1sq, g2sq, g12, cnt, rank9, ace_hits, king_hits, queen_hits = _concat_stats(stats)

        # Budget accounting (original logic)
        investable1 = max(0.0, budget1 - signal_cost_total)
        budget2 = max(0.0, BUDGET - budget1)
        c1 = investable1 / BUDGET
        c2 = budget2 / BUDGET if investable1 > 0 else 0.0

        # Calculate statistics (original logic)
        mean_g1 = g1 / cnt
        mean_g2 = g2 / cnt
        var_g1 = g1sq / cnt - mean_g1 ** 2
        var_g2 = g2sq / cnt - mean_g2 ** 2
        cov_g12 = g12 / cnt - (mean_g1 * mean_g2)

        # Net portfolio statistics with signal cost as permanent sunk cost
        signal_cost_fraction = signal_cost_total / BUDGET
        mean_net = 100.0 * (c1 * (mean_g1 - 1.0) + c2 * (mean_g2 - 1.0) - signal_cost_fraction)
        var_net = (100.0 ** 2) * ((c1 ** 2) * np.clip(var_g1, 0, np.inf) + (c2 ** 2) * np.clip(var_g2, 0, np.inf) + 2.0 * c1 * c2 * cov_g12)
        sd_net = np.sqrt(np.clip(var_net, 0.0, np.inf))

        print(f"  Budget: Stage1=£{budget1:.0f}, signals=£{signal_cost_total:.0f}, investable=£{investable1:.0f}, Stage2=£{budget2:.0f}")
        print(f"  Budget fractions: c1={c1:.4f} ({c1*100:.1f}%), c2={c2:.4f} ({c2*100:.1f}%)")

        # If rounds=1, show detailed pile rankings and selections
        if rounds == 1:
            # Reconstruct the hand and rankings for this n_sig
            from numpy.random import default_rng
            rng_hand = default_rng(round_seed(seed, 0))
            has_ace_debug, hands_debug, medians_debug, top2sum_debug, max_rank_debug, min_rank_debug = _deal_cards_global_deck(rng_hand)
            R2_debug = np.array([_second_highest_rank(h) for h in hands_debug], int)
            p_real_debug = _per_dollar_realized(np.array(max_rank_debug, int), scale_pay, scale_param, ACE_PAYOUT)
            p_real_stage2_debug = 0.5 * p_real_debug

            # Get chosen piles
            rng_sig = default_rng(round_seed(seed, 0))
            pi_debug = rng_sig.permutation(NUM_PILES)
            chosen_idx_debug = pi_debug[:n_sig]
            chosen_set_debug = set(int(x) for x in chosen_idx_debug)

            # Stage 1 ranking
            h1_debug = _hvals(scale_pay, scale_param, ACE_PAYOUT, half=False)
            h2_debug = _hvals(scale_pay, scale_param, ACE_PAYOUT, half=True)
            buckets_debug = np.array(medians_debug if signal_type == "median" else top2sum_debug, int)
            post_table_debug = rmax_tables[signal_type]
            keys_debug, mat3d_debug, rowmap_debug = joint_tables[signal_type]

            s1_debug = np.zeros(NUM_PILES, float)
            for j in range(NUM_PILES):
                vec_debug = np.asarray(post_table_debug.get(int(buckets_debug[j]), prior_rmax), float) if (j in chosen_set_debug) else prior_rmax
                s1_debug[j] = float(np.dot(h1_debug, vec_debug))

            order1_debug = np.argsort(-s1_debug)
            p_m_debug = p_real_debug[order1_debug]
            p_m_stage2_debug = p_real_stage2_debug[order1_debug]

            # Stage 2 ranking
            s2_m_debug = np.zeros(NUM_PILES, float)
            for jj, k in enumerate(order1_debug):
                r2k = int(R2_debug[k]) - 2
                if k in chosen_set_debug:
                    b = int(buckets_debug[k])
                    if (b in rowmap_debug) and (0 <= r2k < 13):
                        vec_debug = np.asarray(mat3d_debug[rowmap_debug[b], r2k, :], float)
                    else:
                        vec_debug = prior_rmax
                else:
                    vec_debug = np.asarray(r2_marginal[r2k, :], float) if (0 <= r2k < 13) else prior_rmax
                s2_m_debug[jj] = float(np.dot(h2_debug, vec_debug))

            perm2_debug = np.argsort(-s2_m_debug)

            print(f"\n  Observed piles: {sorted(chosen_set_debug) if chosen_set_debug else 'none'}")
            print(f"\n  Stage 1 Ranking (by EV):")
            for pos in range(NUM_PILES):
                pile_id = order1_debug[pos]
                obs_marker = "✓" if pile_id in chosen_set_debug else "✗"
                ace_marker = "★" if has_ace_debug[pile_id] else " "
                print(f"    Pos {pos}: Pile {pile_id} [{obs_marker}] EV={s1_debug[pile_id]:.3f}, realized=£{p_m_debug[pos]:.1f} {ace_marker}")

            print(f"\n  Stage 2 Re-ranking (by EV with R2):")
            for new_pos in range(NUM_PILES):
                old_pos = perm2_debug[new_pos]
                pile_id = order1_debug[old_pos]
                ace_marker = "★" if has_ace_debug[pile_id] else " "
                print(f"    Pos {new_pos}: Pile {pile_id} (was pos {old_pos}) EV={s2_m_debug[old_pos]:.3f}, realized=£{p_m_stage2_debug[old_pos]:.1f} {ace_marker}")

        print(f"\n  Strategy 0 (Concentrated):")
        print(f"    Stage 1: mean_g1 = {mean_g1[0]:.4f} (avg return per £1 invested)")
        print(f"             £{investable1:.2f} × {mean_g1[0]:.4f} = £{investable1 * mean_g1[0]:.2f} gross")
        print(f"             excess = {mean_g1[0] - 1.0:+.4f} per £1 → {c1 * (mean_g1[0] - 1.0) * 100:+.2f}% of total budget")
        print(f"    Stage 2: mean_g2 = {mean_g2[0]:.4f} (avg return per £1 invested)")
        print(f"             £{budget2:.2f} × {mean_g2[0]:.4f} = £{budget2 * mean_g2[0]:.2f} gross")
        print(f"             excess = {mean_g2[0] - 1.0:+.4f} per £1 → {c2 * (mean_g2[0] - 1.0) * 100:+.2f}% of total budget")
        print(f"    Net return: {c1 * (mean_g1[0] - 1.0) * 100:+.2f}% + {c2 * (mean_g2[0] - 1.0) * 100:+.2f}% = {mean_net[0]:+.2f}%")
        print(f"    SD = {sd_net[0]:.2f}%, ace_hits={ace_hits[0]}/{cnt}, king_hits={king_hits[0]}/{cnt}")

        print(f"\n  Strategy 1 (Diversified):")
        print(f"    Stage 1: mean_g1 = {mean_g1[1]:.4f} (avg return per £1 invested)")
        print(f"             £{investable1:.2f} × {mean_g1[1]:.4f} = £{investable1 * mean_g1[1]:.2f} gross")
        print(f"             excess = {mean_g1[1] - 1.0:+.4f} per £1 → {c1 * (mean_g1[1] - 1.0) * 100:+.2f}% of total budget")
        print(f"    Stage 2: mean_g2 = {mean_g2[1]:.4f} (avg return per £1 invested)")
        print(f"             £{budget2:.2f} × {mean_g2[1]:.4f} = £{budget2 * mean_g2[1]:.2f} gross")
        print(f"             excess = {mean_g2[1] - 1.0:+.4f} per £1 → {c2 * (mean_g2[1] - 1.0) * 100:+.2f}% of total budget")
        print(f"    Net return: {c1 * (mean_g1[1] - 1.0) * 100:+.2f}% + {c2 * (mean_g2[1] - 1.0) * 100:+.2f}% = {mean_net[1]:+.2f}%")
        print(f"    SD = {sd_net[1]:.2f}%, ace_hits={ace_hits[1]}/{cnt}, king_hits={king_hits[1]}/{cnt}")

        results.append({
            'n_sig': n_sig,
            'mean_g1': mean_g1.copy(),
            'mean_g2': mean_g2.copy(),
            'mean_net': mean_net.copy(),
            'sd_net': sd_net.copy(),
            'ace_hits': ace_hits.copy(),
            'cnt': cnt,
        })

    # Summary table
    print(f"\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'n_sig':<8} {'Conc mean%':<15} {'Conc SD%':<15} {'Div mean%':<15} {'Div SD%':<15}")
    print("─"*80)
    for r in results:
        if r is None:
            print(f"{'N/A':<8} {'---':<15} {'---':<15} {'---':<15} {'---':<15}")
            continue
        print(f"{r['n_sig']:<8} {r['mean_net'][0]:<15.2f} {r['sd_net'][0]:<15.2f} {r['mean_net'][1]:<15.2f} {r['sd_net'][1]:<15.2f}")


def main():
    """
    Local test runner - shows 2 strategies across all n_sig.
    Uses EXACT frontier.py logic, just filtered to 2 strategies.
    """
    ap = argparse.ArgumentParser(description="Local frontier test - 2 strategies, all n_sig")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--rounds", type=int, default=100000, help="Number of rounds to simulate")
    ap.add_argument("--signal_cost", type=float, default=9.0, help="Cost per signal")
    ap.add_argument("--stage1_alloc", type=float, default=0.1, help="Stage 1 allocation (0.0 to 1.0)")
    ap.add_argument("--signal_type", type=str, default="median", choices=["median", "top2"], help="Signal type")
    ap.add_argument("--scale_pay", type=int, default=0, choices=[0, 1], help="Scale pay (0=Ace only, 1=scaled)")
    ap.add_argument("--post_npz", type=str, default=POST_NPZ_DEFAULT, help="Posterior NPZ file")

    args = ap.parse_args()

    run_local_test(
        seed=args.seed,
        rounds=args.rounds,
        signal_cost=args.signal_cost,
        stage1_alloc=args.stage1_alloc,
        signal_type=args.signal_type,
        scale_pay=args.scale_pay,
        post_npz_path=args.post_npz,
    )


if __name__ == "__main__":
    main()
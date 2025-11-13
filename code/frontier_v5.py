#!/usr/bin/env python3
"""
frontier_v5.py — Full enumeration WITHOUT posterior dependencies

KEY INSIGHT: Full enumeration doesn't need expected values or posteriors!
- We enumerate ALL weight combinations regardless of signals
- We only need realized payoffs (not expected values)
- No sorting, no reordering, no posterior lookups
- Weight position i always corresponds to pile i (consistent mapping)

This is 2-3x faster than v4_parallel since we eliminate:
- Posterior loading and lookups (~10-20% of runtime)
- EV computation per pile per round
- Sorting and reordering operations

What we KEEP:
1. Deal cards → get max_rank per pile
2. Compute realized payoffs from max_rank
3. Apply ALL weight combinations (full enumeration)
4. Enforce Stage 2 support constraint (in weight initialization)
5. Deduct signal cost from returns
6. Track premium card hits (ace/king/queen)
"""

import argparse
import json
import pathlib
from typing import List
import multiprocessing as mp
import time

import numpy as np
from numpy.random import default_rng

# Import shared functions and constants
from sim_res import round_seed, _deal_cards_global_deck
from fns import canonicalize_params, NUM_PILES, CARDS_PER_PILE, ACE_RANK, BUDGET

# -----------------------
# Constants / defaults
# -----------------------

ACE_PAYOUT = 20.0
SCALE_PARAM_ON = 0.25
# ALPHA_GRID = np.linspace(0.25, 1.0, 4)
ALPHA_GRID = [0.1, 0.25, 0.5 , 0.75, 1.0]
UNITS = 9
SD_STEP = 1  # 1 basis point

# -----------------------
# GLOBAL SHARED DATA
# -----------------------
_SHARED_WM1 = None
_SHARED_WM2_ALL = None
_SHARED_STRATEGY_MAP = None
_SHARED_STAGE1_INDICES = None


def _initialize_shared_weights():
    """Build weight matrices ONCE in parent process."""
    global _SHARED_WM1, _SHARED_WM2_ALL, _SHARED_STRATEGY_MAP, _SHARED_STAGE1_INDICES

    _SHARED_WM1 = _weight_splits(UNITS, NUM_PILES)
    Ns1 = _SHARED_WM1.shape[0]

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
    _SHARED_STAGE1_INDICES = np.array([i1 for i1, _ in strategy_map], dtype=int)

    Ns_total = len(strategy_map)
    return Ns1, Ns_total


# -----------------------
# Helpers
# -----------------------

def _hvals(scale_pay: int, scale_param: float, ace_payout: float, half: bool = False) -> np.ndarray:
    """Compute payout values per rank (for EV computation - NOT USED in v5)."""
    ranks_all = np.arange(2, 15, dtype=int)
    if scale_pay == 0:
        v = np.array([float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
    else:
        v = np.array([float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)
    if half:
        v = 0.5 * v
    return v


def _per_dollar_realized(max_rank: np.ndarray, scale_pay: int, scale_param: float, ace_payout: float) -> np.ndarray:
    """Compute REALIZED payoff per dollar based on actual max_rank."""
    r = np.asarray(max_rank, int)
    if scale_pay == 0:
        return np.where(r == ACE_RANK, float(ace_payout), 0.0).astype(float)
    steps = (ACE_RANK - r).clip(min=0)
    return (float(ace_payout) * (float(scale_param) ** steps)).astype(float)


def _weight_splits(units: int, m: int) -> np.ndarray:
    """Enumerate nonnegative integer compositions."""
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
    """Build Stage 2 weight vectors within support."""
    support_indices = np.flatnonzero(support_mask)
    m = len(support_indices)

    if m == 0:
        return np.zeros((1, NUM_PILES), float)

    weights_on_support = _weight_splits(units, m)
    n_combinations = weights_on_support.shape[0]
    full_weights = np.zeros((n_combinations, NUM_PILES), float)
    full_weights[:, support_indices] = weights_on_support

    return full_weights


def _worker_chunk_il(args):
    """Worker using SHARED weight matrices - NO POSTERIORS NEEDED!"""
    base_seed, round_start, rounds_chunk, n_sig, sp, scale_param = args

    # Access shared matrices (inherited via fork)
    Wm1 = _SHARED_WM1
    Wm2_all = _SHARED_WM2_ALL
    stage1_indices = _SHARED_STAGE1_INDICES
    Ns_total = len(_SHARED_STRATEGY_MAP)

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

    for i in range(rounds_chunk):
        r = int(round_start) + int(i)
        rng = default_rng(round_seed(int(base_seed), r))

        # Deal cards to get max_rank per pile
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        # Track premium cards for analysis
        has_king = np.array([13 in np.asarray(h, int) for h in hands], dtype=bool)
        has_queen = np.array([12 in np.asarray(h, int) for h in hands], dtype=bool)

        # Compute REALIZED payoffs (no EV needed!)
        p_real = _per_dollar_realized(np.asarray(max_rank, int), sp, scale_param, ACE_PAYOUT)
        p_real_stage2 = 0.5 * p_real  # Stage 2 gets 0.5x payoff

        # NO REORDERING! Weight position i = pile i (consistent mapping)
        # Stage 1 returns: all strategies
        g1_all = Wm1 @ p_real  # Shape: (Ns1,)

        # Stage 2 returns: all strategies (within support constraints)
        g2_vec = Wm2_all @ p_real_stage2  # Shape: (Ns_total,)

        # Expand g1 to match strategy pairs
        g1_expanded = g1_all[stage1_indices]  # Shape: (Ns_total,)

        # Accumulate statistics
        stats["sum_g1"] += g1_expanded
        stats["sum_g2"] += g2_vec
        stats["sum_g1_sq"] += g1_expanded * g1_expanded
        stats["sum_g2_sq"] += g2_vec * g2_vec
        stats["sum_g12"] += g1_expanded * g2_vec

        # Track premium card hits (which strategies invested in piles with ace/king/queen)
        # NO REORDERING - has_ace[i] corresponds to pile i
        ace_hit_mask_s1 = np.any((Wm1 > 0) & has_ace, axis=1)  # Shape: (Ns1,)
        king_hit_mask_s1 = np.any((Wm1 > 0) & has_king, axis=1)
        queen_hit_mask_s1 = np.any((Wm1 > 0) & has_queen, axis=1)

        # Expand to strategy pairs
        stats["ace_hits"] += ace_hit_mask_s1[stage1_indices].astype(int)
        stats["king_hits"] += king_hit_mask_s1[stage1_indices].astype(int)
        stats["queen_hits"] += queen_hit_mask_s1[stage1_indices].astype(int)

        stats["count"] += 1

    return stats


def _collect_stats_il(seed: int, rounds: int, procs: int, n_sig: int, sp: int, scale_param: float):
    """Collect stats using multiprocessing.Pool."""
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

        args_list = [
            (int(seed), int(start), int(sz), int(n_sig), int(sp), float(scale_param))
            for (start, sz) in starts
        ]

        with mp.Pool(processes=len(starts)) as pool:
            results = pool.map(_worker_chunk_il, args_list)

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
        args = (int(seed), 0, int(rounds), int(n_sig), int(sp), float(scale_param))
        return _worker_chunk_il(args)


def simulate_and_save_single_nsig(seed_int, rounds, n_sig, procs, params, stage1_alloc, out_dir: pathlib.Path):
    """Process SINGLE n_sig value and save separate NPZ (saves twice for web compatibility)."""

    print(f"[v5] Starting job: alpha={stage1_alloc:.2f}, scale_pay={params['scale_pay']}, n_sig={n_sig}")
    print(f"[v5] Rounds: {rounds}, Workers: {procs}")
    print(f"[v5] NO POSTERIORS - Full enumeration with realized payoffs only!")
    print(f"[v5] Will save BOTH median and top2 files (identical results)")

    sp = int(params["scale_pay"])
    scale_param = float(params["scale_param"]) if sp == 1 else 0.0

    # Validate affordability
    budget1 = float(stage1_alloc) * BUDGET
    signal_cost = float(params.get('signal_cost', 3.0))
    signal_cost_total = float(n_sig) * signal_cost

    if stage1_alloc <= 0 or budget1 < signal_cost_total:
        print(f"[v5] SKIPPED: Cannot afford {n_sig} signals with alpha={stage1_alloc:.2f}")
        return

    # Run simulation (NO posteriors loaded!)
    t_sim = time.time()
    print(f"[v5] Running {rounds} rounds...")
    stats = _collect_stats_il(
        seed=int(seed_int), rounds=int(rounds), procs=int(procs),
        n_sig=int(n_sig), sp=sp, scale_param=scale_param
    )
    print(f"[v5] Simulation complete in {time.time() - t_sim:.1f}s")

    # Compute bins
    t_bin = time.time()
    print(f"[v5] Computing bins...")

    g1 = stats["sum_g1"]
    g2 = stats["sum_g2"]
    g1sq = stats["sum_g1_sq"]
    g2sq = stats["sum_g2_sq"]
    g12 = stats["sum_g12"]
    ace_hits = stats["ace_hits"]
    king_hits = stats["king_hits"]
    queen_hits = stats["queen_hits"]
    cnt = int(stats["count"])

    if cnt <= 1 or g1.size == 0:
        print(f"[v5] No valid statistics, skipping save")
        return

    investable1 = max(0.0, budget1 - signal_cost_total)
    budget2 = max(0.0, BUDGET - budget1)
    c1 = investable1 / BUDGET
    c2 = budget2 / BUDGET if investable1 > 0 else 0.0

    mean_g1 = g1 / cnt
    mean_g2 = g2 / cnt
    var_g1 = g1sq / cnt - mean_g1 ** 2
    var_g2 = g2sq / cnt - mean_g2 ** 2
    cov_g12 = g12 / cnt - (mean_g1 * mean_g2)

    signal_cost_fraction = signal_cost_total / BUDGET
    mean_net = 100.0 * (c1 * (mean_g1 - 1.0) + c2 * (mean_g2 - 1.0) - signal_cost_fraction)
    var_net = (100.0 ** 2) * ((c1 ** 2) * np.clip(var_g1, 0, np.inf) + (c2 ** 2) * np.clip(var_g2, 0, np.inf) + 2.0 * c1 * c2 * cov_g12)
    sd_net = np.sqrt(np.clip(var_net, 0.0, np.inf))

    bins = np.floor(sd_net / SD_STEP).astype(int)
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
            i1, i2 = _SHARED_STRATEGY_MAP[sel]
            sd_levels.append(float(b) * SD_STEP)
            best_means.append(float(mean_net[sel]))
            best_weights_s1.append(_SHARED_WM1[i1])
            best_weights_s2.append(_SHARED_WM2_ALL[i2])
            best_ace_hits.append(int(ace_hits[sel]))
            best_king_hits.append(int(king_hits[sel]))
            best_queen_hits.append(int(queen_hits[sel]))

    print(f"[v5] Binning complete in {time.time() - t_bin:.1f}s, {len(sd_levels)} bins")

    # Save NPZ twice (median and top2) for web_game.py compatibility
    norm_params, key_tuple, key_id = canonicalize_params(params)
    a_tag = f"a{int(round(float(stage1_alloc)*100)):03d}"
    n_tag = f"n{n_sig:02d}"

    import tempfile, os
    out_dir.mkdir(parents=True, exist_ok=True)

    for signal_type in ("median", "top2"):
        out_path = out_dir / f"{key_id}_{signal_type}_{a_tag}_{n_tag}.npz"

        payload = dict(
            sd_step=float(SD_STEP),
            sd_levels=np.array(sd_levels, float),
            best_means=np.array(best_means, float),
            best_weights_s1=np.array(best_weights_s1, float),
            best_weights_s2=np.array(best_weights_s2, float),
            best_ace_hits=np.array(best_ace_hits, int),
            best_king_hits=np.array(best_king_hits, int),
            best_queen_hits=np.array(best_queen_hits, int),
            meta=json.dumps(dict(
                mode="frontier_il_v5_no_posteriors",
                stage1_alloc=float(stage1_alloc),
                signal_type=signal_type,  # For web compatibility only
                n_sig=int(n_sig),
                params=norm_params,
                total_rounds=int(rounds),
                units=int(UNITS),
                full_stage2_enumeration=True,
                memory_optimized=True,
                no_posterior_lookups=True,
                no_ev_sorting=True,
                note="median and top2 files contain identical data (no signal info used)",
            )),
        )

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

        print(f"[v5] Saved: {out_path.name}")

    print(f"[v5] Job COMPLETE!")


def main():
    """
    Parallelized frontier computation - ONE n_sig per job.

    NO POSTERIORS NEEDED - full enumeration uses only realized payoffs!
    Signal_type not needed since we don't use signal info - saves both median and top2 files.

    Array job mapping (50 total jobs):
    - Index 0-4: n_sig=0 for files 0-4 (5 alpha values)
    - Index 5-9: n_sig=1 for files 0-4
    - ...
    - Index 45-49: n_sig=9 for files 0-4
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--sweep_out", type=str, default="frontier_output_v5")
    ap.add_argument("--array_index", type=int, required=True, help="PBS array index (0-49): 5 files × 10 n_sig")

    args = ap.parse_args()

    # Build parameter combinations (5 files total - no signal_type since we don't use posteriors)
    SIGNAL_COSTS = [3]
    SCALE_PARAMS = [0.25]
    SCALE_PAYS = [1]
    ACE_PAYOUTS = [20.0]

    files = []
    for alpha in ALPHA_GRID:
        for sc in SIGNAL_COSTS:
            for sp in SCALE_PAYS:
                for s in SCALE_PARAMS:
                    for ap_val in ACE_PAYOUTS:
                        raw = dict(signal_cost=float(sc), scale_pay=sp,
                                   scale_param=(s if sp == 1 else 0.0), ace_payout=ap_val)
                        files.append((raw, alpha))

    # Map array index to (file_idx, n_sig)
    n_files = len(files)  # 5 alpha values = 5 files
    max_signals = 9
    n_sig = args.array_index // n_files
    file_idx = args.array_index % n_files

    if n_sig > max_signals:
        raise ValueError(f"Array index {args.array_index} out of range (max {n_files * (max_signals + 1) - 1})")

    raw, alpha = files[file_idx]

    out_dir = pathlib.Path(args.sweep_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize shared weights ONCE
    print(f"[v5] ========================================")
    print(f"[v5] Array index: {args.array_index} → file {file_idx}, n_sig {n_sig}")
    print(f"[v5] Building shared weight matrices...")
    t_init = time.time()
    Ns1, Ns_total = _initialize_shared_weights()
    print(f"[v5] Initialization complete in {time.time() - t_init:.1f}s")
    print(f"[v5] Stage 1 strategies: {Ns1:,}")
    print(f"[v5] Total strategy pairs: {Ns_total:,}")
    print(f"[v5] Memory: ~{Ns_total * NUM_PILES * 8 / 1e9:.2f} GB")
    print(f"[v5] ========================================\n")

    # Process single (file, n_sig) combination
    simulate_and_save_single_nsig(
        seed_int=int(args.seed), rounds=int(args.rounds), n_sig=int(n_sig),
        procs=int(args.procs), params=raw, stage1_alloc=float(alpha),
        out_dir=out_dir
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
frontier_v4_parallel.py — Parallelized across n_sig values (separate NPZ per n_sig)

Key change from v4:
- Each job processes ONE n_sig value for ONE file
- Outputs separate NPZ per (alpha, scale_pay, signal_type, n_sig)
- Enables 80 parallel array jobs (8 files × 10 n_sig) instead of 8 serial jobs

Array job structure:
- Job 0-7: n_sig=0 for 8 different parameter files
- Job 8-15: n_sig=1 for 8 different parameter files
- ...
- Job 72-79: n_sig=9 for 8 different parameter files
"""

import argparse
import json
import pathlib
from typing import Dict, List, Tuple
import multiprocessing as mp
import time

import numpy as np
from numpy.random import default_rng

# Import shared functions and constants
from sim_res import round_seed, _deal_cards_global_deck
from precomp import _second_highest_rank
from fns import canonicalize_params, NUM_PILES, CARDS_PER_PILE, ACE_RANK, BUDGET

# -----------------------
# Constants / defaults
# -----------------------

POST_NPZ_DEFAULT = "../precomp_output/post_mc.npz"

ACE_PAYOUT = 20.0
SCALE_PARAM_ON = 0.25
ALPHA_GRID = np.linspace(0.25, 1.0, 4)
UNITS = 9
SD_STEP = 1

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

def _load_posteriors(npz_path: str):
    """Load posteriors from NPZ file."""
    p = pathlib.Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"post_npz not found: {p}")

    with np.load(p, allow_pickle=False) as z:
        req = {
            "rmax_median_keys", "rmax_median_mat", "rmax_top2_keys", "rmax_top2_mat",
            "joint_median_keys", "joint_median_mat", "joint_top2_keys", "joint_top2_mat",
            "prior_rmax", "r2_marginal_mat"
        }
        missing = [k for k in req if k not in z.files]
        if missing:
            raise ValueError(f"post_npz missing arrays: {missing}")

        m_keys = np.asarray(z["rmax_median_keys"], int)
        m_mat = np.asarray(z["rmax_median_mat"], float)
        t_keys = np.asarray(z["rmax_top2_keys"], int)
        t_mat = np.asarray(z["rmax_top2_mat"], float)
        jm_keys = np.asarray(z["joint_median_keys"], int)
        jm_mat = np.asarray(z["joint_median_mat"], float)
        jt_keys = np.asarray(z["joint_top2_keys"], int)
        jt_mat = np.asarray(z["joint_top2_mat"], float)
        prior = np.asarray(z["prior_rmax"], float)
        r2_marg = np.asarray(z["r2_marginal_mat"], float)

    rmax_median = {int(k): np.array(m_mat[i], float) for i, k in enumerate(m_keys)}
    rmax_top2 = {int(k): np.array(t_mat[i], float) for i, k in enumerate(t_keys)}
    rmax_tables = {"median": rmax_median, "top2": rmax_top2}

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
    """Worker using SHARED weight matrices."""
    base_seed, round_start, rounds_chunk, signal_type, n_sig, sp, scale_param, rmax_tables, joint_tables, prior_rmax, r2_marginal = args

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

    h1 = _hvals(sp, scale_param, ACE_PAYOUT, half=False)
    h2 = _hvals(sp, scale_param, ACE_PAYOUT, half=True)
    post_table = rmax_tables[signal_type]

    for i in range(rounds_chunk):
        r = int(round_start) + int(i)
        rng = default_rng(round_seed(int(base_seed), r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
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
        p_real_stage2 = 0.5 * p_real

        top_idx = order1
        p_m = p_real[top_idx]
        p_m_stage2 = p_real_stage2[top_idx]

        g1_all = Wm1 @ p_m
        g2_vec = Wm2_all @ p_m_stage2
        g1_expanded = g1_all[stage1_indices]

        stats["sum_g1"] += g1_expanded
        stats["sum_g2"] += g2_vec
        stats["sum_g1_sq"] += g1_expanded * g1_expanded
        stats["sum_g2_sq"] += g2_vec * g2_vec
        stats["sum_g12"] += g1_expanded * g2_vec

        has_ace_ranked = has_ace[top_idx]
        has_king_ranked = has_king[top_idx]
        has_queen_ranked = has_queen[top_idx]
        ace_hit_mask_s1 = np.any((Wm1 > 0) & has_ace_ranked, axis=1)
        king_hit_mask_s1 = np.any((Wm1 > 0) & has_king_ranked, axis=1)
        queen_hit_mask_s1 = np.any((Wm1 > 0) & has_queen_ranked, axis=1)

        stats["ace_hits"] += ace_hit_mask_s1[stage1_indices].astype(int)
        stats["king_hits"] += king_hit_mask_s1[stage1_indices].astype(int)
        stats["queen_hits"] += queen_hit_mask_s1[stage1_indices].astype(int)

        stats["count"] += 1

    return stats


def _collect_stats_il(seed: int, rounds: int, procs: int, signal_type: str, n_sig: int, sp: int, scale_param: float,
                      rmax_tables, joint_tables, prior_rmax, r2_marginal):
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
            (int(seed), int(start), int(sz), signal_type, int(n_sig), int(sp), float(scale_param),
             rmax_tables, joint_tables, prior_rmax, r2_marginal)
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
        args = (int(seed), 0, int(rounds), signal_type, int(n_sig), int(sp), float(scale_param),
                rmax_tables, joint_tables, prior_rmax, r2_marginal)
        return _worker_chunk_il(args)


def simulate_and_save_single_nsig(seed_int, rounds, n_sig, procs, params, stage1_alloc, out_dir: pathlib.Path, signal_type: str):
    """Process SINGLE n_sig value and save separate NPZ."""

    print(f"[v4p] Starting job: alpha={stage1_alloc:.2f}, scale_pay={params['scale_pay']}, signal_type={signal_type}, n_sig={n_sig}")
    print(f"[v4p] Rounds: {rounds}, Workers: {procs}")

    # Load posteriors
    t_load = time.time()
    rmax_tables, joint_tables, prior_rmax, r2_marginal = _load_posteriors(POST_NPZ_DEFAULT)
    print(f"[v4p] Loaded posteriors in {time.time() - t_load:.1f}s")

    sp = int(params["scale_pay"])
    scale_param = float(params["scale_param"]) if sp == 1 else 0.0
    st = str(signal_type)

    # Validate affordability
    budget1 = float(stage1_alloc) * BUDGET
    signal_cost = float(params.get('signal_cost', 3.0))
    signal_cost_total = float(n_sig) * signal_cost

    if stage1_alloc <= 0 or budget1 < signal_cost_total:
        print(f"[v4p] SKIPPED: Cannot afford {n_sig} signals with alpha={stage1_alloc:.2f}")
        return

    # Run simulation
    t_sim = time.time()
    print(f"[v4p] Running {rounds} rounds...")
    stats = _collect_stats_il(
        seed=int(seed_int), rounds=int(rounds), procs=int(procs), signal_type=st, n_sig=int(n_sig),
        sp=sp, scale_param=scale_param, rmax_tables=rmax_tables, joint_tables=joint_tables,
        prior_rmax=prior_rmax, r2_marginal=r2_marginal,
    )
    print(f"[v4p] Simulation complete in {time.time() - t_sim:.1f}s")

    # Compute bins
    t_bin = time.time()
    print(f"[v4p] Computing bins...")

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
        print(f"[v4p] No valid statistics, skipping save")
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

    print(f"[v4p] Binning complete in {time.time() - t_bin:.1f}s, {len(sd_levels)} bins")

    # Save NPZ
    norm_params, key_tuple, key_id = canonicalize_params(params)
    a_tag = f"a{int(round(float(stage1_alloc)*100)):03d}"
    n_tag = f"n{n_sig:02d}"
    out_path = out_dir / f"{key_id}_{st}_{a_tag}_{n_tag}.npz"

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
            mode="frontier_il_v4_parallel",
            stage1_alloc=float(stage1_alloc),
            signal_type=st,
            n_sig=int(n_sig),
            params=norm_params,
            total_rounds=int(rounds),
            units=int(UNITS),
            full_stage2_enumeration=True,
            memory_optimized=True,
        )),
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

    print(f"[v4p] Saved: {out_path.name}")
    print(f"[v4p] Job COMPLETE!")


def main():
    """
    Parallelized frontier computation - ONE n_sig per job.

    Array job index maps to (file_idx, n_sig):
    - Index 0-7: n_sig=0 for files 0-7
    - Index 8-15: n_sig=1 for files 0-7
    - ...
    - Index 72-79: n_sig=9 for files 0-7
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--sweep_out", type=str, default="frontier_output_v4_parallel")
    ap.add_argument("--array_index", type=int, required=True, help="PBS array index (0-79)")

    args = ap.parse_args()

    # Build parameter combinations (8 files total)
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
                        for st in ("median", "top2"):
                            files.append((raw, alpha, st))

    # Map array index to (file_idx, n_sig)
    n_files = len(files)
    max_signals = 9
    n_sig = args.array_index // n_files
    file_idx = args.array_index % n_files

    if n_sig > max_signals:
        raise ValueError(f"Array index {args.array_index} out of range (max {n_files * (max_signals + 1) - 1})")

    raw, alpha, st = files[file_idx]

    out_dir = pathlib.Path(args.sweep_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize shared weights ONCE
    print(f"[v4p] ========================================")
    print(f"[v4p] Array index: {args.array_index} → file {file_idx}, n_sig {n_sig}")
    print(f"[v4p] Building shared weight matrices...")
    t_init = time.time()
    Ns1, Ns_total = _initialize_shared_weights()
    print(f"[v4p] Initialization complete in {time.time() - t_init:.1f}s")
    print(f"[v4p] Stage 1 strategies: {Ns1:,}")
    print(f"[v4p] Total strategy pairs: {Ns_total:,}")
    print(f"[v4p] Memory: ~{Ns_total * NUM_PILES * 8 / 1e9:.2f} GB")
    print(f"[v4p] ========================================\n")

    # Process single (file, n_sig) combination
    simulate_and_save_single_nsig(
        seed_int=int(args.seed), rounds=int(args.rounds), n_sig=int(n_sig),
        procs=int(args.procs), params=raw, stage1_alloc=float(alpha),
        out_dir=out_dir, signal_type=st
    )


if __name__ == "__main__":
    main()

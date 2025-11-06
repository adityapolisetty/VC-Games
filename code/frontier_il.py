#!/usr/bin/env python3
"""
frontier_il.py — Information‑limited (IL) mean‑variance frontier

Brute‑force within‑stage allocations at 0.05 granularity (20 units) using
posterior expectations and Stage‑2 updating, with Stage‑2 restricted to the
Stage‑1 support. Stage‑1/Stage‑2 budget split α is swept on a 0.1 grid.

Scope (fixed params per request)
- signal_type ∈ {median, top2}
- scale_pay ∈ {0, 1}; when 1, scale_param=0.25; signal_cost=3; ace_payout=20
- stage1_alloc α ∈ {0.0, 0.1, …, 1.0}
- within‑stage weights use support size m ∈ {1, 2, 3} over top‑m piles by Stage‑1
  expected value; weights on those m piles split in 0.05 increments (20 units).
  Stage‑2 reassigns Stage‑1 weights within support according to updated Stage‑2
  expected values (no renormalization needed — it’s a permutation within support).

Outputs
- Prints, for each (scale_pay, signal_type, n_sig, alpha), the number of
  nondominated (mean, sd) points on the IL frontier and the best mean.
- Optionally saves a compact NPZ per regime with arrays:
  alpha_grid, support_sizes, weights_m (ragged via object arrays),
  mean_grid[alpha, idx], sd_grid[alpha, idx], and a boolean mask frontier[alpha, idx].
"""

import argparse
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
from numpy.random import default_rng


# -----------------------
# Constants / defaults
# -----------------------
NUM_PILES = 9
CARDS_PER_PILE = 5
ACE_RANK = 14
BUDGET = 100.0

# Posterior NPZs (same defaults as dynamic code, relative to this file)
POST_NPZ_DEFAULT = "../output/post_mc.npz"
POST_NPZ_JOINT_DEFAULT = "../output_joint/post_joint.npz"

# Fixed params per request
SIGNAL_COST = 3.0
ACE_PAYOUT = 20.0
SCALE_PARAM_ON = 0.25
ALPHA_GRID = np.linspace(0.0, 1.0, 11)  # 0.0..1.0 step 0.1
UNITS = 20  # 0.05 granularity
MAX_SUPPORT = 3  # support size m ∈ {1,2,3}


# -----------------------
# Helpers (synced with dynamic code)
# -----------------------
def round_seed(base_seed: int, r: int) -> int:
    import zlib

    s = f"{int(base_seed)}|round|{int(r)}".encode("utf-8")
    return int(np.uint32(zlib.adler32(s)))


def _deal_cards_global_deck(rng):
    special_cards = np.array([ACE_RANK, 13, 13, 12, 12], dtype=int)
    if special_cards.size:
        rng.shuffle(special_cards)
    hands = [[] for _ in range(NUM_PILES)]
    for card in special_cards:
        available = [i for i in range(NUM_PILES) if len(hands[i]) < CARDS_PER_PILE]
        if not available:
            break
        pile_idx = int(rng.choice(available))
        hands[pile_idx].append(int(card))
    pool = np.repeat(np.arange(2, 12, dtype=int), 4)
    for i in range(NUM_PILES):
        need = CARDS_PER_PILE - len(hands[i])
        if need > 0:
            if need > pool.shape[0]:
                need = int(pool.shape[0])
            idx = rng.choice(pool.shape[0], size=need, replace=False)
            draw = pool[idx]
            hands[i].extend(draw.tolist())
            pool = np.delete(pool, idx)
    has_ace = np.zeros(NUM_PILES, dtype=bool)
    medians = np.empty(NUM_PILES, dtype=int)
    top2sum = np.empty(NUM_PILES, dtype=int)
    max_rank = np.empty(NUM_PILES, dtype=int)
    for i in range(NUM_PILES):
        arr = np.array(sorted(hands[i]), dtype=int)
        has_ace[i] = bool(np.any(arr == ACE_RANK))
        medians[i] = int(arr[CARDS_PER_PILE // 2])
        top2sum[i] = int(arr[-1] + arr[-2])
        max_rank[i] = int(arr[-1])
    return has_ace, [np.array(sorted(h), int) for h in hands], medians, top2sum, max_rank


def _second_highest_rank(pile: np.ndarray) -> int:
    arr = np.sort(np.asarray(pile, int))
    return int(arr[-2]) if arr.size >= 2 else int(arr[-1])


def _load_mc_posteriors(npz_path: str):
    p = pathlib.Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"post_npz not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        req = {"rmax_median_keys", "rmax_median_mat", "rmax_top2_keys", "rmax_top2_mat", "prior_rmax"}
        missing = [k for k in req if k not in z.files]
        if missing:
            raise ValueError(f"post_npz missing arrays: {missing}")
        m_keys = np.asarray(z["rmax_median_keys"], int)
        m_mat = np.asarray(z["rmax_median_mat"], float)
        t_keys = np.asarray(z["rmax_top2_keys"], int)
        t_mat = np.asarray(z["rmax_top2_mat"], float)
        prior = np.asarray(z["prior_rmax"], float)
    rmax_median = {int(k): np.array(m_mat[i], float) for i, k in enumerate(m_keys)}
    rmax_top2 = {int(k): np.array(t_mat[i], float) for i, k in enumerate(t_keys)}
    return rmax_median, rmax_top2, prior


def _load_joint_posteriors(npz_path: str):
    p = pathlib.Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"joint post_npz not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        req = {"joint_median_keys", "joint_median_mat", "joint_top2_keys", "joint_top2_mat", "prior_rmax", "r2_marginal_mat"}
        missing = [k for k in req if k not in z.files]
        if missing:
            raise ValueError(f"joint NPZ missing arrays: {missing}")
        jm_keys = np.asarray(z["joint_median_keys"], int)
        jm_mat = np.asarray(z["joint_median_mat"], float)
        jt_keys = np.asarray(z["joint_top2_keys"], int)
        jt_mat = np.asarray(z["joint_top2_mat"], float)
        prior = np.asarray(z["prior_rmax"], float)
        r2_marg = np.asarray(z["r2_marginal_mat"], float)
    def _rowmap(keys: np.ndarray):
        return {int(k): int(i) for i, k in enumerate(keys.tolist())}
    joint_tables = {
        "median": (jm_keys, jm_mat, _rowmap(jm_keys)),
        "top2": (jt_keys, jt_mat, _rowmap(jt_keys)),
    }
    return joint_tables, prior, r2_marg


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


def _pareto_frontier(mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    # return boolean mask of nondominated points maximizing mean and minimizing sd
    idx = np.lexsort((sd, -mean))  # sort by -mean then sd
    best_sd = np.inf
    keep = np.zeros_like(mean, dtype=bool)
    for j in idx:
        if sd[j] < best_sd - 1e-12:
            keep[j] = True
            best_sd = sd[j]
    return keep


def simulate_il_frontier(seed: int, rounds: int, max_signals: int):
    rmax_median, rmax_top2, prior_mc = _load_mc_posteriors(POST_NPZ_DEFAULT)
    joint_tables, prior_joint, r2_marginal = _load_joint_posteriors(POST_NPZ_JOINT_DEFAULT)

    # two regimes
    signal_types = ["median", "top2"]
    scale_pays = [0, 1]

    # Precompute weight splits for m=1..MAX_SUPPORT
    W_by_m: Dict[int, np.ndarray] = {m: _weight_splits(UNITS, m) for m in range(1, MAX_SUPPORT + 1)}

    # Iterate regimes
    for sp in scale_pays:
        scale_param = SCALE_PARAM_ON if int(sp) == 1 else 0.0
        h1 = _hvals(sp, scale_param, ACE_PAYOUT, half=False)
        h2 = _hvals(sp, scale_param, ACE_PAYOUT, half=True)
        for st in signal_types:
            post_table = rmax_median if st == "median" else rmax_top2
            keys, mat3d, rowmap = joint_tables[st]
            print(f"IL frontier: scale_pay={sp}, signal_type={st}")
            for n_sig in range(int(max_signals) + 1):
                # Accumulators for each m and split index
                stats = {}
                for m, Wm in W_by_m.items():
                    Ns = Wm.shape[0]
                    stats[m] = dict(
                        sum_g1=np.zeros(Ns, float),
                        sum_g2=np.zeros(Ns, float),
                        sum_g1_sq=np.zeros(Ns, float),
                        sum_g2_sq=np.zeros(Ns, float),
                        sum_g12=np.zeros(Ns, float),
                        count=0,
                    )

                for r in range(int(rounds)):
                    rng = default_rng(round_seed(int(seed), int(r)))
                    _, hands, medians, top2sum, max_rank = _deal_cards_global_deck(rng)
                    pi = rng.permutation(NUM_PILES)
                    chosen_idx = pi[:n_sig]
                    chosen_set = set(int(x) for x in np.asarray(chosen_idx, int))

                    # Stage‑1 expected per‑dollar by posterior
                    buckets = np.asarray(medians if st == "median" else top2sum, int)
                    prior_vec = np.asarray(prior_joint if isinstance(prior_joint, np.ndarray) else prior_mc, float)
                    s1 = np.zeros(NUM_PILES, float)
                    for i in range(NUM_PILES):
                        vec = np.asarray(post_table.get(int(buckets[i]), prior_vec), float) if (i in chosen_set) else prior_vec
                        s1[i] = float(np.dot(h1, vec))
                    order1 = np.argsort(-s1)

                    # Stage‑2 expected per‑dollar on support uses joint or r2 marginal
                    R2 = np.array([_second_highest_rank(h) for h in hands], int)

                    # Realized per‑dollar on this board
                    p_real = _per_dollar_realized(np.asarray(max_rank, int), sp, scale_param, ACE_PAYOUT)

                    for m, Wm in W_by_m.items():
                        top_idx = order1[:m]
                        p_m = p_real[top_idx]
                        # s2 only needed for ordering within support
                        s2_m = np.zeros(m, float)
                        for jj, i in enumerate(top_idx):
                            r2k = int(R2[i]) - 2
                            if i in chosen_set:
                                b = int(buckets[i])
                                if (b in rowmap) and (0 <= r2k < 13):
                                    vec = np.asarray(mat3d[rowmap[b], r2k, :], float)
                                else:
                                    vec = prior_vec
                            else:
                                vec = np.asarray(r2_marginal[r2k, :], float) if (0 <= r2k < 13) else prior_vec
                            s2_m[jj] = float(np.dot(h2, vec))
                        perm2 = np.argsort(-s2_m)

                        # g1 = Wm @ p_m ; g2 = (Wm permuted by perm2) @ p_m
                        g1 = Wm @ p_m
                        g2 = Wm[:, perm2] @ p_m
                        st_m = stats[m]
                        st_m["sum_g1"] += g1
                        st_m["sum_g2"] += g2
                        st_m["sum_g1_sq"] += g1 * g1
                        st_m["sum_g2_sq"] += g2 * g2
                        st_m["sum_g12"] += g1 * g2
                        st_m["count"] += 1

                # For each alpha, compute mean/sd for all splits across m and report frontier
                for a in ALPHA_GRID:
                    info_cost = float(n_sig) * SIGNAL_COST
                    budget1 = float(a) * BUDGET
                    investable1 = max(0.0, budget1 - info_cost)
                    budget2 = max(0.0, BUDGET - budget1)
                    c1 = investable1 / BUDGET
                    c2 = 0.5 * budget2 / BUDGET
                    mu_all = []
                    sd_all = []
                    for m, Wm in W_by_m.items():
                        st_m = stats[m]
                        cnt = max(1, int(st_m["count"]))
                        mean_g1 = st_m["sum_g1"] / cnt
                        mean_g2 = st_m["sum_g2"] / cnt
                        var_g1 = (st_m["sum_g1_sq"] / cnt) - (mean_g1 ** 2)
                        var_g2 = (st_m["sum_g2_sq"] / cnt) - (mean_g2 ** 2)
                        cov_g12 = (st_m["sum_g12"] / cnt) - (mean_g1 * mean_g2)
                        mean_net = 100.0 * (c1 * mean_g1 + c2 * mean_g2 - 1.0)
                        var_net = (100.0 ** 2) * (
                            (c1 ** 2) * np.clip(var_g1, 0.0, np.inf)
                            + (c2 ** 2) * np.clip(var_g2, 0.0, np.inf)
                            + 2.0 * c1 * c2 * cov_g12
                        )
                        mu_all.append(mean_net)
                        sd_all.append(np.sqrt(np.clip(var_net, 0.0, np.inf)))
                    mu_cat = np.concatenate(mu_all)
                    sd_cat = np.concatenate(sd_all)
                    keep = _pareto_frontier(mu_cat, sd_cat)
                    best_mean = float(np.max(mu_cat)) if mu_cat.size else float("nan")
                    n_front = int(np.count_nonzero(keep))
                    print(f"  n={n_sig:02d} alpha={a:.1f} frontier={n_front} best_mean={best_mean:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--max_signals", type=int, default=9)
    ap.add_argument("--procs", type=int, default=1)  # unused (serial)
    args = ap.parse_args()

    simulate_il_frontier(seed=int(args.seed), rounds=int(args.rounds), max_signals=int(args.max_signals))


if __name__ == "__main__":
    main()


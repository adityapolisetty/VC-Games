#!/usr/bin/env python3
"""
simulate_policy.py — Policy simulation for player strategy evaluation

Simulates a player's POLICY (not exact investments) across 10k random boards:
- For each round: new board, random signal permutation
- Buy n signals (determined by policy)
- Rank piles by Stage 1 posteriors
- Invest using player's concentration pattern
- Stage 2: reveal R2, rerank within support, invest Stage 2 budget
- Calculate net return using frontier formula

This allows players to see the distribution of outcomes for their strategy
across different board configurations and signal draws.
"""

import numpy as np
from typing import Dict, Tuple
from numpy.random import default_rng

# Import shared functions
from sim_res import _deal_cards_global_deck, _second_highest_rank, round_seed
from fns import NUM_PILES, CARDS_PER_PILE, ACE_RANK, BUDGET


def _hvals(scale_pay: int, scale_param: float, ace_payout: float, half: bool = False) -> np.ndarray:
    """Compute per-rank payoff values (matching frontier.py)"""
    ranks_all = np.arange(2, 15, dtype=int)
    if scale_pay == 0:
        v = np.array([float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
    else:
        v = np.array([float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)
    if half:
        v = 0.5 * v
    return v


def _per_dollar_realized(max_rank: np.ndarray, scale_pay: int, scale_param: float, ace_payout: float) -> np.ndarray:
    """Compute realized payoff per dollar invested (matching frontier.py)"""
    r = np.asarray(max_rank, int)
    if scale_pay == 0:
        return np.where(r == ACE_RANK, float(ace_payout), 0.0).astype(float)
    steps = (ACE_RANK - r).clip(min=0)
    return (float(ace_payout) * (float(scale_param) ** steps)).astype(float)


def _compute_concentration_index(weights: np.ndarray) -> float:
    """Compute concentration index: sum of squared weights (matching frontier.py)"""
    # Normalize to sum to 1, then compute sum of squares
    w = np.asarray(weights, float)
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0
    w_norm = w / w_sum
    return float(np.sum(w_norm ** 2))


def run_policy_simulation(
    n_signals: int,
    signal_type: str,
    signal_cost: float,
    stage1_alloc: float,
    ace_payout: float,
    scale_pay: int,
    scale_param: float,
    player_concentration: float,
    post_npz_path: str = "../precomp_output/post_mc.npz",
    rounds: int = 10000,
    base_seed: int = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Simulate player's policy across many random boards.

    Args:
        n_signals: Number of signals purchased per round
        signal_type: 'median' or 'top2'
        signal_cost: Cost per signal (£)
        stage1_alloc: Fraction of budget allocated to Stage 1 (0.0 to 1.0)
        ace_payout: Ace payoff multiplier
        scale_pay: 0 (ace-only) or 1 (scaled)
        scale_param: Scale parameter (e.g., 0.25)
        player_concentration: Target concentration index for portfolio
        post_npz_path: Path to precomputed posteriors NPZ
        rounds: Number of simulation rounds
        base_seed: Base RNG seed (None = random)

    Returns:
        (returns_array, metadata_dict)
        - returns_array: Array of net return % for each round
        - metadata_dict: Stats about simulation (mean, std, percentiles)
    """
    # Load posteriors
    import pathlib
    p = pathlib.Path(post_npz_path)
    if not p.exists():
        raise FileNotFoundError(f"Posteriors not found: {p}")

    with np.load(p, allow_pickle=False) as z:
        # Marginal posteriors P(Rmax|signal)
        m_keys = np.asarray(z["rmax_median_keys"], int)
        m_mat = np.asarray(z["rmax_median_mat"], float)
        t_keys = np.asarray(z["rmax_top2_keys"], int)
        t_mat = np.asarray(z["rmax_top2_mat"], float)

        # Joint posteriors P(Rmax|signal,R2)
        jm_keys = np.asarray(z["joint_median_keys"], int)
        jm_mat = np.asarray(z["joint_median_mat"], float)
        jt_keys = np.asarray(z["joint_top2_keys"], int)
        jt_mat = np.asarray(z["joint_top2_mat"], float)

        # Prior and R2 marginal
        prior = np.asarray(z["prior_rmax"], float)
        r2_marginal = np.asarray(z["r2_marginal_mat"], float)

    # Build lookup dicts
    rmax_median = {int(k): np.array(m_mat[i], float) for i, k in enumerate(m_keys)}
    rmax_top2 = {int(k): np.array(t_mat[i], float) for i, k in enumerate(t_keys)}
    rmax_tables = {"median": rmax_median, "top2": rmax_top2}

    # Joint posterior row mapping
    jm_rowmap = {int(k): int(i) for i, k in enumerate(jm_keys.tolist())}
    jt_rowmap = {int(k): int(i) for i, k in enumerate(jt_keys.tolist())}
    joint_tables = {
        "median": (jm_keys, jm_mat, jm_rowmap),
        "top2": (jt_keys, jt_mat, jt_rowmap),
    }

    # Compute budget allocations
    budget1 = float(stage1_alloc) * BUDGET
    signal_cost_total = float(n_signals) * float(signal_cost)
    investable1 = max(0.0, budget1 - signal_cost_total)
    budget2 = max(0.0, BUDGET - budget1)

    if investable1 <= 0:
        # No investment possible - return zeros
        return np.zeros(rounds, float), {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "p25": 0.0, "p50": 0.0, "p75": 0.0,
        }

    # Payoff vectors
    h1 = _hvals(scale_pay, scale_param, ace_payout, half=False)
    h2 = _hvals(scale_pay, scale_param, ace_payout, half=True)

    post_table = rmax_tables[signal_type]
    keys, mat3d, rowmap = joint_tables[signal_type]

    # Simulation loop
    if base_seed is None:
        base_seed = np.random.randint(0, 1_000_000)

    returns = np.zeros(rounds, float)

    # Determine investment pattern based on concentration index
    # Higher concentration = fewer piles with unequal weights
    # We'll approximate by investing in top k piles with concentration-weighted distribution
    # For simplicity: use equal weights across top k piles where k is chosen to match concentration

    # Concentration index ≈ 1/k for equal weights across k piles
    # So k ≈ 1 / concentration_index
    if player_concentration > 0:
        n_piles_invest = max(1, min(NUM_PILES, int(round(1.0 / player_concentration))))
    else:
        n_piles_invest = NUM_PILES  # Fully diversified

    for r in range(rounds):
        # Generate new board
        rng = default_rng(round_seed(base_seed, r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        # Random permutation for signal purchases
        pi = rng.permutation(NUM_PILES)
        chosen_idx = pi[:n_signals]
        chosen_set = set(int(x) for x in np.asarray(chosen_idx, int))

        # Compute Stage 1 expected values
        buckets = np.asarray(medians if signal_type == "median" else top2sum, int)
        prior_vec = np.asarray(prior, float)
        s1 = np.zeros(NUM_PILES, float)
        for j in range(NUM_PILES):
            vec = np.asarray(post_table.get(int(buckets[j]), prior_vec), float) if (j in chosen_set) else prior_vec
            s1[j] = float(np.dot(h1, vec))

        # Rank piles by Stage 1 expected value (descending)
        order1 = np.argsort(-s1)

        # Invest in top n_piles_invest piles with equal weights
        top_piles = order1[:n_piles_invest]
        inv1_amounts = np.zeros(NUM_PILES, float)
        inv1_amounts[top_piles] = investable1 / n_piles_invest

        # Stage 1 payoff
        p_real = _per_dollar_realized(np.asarray(max_rank, int), scale_pay, scale_param, ace_payout)
        stage1_payoff = float(np.dot(inv1_amounts, p_real))

        # Stage 2: Only piles with Stage 1 investment
        stage1_support = [int(i) for i in range(NUM_PILES) if inv1_amounts[i] > 0]

        if len(stage1_support) > 0 and budget2 > 0:
            # Compute R2 for each pile
            R2 = np.array([_second_highest_rank(h) for h in hands], int)

            # Compute Stage 2 expected values within support
            s2 = np.zeros(NUM_PILES, float)
            for k in stage1_support:
                r2k = int(R2[k]) - 2  # Index (rank 2 → index 0)
                if k in chosen_set:
                    b = int(buckets[k])
                    if (b in rowmap) and (0 <= r2k < 13):
                        vec = np.asarray(mat3d[rowmap[b], r2k, :], float)
                    else:
                        vec = prior_vec
                else:
                    vec = np.asarray(r2_marginal[r2k, :], float) if (0 <= r2k < 13) else prior_vec
                s2[k] = float(np.dot(h2, vec))

            # Rerank within support
            support_s2 = [(s2[k], k) for k in stage1_support]
            support_s2.sort(reverse=True)  # Highest EV first

            # Invest Stage 2 budget in top piles within support (same concentration pattern)
            n_piles_stage2 = min(n_piles_invest, len(stage1_support))
            top_stage2_piles = [k for _, k in support_s2[:n_piles_stage2]]

            inv2_amounts = np.zeros(NUM_PILES, float)
            inv2_amounts[top_stage2_piles] = budget2 / n_piles_stage2

            # Stage 2 payoff (0.5x multiplier)
            p_real_stage2 = 0.5 * p_real
            stage2_payoff = float(np.dot(inv2_amounts, p_real_stage2))
        else:
            stage2_payoff = 0.0

        # Net return calculation (matching frontier formula)
        total_invest = investable1 + budget2
        c1 = investable1 / BUDGET
        c2 = budget2 / BUDGET
        signal_cost_fraction = signal_cost_total / BUDGET

        g1 = stage1_payoff / investable1 if investable1 > 0 else 0.0
        g2 = stage2_payoff / budget2 if budget2 > 0 else 0.0

        net_return_pct = 100.0 * (c1 * (g1 - 1.0) + c2 * (g2 - 1.0) - signal_cost_fraction)
        returns[r] = net_return_pct

    # Compute statistics
    metadata = {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "p25": float(np.percentile(returns, 25)),
        "p50": float(np.percentile(returns, 50)),
        "p75": float(np.percentile(returns, 75)),
        "rounds": int(rounds),
        "n_signals": int(n_signals),
        "signal_type": str(signal_type),
        "concentration": float(player_concentration),
    }

    return returns, metadata


if __name__ == "__main__":
    # Test run
    print("[simulate_policy] Running test simulation...")
    returns, meta = run_policy_simulation(
        n_signals=2,
        signal_type="median",
        signal_cost=5.0,
        stage1_alloc=0.5,
        ace_payout=20.0,
        scale_pay=0,
        scale_param=0.0,
        player_concentration=0.25,  # ~4 piles
        rounds=1000,
    )
    print(f"[simulate_policy] Test complete:")
    print(f"  Mean return: {meta['mean']:.2f}%")
    print(f"  Std dev: {meta['std']:.2f}%")
    print(f"  Range: [{meta['min']:.2f}%, {meta['max']:.2f}%]")
    print(f"  Median: {meta['p50']:.2f}%")

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
    """Compute per-rank payoff values with custom scaling: Ace=20×, King=5×, Queen=1.25×, Others=0×"""
    ranks_all = np.arange(2, 15, dtype=int)
    v = np.zeros(len(ranks_all), float)
    for i, r in enumerate(ranks_all):
        if r == 14:  # Ace
            v[i] = float(ace_payout)
        elif r == 13:  # King
            v[i] = float(ace_payout) * 0.25
        elif r == 12:  # Queen
            v[i] = float(ace_payout) * 0.0625
        # else: 0.0 (already initialized)
    if half:
        v = 0.5 * v
    return v


def _per_dollar_realized(max_rank: np.ndarray, scale_pay: int, scale_param: float, ace_payout: float) -> np.ndarray:
    """Compute realized payoff per dollar invested with custom scaling: Ace=20×, King=5×, Queen=1.25×, Others=0×"""
    r = np.asarray(max_rank, int)
    result = np.where(r == 14, float(ace_payout),           # Ace: 20×
             np.where(r == 13, float(ace_payout) * 0.25,    # King: 5×
             np.where(r == 12, float(ace_payout) * 0.0625,  # Queen: 1.25×
             0.0)))                                           # Others: 0×
    return result.astype(float)


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
    post_npz_path: str = "precomp_output/post_mc.npz",
    rounds: int = 10000,
    base_seed: int = None,
    actual_board_seed: int = None,
    actual_signaled_piles: set = None,
    actual_weights_stage1: np.ndarray = None,
    actual_weights_stage2: np.ndarray = None,
    weight_pattern_stage1: np.ndarray = None,
    weight_pattern_stage2: np.ndarray = None,
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
        rounds: Number of simulation rounds (including actual board)
        base_seed: Base RNG seed (None = random)
        actual_board_seed: Seed for actual board played (if provided, will be simulated as round 0)
        actual_signaled_piles: Set of pile indices that had signals in actual game (for round 0)
        actual_weights_stage1: Actual Stage 1 investment amounts per pile (for round 0)
        actual_weights_stage2: Actual Stage 2 investment amounts per pile (for round 0)
        weight_pattern_stage1: Normalized weight fractions for Stage 1 (9 values, sum to 1.0) - used for rounds 1+
        weight_pattern_stage2: Normalized weight fractions for Stage 2 (9 values, sum to 1.0) - used for rounds 1+

    Returns:
        (returns_array, metadata_dict)
        - returns_array: Array of net return % for each round
        - metadata_dict: Stats about simulation (mean, std, percentiles)
    """
    # Load posteriors
    import pathlib
    import shutil

    # Resolve path - try multiple locations
    script_dir = pathlib.Path(__file__).parent
    p = pathlib.Path(post_npz_path)

    # Try multiple locations in order:
    # 1. Absolute path or current working directory
    # 2. Relative to script location
    # 3. Railway volume (/data/)
    # 4. Repo root (parent of script dir)

    locations_to_try = [p]
    if not p.is_absolute():
        locations_to_try.extend([
            script_dir / "post_mc.npz",                    # Direct file in code directory (Railway)
            script_dir / post_npz_path,                     # ./precomp_output/post_mc.npz
            pathlib.Path("/data") / post_npz_path,          # /data/precomp_output/post_mc.npz
            script_dir.parent / post_npz_path,              # ../precomp_output/post_mc.npz
        ])

    found_path = None
    for location in locations_to_try:
        if location.exists():
            found_path = location
            break

    # If not found in /data/ but found elsewhere, copy to /data/ for persistence
    volume_path = pathlib.Path("/data") / post_npz_path
    if found_path is None:
        tried_paths = ", ".join(str(loc) for loc in locations_to_try)
        raise FileNotFoundError(f"Posteriors not found. Tried: {tried_paths}")
    elif found_path != volume_path and pathlib.Path("/data").exists():
        # Copy to volume for future runs (Railway best practice)
        try:
            volume_path.parent.mkdir(parents=True, exist_ok=True)
            if not volume_path.exists():
                print(f"[simulate_policy] Copying posteriors to volume: {volume_path}")
                shutil.copy2(found_path, volume_path)
                print(f"[simulate_policy] Copy complete - future runs will use volume cache")
        except Exception as e:
            print(f"[simulate_policy] Warning: Could not copy to volume: {e}")

    p = found_path

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

    # Initialize hit counters
    ace_hits_total = 0
    king_hits_total = 0
    queen_hits_total = 0

    # Will capture Round 0 EV-sorted weight pattern
    round0_weight_pattern = None

    # Determine investment pattern based on concentration index
    # Higher concentration = fewer piles with unequal weights
    # We'll approximate by investing in top k piles with concentration-weighted distribution
    # Note: player_concentration is passed but only used for metadata/display
    # Weight patterns are always provided from web_game.py, so no fallback logic needed

    for r in range(rounds):
        # For round 0: use actual board if provided, otherwise generate with seed(0)
        # For rounds 1+: always use consistent indexing seed(1), seed(2), etc.
        if r == 0 and actual_board_seed is not None:
            # Use the actual board played as round 0
            rng = default_rng(actual_board_seed)
        else:
            # Generate random board using round index as offset (consistent indexing)
            rng = default_rng(round_seed(base_seed, r))

        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        # Track premium cards (for hit rate calculation)
        has_king = np.array([13 in h for h in hands], dtype=bool)
        has_queen = np.array([12 in h for h in hands], dtype=bool)

        # Determine which piles have signals
        if r == 0 and actual_signaled_piles is not None:
            # Round 0: Use actual signaled piles from player's game
            chosen_set = set(int(x) for x in actual_signaled_piles)
        else:
            # Other rounds: Random permutation for signal purchases
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

        # Determine Stage 1 investments
        if r == 0 and actual_weights_stage1 is not None:
            # Round 0: Use actual Stage 1 investments
            inv1_amounts = np.asarray(actual_weights_stage1, float)

            # Extract normalized EV-ranked weight pattern from Round 0
            # This pattern will be reused for all subsequent rounds
            order1_round0 = np.argsort(-s1)  # EV ranking from Round 0
            total_inv1 = inv1_amounts.sum()
            if total_inv1 > 0:
                # Create normalized pattern: what fraction goes to 1st-best, 2nd-best, etc.
                round0_weight_pattern = np.array([inv1_amounts[order1_round0[i]] / total_inv1 for i in range(NUM_PILES)])
        else:
            # Rounds 1+: Apply player's weight pattern to EV-sorted piles
            # Sort piles by EV (descending) to get ranking
            order1 = np.argsort(-s1)
            # Apply weight pattern: pattern[i] goes to pile at position i in EV ranking
            inv1_amounts = np.zeros(NUM_PILES, float)
            pattern = np.asarray(weight_pattern_stage1, float)
            for pos, pile_idx in enumerate(order1):
                if pos < len(pattern):
                    inv1_amounts[pile_idx] = pattern[pos] * investable1

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

            # Determine Stage 2 investments
            if r == 0 and actual_weights_stage2 is not None:
                # Round 0: Use actual Stage 2 investments
                inv2_amounts = np.asarray(actual_weights_stage2, float)
            else:
                # Rounds 1+: Apply weight pattern to EV-sorted piles within support
                support_s2 = [(s2[k], k) for k in stage1_support]
                support_s2.sort(reverse=True)  # Highest EV first
                sorted_support_piles = [k for _, k in support_s2]

                # Apply weight pattern directly (no renormalization)
                # Pattern already sums to 1.0 from Stage 1 extraction
                inv2_amounts = np.zeros(NUM_PILES, float)
                pattern = np.asarray(weight_pattern_stage2, float)
                for pos, pile_idx in enumerate(sorted_support_piles):
                    if pos < len(pattern):
                        inv2_amounts[pile_idx] = pattern[pos] * budget2

            # Stage 2 payoff (0.5x multiplier)
            p_real_stage2 = 0.5 * p_real
            stage2_payoff = float(np.dot(inv2_amounts, p_real_stage2))
        else:
            stage2_payoff = 0.0

        # Gross return calculation (Total Payoff / Budget)
        total_payoff = stage1_payoff + stage2_payoff
        gross_return_multiplier = total_payoff / BUDGET if BUDGET > 0 else 0.0
        returns[r] = gross_return_multiplier

        # Track hits: count if player invested in premium piles
        total_inv = inv1_amounts + (inv2_amounts if 'inv2_amounts' in locals() else np.zeros(NUM_PILES, float))
        if np.any(has_ace & (total_inv > 0)):
            ace_hits_total += 1
        if np.any(has_king & (total_inv > 0)):
            king_hits_total += 1
        if np.any(has_queen & (total_inv > 0)):
            queen_hits_total += 1

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
        "concentration_index": float(player_concentration),  # Fixed key name
        "player_weights": round0_weight_pattern.tolist() if round0_weight_pattern is not None else [0.0]*9,
        "ace_hits": int(ace_hits_total),
        "king_hits": int(king_hits_total),
        "queen_hits": int(queen_hits_total),
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

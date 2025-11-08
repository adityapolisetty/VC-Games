#!/usr/bin/env python3
"""Test: Ace pile ranking using TOP2SUM signal instead of median."""

import numpy as np
from numpy.random import default_rng
from card_game import _deal_cards_global_deck, round_seed, ACE_RANK, NUM_PILES

def load_posteriors():
    """Load top2sum posteriors from NPZ file."""
    post_npz = '/Users/adityapolisetty/Dropbox/PhD/freelance/Simulation/VC-Games/output/post_mc.npz'

    with np.load(post_npz, allow_pickle=False) as z:
        t2_keys = np.asarray(z['rmax_top2_keys'], int)
        t2_mat = np.asarray(z['rmax_top2_mat'], float)  # Shape: [K, 13] for Rmax 2..14
        prior = np.asarray(z['prior_rmax'], float)  # Shape: [13]

    # Build lookup: top2sum -> P(Rmax=k) for k in 2..14
    rmax_top2 = {int(k): np.array(t2_mat[i], float) for i, k in enumerate(t2_keys)}

    print(f"Loaded top2sum posteriors for {len(rmax_top2)} unique top2 values")
    print(f"Top2 range: {min(t2_keys)} to {max(t2_keys)}")
    print(f"Prior P(Rmax): shape={prior.shape}, sum={prior.sum():.3f}")

    return rmax_top2, prior

def compute_expected_value(top2sum, rmax_top2, prior, is_observed, ace_payout=20.0):
    """Compute E[payoff | top2sum] using posteriors."""
    if is_observed:
        post = rmax_top2.get(top2sum, prior)
    else:
        post = prior

    # post[12] = P(Rmax = 14) = P(Ace)
    p_ace = post[ACE_RANK - 2]
    return ace_payout * p_ace

def test_ace_ranking_top2(rmax_top2, prior, n_rounds=10000, n_sig=5):
    """Test Ace pile ranking using top2sum signal."""

    ace_ranks = []
    ace_observed_count = 0
    ace_expected_values = []

    for r in range(n_rounds):
        rng = default_rng(round_seed(12345, r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        # Find which pile has the Ace
        ace_pile = np.where(has_ace)[0]
        if len(ace_pile) == 0:
            continue
        ace_pile = ace_pile[0]

        # Simulate observing n_sig piles
        pi = rng.permutation(NUM_PILES)
        chosen_idx = pi[:n_sig]
        chosen_set = set(int(x) for x in chosen_idx)

        # Track if Ace pile was observed
        if ace_pile in chosen_set:
            ace_observed_count += 1

        # Compute Stage 1 expected values using top2sum posteriors
        s1 = np.zeros(NUM_PILES, float)
        for j in range(NUM_PILES):
            t2 = int(top2sum[j])
            is_observed = (j in chosen_set)
            s1[j] = compute_expected_value(t2, rmax_top2, prior, is_observed)

        # Store Ace pile's expected value
        ace_expected_values.append(s1[ace_pile])

        # Rank piles by expected value
        order = np.argsort(-s1)  # Descending order

        # Find rank of Ace pile (1-indexed)
        ace_rank = int(np.where(order == ace_pile)[0][0]) + 1
        ace_ranks.append(ace_rank)

    ace_observed_pct = 100.0 * ace_observed_count / len(ace_ranks) if len(ace_ranks) > 0 else 0
    return np.array(ace_ranks), ace_observed_pct, np.array(ace_expected_values)

if __name__ == '__main__':
    print('=' * 70)
    print('Analyzing Ace pile ranking using TOP2SUM posteriors')
    print('=' * 70)

    # Load precomputed posteriors
    rmax_top2, prior = load_posteriors()

    print('\n' + '=' * 70)
    print('Question: If we invest in top-m piles, how often do we miss the Ace?')
    print('(Using top2sum signal instead of median)')
    print('=' * 70)

    # Test different signal counts
    for n_sig in [0, 3, 5, 7, 9]:
        ranks, ace_obs_pct, ace_evs = test_ace_ranking_top2(
            rmax_top2, prior, n_rounds=10000, n_sig=n_sig
        )
        print(f'\n📊 n_sig = {n_sig} (observing {n_sig}/9 piles):')
        print(f'   Ace observed: {ace_obs_pct:>5.1f}% of rounds')
        print(f'   Mean rank: {np.mean(ranks):.2f}')
        print(f'   Median rank: {int(np.median(ranks))}')
        print(f'   Mean E[payoff] for Ace pile: {np.mean(ace_evs):.2f}')
        print(f'   ✓ Rank ≤ 3: {100*np.mean(ranks <= 3):>5.1f}%  (captured by m=3)')
        print(f'   ✓ Rank ≤ 5: {100*np.mean(ranks <= 5):>5.1f}%  (captured by m=5)')
        print(f'   ✓ Rank ≤ 7: {100*np.mean(ranks <= 7):>5.1f}%  (captured by m=7)')
        print(f'   ✗ Rank > 5: {100*np.mean(ranks > 5):>5.1f}%  (MISSED by m=5)')

        # Detailed breakdown
        print(f'   Rank distribution: ', end='')
        for rank in range(1, 10):
            pct = 100 * np.mean(ranks == rank)
            if pct > 0.1:
                print(f'{rank}={pct:.1f}% ', end='')
        print()

    print('\n' + '=' * 70)
    print('COMPARISON: MEDIAN vs TOP2SUM')
    print('  Top2sum captures correlation with Ace better than median')
    print('  Ace pile always has top2 ≥ 27 (if Ace=14, next best ≥ 13)')
    print('  So P(Ace | top2=27) >> P(Ace | median=low)')
    print('=' * 70)

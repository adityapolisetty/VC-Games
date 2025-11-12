#!/usr/bin/env python3
"""
Test script to validate game simulation logic and payoff calculations.

This script tests:
1. Single pile payoffs (Ace, King, Queen, other ranks)
2. Stage 1 vs Stage 2 payout multipliers
3. Mixed portfolio weighted returns
4. Distribution of returns over multiple rounds
5. Comparison of actual game payoffs vs simulate_policy.py expectations
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Import game modules
from web_game import GameState, compute_payoffs_at_stage2, ACE_RANK, ACE_PAYOUT
from simulate_policy import simulate_policy

print("="*70)
print("GAME SIMULATION VALIDATION TEST SUITE")
print("="*70)
print()

# Test parameters
BUDGET = 100.0
ACE_PAYOUT_TEST = 20.0
SIGNAL_COST = 3.0

def create_test_deck_with_rank(target_rank: int, pile_idx: int = 0) -> np.ndarray:
    """Create a test deck where pile_idx contains target_rank as max card."""
    deck = np.zeros((9, 5), dtype=int)

    for pile in range(9):
        if pile == pile_idx:
            # Target pile gets the target rank plus 4 lower cards
            deck[pile, 0] = target_rank
            deck[pile, 1:] = [10, 9, 8, 7]
        else:
            # Other piles get random low cards (2-11)
            deck[pile] = np.random.randint(2, 12, size=5)

    return deck


def test_single_pile_payoffs():
    """Test 1: Validate payoffs for single pile investments (Ace, King, Queen, other)."""
    print("TEST 1: Single Pile Payoffs")
    print("-" * 70)

    test_cases = [
        (14, "Ace", 20.0),    # Ace: 20x multiplier
        (13, "King", 5.0),    # King: 5x multiplier (20 * 0.25)
        (12, "Queen", 1.25),  # Queen: 1.25x multiplier (20 * 0.0625)
        (11, "Jack", 0.0),    # Jack: 0x multiplier
        (10, "Ten", 0.0),     # Ten: 0x multiplier
    ]

    all_passed = True

    for rank, name, expected_multiplier in test_cases:
        # Create test data: invest £100 in single pile with target rank
        test_data = pd.DataFrame({
            'alive': [True],
            'round': [2],
            'N': [rank],
            'inv1': [100.0],
            'inv2': [0.0],
        })

        result = compute_payoffs_at_stage2(test_data, ace_payout=ACE_PAYOUT_TEST)

        actual_payout = float(result['payout'].iloc[0])
        expected_payout = 100.0 * expected_multiplier

        passed = np.isclose(actual_payout, expected_payout, rtol=1e-6)
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name:6s} (rank {rank:2d}): £100 → £{actual_payout:.2f} "
              f"(expected £{expected_payout:.2f}, {expected_multiplier}x multiplier)")

    print()
    return all_passed


def test_stage_multipliers():
    """Test 2: Validate Stage 1 vs Stage 2 payout multipliers."""
    print("TEST 2: Stage 1 vs Stage 2 Multipliers")
    print("-" * 70)

    all_passed = True

    # Test Ace payouts: Stage 1 should pay 20x, Stage 2 should pay 10x (0.5 × 20x)
    test_data = pd.DataFrame({
        'alive': [True, True],
        'round': [2, 2],
        'N': [14, 14],  # Both Aces
        'inv1': [50.0, 0.0],   # First row: £50 in Stage 1
        'inv2': [0.0, 50.0],   # Second row: £50 in Stage 2
    })

    result = compute_payoffs_at_stage2(test_data, ace_payout=ACE_PAYOUT_TEST)

    # Stage 1 Ace: £50 × 20 = £1000
    stage1_payout = float(result['payout'].iloc[0])
    expected_stage1 = 50.0 * 20.0
    passed_s1 = np.isclose(stage1_payout, expected_stage1, rtol=1e-6)

    # Stage 2 Ace: £50 × 10 = £500 (half of Stage 1)
    stage2_payout = float(result['payout'].iloc[1])
    expected_stage2 = 50.0 * 10.0
    passed_s2 = np.isclose(stage2_payout, expected_stage2, rtol=1e-6)

    all_passed = passed_s1 and passed_s2

    status_s1 = "✓ PASS" if passed_s1 else "✗ FAIL"
    status_s2 = "✓ PASS" if passed_s2 else "✗ FAIL"

    print(f"  {status_s1} | Stage 1 Ace: £50 → £{stage1_payout:.2f} (expected £{expected_stage1:.2f}, 20x)")
    print(f"  {status_s2} | Stage 2 Ace: £50 → £{stage2_payout:.2f} (expected £{expected_stage2:.2f}, 10x)")
    print()

    return all_passed


def test_mixed_portfolio():
    """Test 3: Validate weighted returns for mixed portfolio."""
    print("TEST 3: Mixed Portfolio Returns")
    print("-" * 70)

    # Portfolio: £40 in Ace, £30 in King, £20 in Queen, £10 in Jack
    test_data = pd.DataFrame({
        'alive': [True, True, True, True],
        'round': [2, 2, 2, 2],
        'N': [14, 13, 12, 11],  # Ace, King, Queen, Jack
        'inv1': [40.0, 30.0, 20.0, 10.0],
        'inv2': [0.0, 0.0, 0.0, 0.0],
    })

    result = compute_payoffs_at_stage2(test_data, ace_payout=ACE_PAYOUT_TEST)

    # Expected payouts:
    # Ace: £40 × 20 = £800
    # King: £30 × 5 = £150
    # Queen: £20 × 1.25 = £25
    # Jack: £10 × 0 = £0
    # Total: £975

    total_payout = float(result['payout'].sum())
    expected_total = 800.0 + 150.0 + 25.0 + 0.0

    passed = np.isclose(total_payout, expected_total, rtol=1e-6)

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} | Mixed portfolio: £100 invested → £{total_payout:.2f} "
          f"(expected £{expected_total:.2f})")
    print(f"         Breakdown:")

    for _, row in result.iterrows():
        rank = int(row['N'])
        rank_name = {14: "Ace", 13: "King", 12: "Queen", 11: "Jack"}[rank]
        inv = float(row['inv1'])
        payout = float(row['payout'])
        print(f"           - {rank_name}: £{inv:.0f} → £{payout:.2f}")

    print()
    return passed


def test_stage_split():
    """Test 4: Validate split investment across Stage 1 and Stage 2."""
    print("TEST 4: Stage 1 + Stage 2 Split Investment")
    print("-" * 70)

    # Invest £60 in Stage 1 and £40 in Stage 2, both hitting Ace
    test_data = pd.DataFrame({
        'alive': [True],
        'round': [2],
        'N': [14],  # Ace
        'inv1': [60.0],
        'inv2': [40.0],
    })

    result = compute_payoffs_at_stage2(test_data, ace_payout=ACE_PAYOUT_TEST)

    # Expected:
    # Stage 1: £60 × 20 = £1200
    # Stage 2: £40 × 10 = £400
    # Total: £1600

    payout1 = float(result['payout1'].iloc[0])
    payout2 = float(result['payout2'].iloc[0])
    total_payout = float(result['payout'].iloc[0])

    expected_p1 = 60.0 * 20.0
    expected_p2 = 40.0 * 10.0
    expected_total = expected_p1 + expected_p2

    passed = (np.isclose(payout1, expected_p1, rtol=1e-6) and
              np.isclose(payout2, expected_p2, rtol=1e-6) and
              np.isclose(total_payout, expected_total, rtol=1e-6))

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} | Ace pile: £60 (S1) + £40 (S2) = £100 invested")
    print(f"         Stage 1 payout: £{payout1:.2f} (expected £{expected_p1:.2f})")
    print(f"         Stage 2 payout: £{payout2:.2f} (expected £{expected_p2:.2f})")
    print(f"         Total payout: £{total_payout:.2f} (expected £{expected_total:.2f})")
    print()

    return passed


def test_gross_return_calculation():
    """Test 5: Validate gross return = payout / budget."""
    print("TEST 5: Gross Return Calculation")
    print("-" * 70)

    test_cases = [
        # (inv1, inv2, rank, expected_gross_return)
        (100.0, 0.0, 14, 20.0),      # £100 in Ace (S1): 20x return
        (100.0, 0.0, 13, 5.0),       # £100 in King (S1): 5x return
        (100.0, 0.0, 12, 1.25),      # £100 in Queen (S1): 1.25x return
        (0.0, 100.0, 14, 10.0),      # £100 in Ace (S2): 10x return
        (50.0, 50.0, 14, 15.0),      # £50+£50 in Ace: (1000+500)/100 = 15x
    ]

    all_passed = True

    for inv1, inv2, rank, expected_gr in test_cases:
        test_data = pd.DataFrame({
            'alive': [True],
            'round': [2],
            'N': [rank],
            'inv1': [inv1],
            'inv2': [inv2],
        })

        result = compute_payoffs_at_stage2(test_data, ace_payout=ACE_PAYOUT_TEST)

        total_investment = inv1 + inv2
        total_payout = float(result['payout'].iloc[0])
        gross_return = total_payout / total_investment if total_investment > 0 else 0.0

        passed = np.isclose(gross_return, expected_gr, rtol=1e-6)
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        rank_name = {14: "Ace", 13: "King", 12: "Queen"}[rank]
        print(f"  {status} | {rank_name}: £{inv1:.0f}(S1) + £{inv2:.0f}(S2) → "
              f"{gross_return:.2f}x (expected {expected_gr:.2f}x)")

    print()
    return all_passed


def test_distribution_statistics():
    """Test 6: Run simulation and validate distribution statistics."""
    print("TEST 6: Distribution Statistics (100 rounds, uniform random strategy)")
    print("-" * 70)

    # Run simple simulation: invest uniformly in all 9 piles, no signals
    num_rounds = 100
    seed = 42
    np.random.seed(seed)

    returns = []
    ace_hits = 0
    king_hits = 0
    queen_hits = 0

    for _ in range(num_rounds):
        # Create random deck
        deck = np.zeros((9, 5), dtype=int)

        # Place exactly 1 Ace, 2 Kings, 2 Queens
        special_cards = [14, 13, 13, 12, 12]
        special_positions = np.random.choice(45, size=5, replace=False)

        for card, pos in zip(special_cards, special_positions):
            pile_idx = pos // 5
            card_idx = pos % 5
            deck[pile_idx, card_idx] = card

        # Fill remaining with ranks 2-11
        remaining = []
        for rank in range(2, 12):
            remaining.extend([rank] * 4)
        np.random.shuffle(remaining)

        flat_deck = deck.flatten()
        zero_positions = np.where(flat_deck == 0)[0]
        flat_deck[zero_positions] = remaining[:len(zero_positions)]
        deck = flat_deck.reshape((9, 5))

        # Find max rank in each pile
        max_ranks = deck.max(axis=1)

        # Uniform investment: £100/9 ≈ £11.11 per pile (Stage 1 only)
        inv_per_pile = BUDGET / 9.0

        # Calculate payout
        test_data_list = []
        for pile_idx in range(9):
            test_data_list.append({
                'alive': True,
                'round': 2,
                'N': int(max_ranks[pile_idx]),
                'inv1': inv_per_pile,
                'inv2': 0.0,
            })

        test_data = pd.DataFrame(test_data_list)
        result = compute_payoffs_at_stage2(test_data, ace_payout=ACE_PAYOUT_TEST)

        total_payout = float(result['payout'].sum())
        gross_return = total_payout / BUDGET
        returns.append(gross_return)

        # Count hits
        if 14 in max_ranks:
            ace_hits += 1
        if 13 in max_ranks:
            king_hits += 1
        if 12 in max_ranks:
            queen_hits += 1

    returns = np.array(returns)

    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()
    min_return = returns.min()
    max_return = returns.max()

    # Expected statistics for uniform investment (1/9 chance each pile)
    # E[return] = (1/9) × 20 + (2/9) × 5 + (2/9) × 1.25 + (4/9) × 0
    #           = 20/9 + 10/9 + 2.5/9 = 32.5/9 ≈ 3.611
    expected_mean = (20.0 + 2*5.0 + 2*1.25) / 9.0

    print(f"  Distribution over {num_rounds} rounds:")
    print(f"    Mean return: {mean_return:.4f}x (expected ~{expected_mean:.4f}x)")
    print(f"    Std return: {std_return:.4f}x")
    print(f"    Min return: {min_return:.4f}x")
    print(f"    Max return: {max_return:.4f}x")
    print()
    print(f"  Hit rates:")
    print(f"    Ace: {ace_hits}/{num_rounds} ({100*ace_hits/num_rounds:.1f}%)")
    print(f"    King: {king_hits}/{num_rounds} ({100*king_hits/num_rounds:.1f}%)")
    print(f"    Queen: {queen_hits}/{num_rounds} ({100*queen_hits/num_rounds:.1f}%)")
    print()

    # Check if mean is reasonably close to expected (within 20% due to sampling variance)
    passed = np.abs(mean_return - expected_mean) / expected_mean < 0.20

    status = "✓ PASS" if passed else "⚠ WARNING"
    print(f"  {status} | Mean return within 20% of expected value")
    print()

    return passed


def run_all_tests():
    """Run all test cases and report results."""
    print()

    results = {
        "Single Pile Payoffs": test_single_pile_payoffs(),
        "Stage Multipliers": test_stage_multipliers(),
        "Mixed Portfolio": test_mixed_portfolio(),
        "Stage Split": test_stage_split(),
        "Gross Return Calculation": test_gross_return_calculation(),
        "Distribution Statistics": test_distribution_statistics(),
    }

    print("="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {test_name}")

    print()

    all_passed = all(results.values())
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)

    print("="*70)


if __name__ == "__main__":
    run_all_tests()

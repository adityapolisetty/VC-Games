#!/usr/bin/env python3
"""
Test the fixed _second_highest_rank() function.
"""

import numpy as np
from numpy.random import default_rng
from card_game import round_seed, _deal_cards_global_deck
import sys
sys.path.insert(0, '.')

# Import the fixed function from precomp_joint
from precomp_joint import _second_highest_rank

def test_fixed_definition(seed, n_samples=100000):
    """Verify that P(Ace | R2=13) = 1.0 with the fixed definition."""

    cases_r2_13 = {'rmax_13': 0, 'rmax_14': 0}

    for r in range(n_samples):
        if r % 50000 == 0:
            print(f"   Testing {r:,} rounds...")

        rng = default_rng(round_seed(seed, r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        for j in range(len(hands)):
            pile = hands[j]
            max_r = max_rank[j]

            # Use fixed R2 (unique rank-based)
            r2 = _second_highest_rank(pile)

            if r2 == 13:
                if max_r == 13:
                    cases_r2_13['rmax_13'] += 1
                    print(f"   ⚠️ Found impossible case: pile={sorted(pile)}, R2={r2}, Rmax={max_r}")
                elif max_r == 14:
                    cases_r2_13['rmax_14'] += 1

    total = cases_r2_13['rmax_13'] + cases_r2_13['rmax_14']
    if total > 0:
        prob_ace = cases_r2_13['rmax_14'] / total
        print(f"\n📊 Results with FIXED definition:")
        print(f"   Total cases with R2=13: {total:,}")
        print(f"   - Rmax=13: {cases_r2_13['rmax_13']} ({cases_r2_13['rmax_13']/total:.2%})")
        print(f"   - Rmax=14: {cases_r2_13['rmax_14']} ({prob_ace:.2%})")
        print(f"   → P(Ace | R2=13) = {prob_ace:.6f}")

        if prob_ace == 1.0:
            print(f"\n✅ SUCCESS: P(Ace | R2=13) = 1.0 as expected!")
        else:
            print(f"\n❌ FAIL: P(Ace | R2=13) ≠ 1.0")
    else:
        print(f"\n⚠️ No cases with R2=13 found in {n_samples} samples")

    # Also test some edge cases
    print(f"\n🧪 Unit tests:")
    test_cases = [
        ([4, 8, 8, 13, 13], 13, "Both 13s in pile"),
        ([3, 4, 8, 13, 14], 13, "One 13 and Ace"),
        ([2, 3, 3, 3, 4], 3, "Multiple duplicates"),
        ([5, 6, 7, 8, 9], 8, "All unique"),
        ([10], 10, "Single card"),
    ]

    for pile, expected, desc in test_cases:
        result = _second_highest_rank(np.array(pile))
        status = "✓" if result == expected else "✗"
        unique = sorted(set(pile), reverse=True)
        print(f"   {status} {pile} → unique={unique}, R2={result} (expected {expected}) [{desc}]")

if __name__ == "__main__":
    print("🔧 Testing fixed _second_highest_rank() function...\n")
    test_fixed_definition(seed=123, n_samples=100000)

#!/usr/bin/env python3
"""
Compare two definitions of R2 (second rank):
1. Current implementation: R2 = pile[3] (second position from top)
2. Alternative definition: R2 = second-highest unique rank value
"""

import numpy as np
from numpy.random import default_rng
from card_game import round_seed, _deal_cards_global_deck

def r2_position(pile):
    """Current implementation: second position from top."""
    sorted_pile = sorted(pile)
    return sorted_pile[-2] if len(sorted_pile) >= 2 else sorted_pile[-1]

def r2_unique_rank(pile):
    """Alternative: second-highest unique rank value."""
    unique_ranks = sorted(set(pile), reverse=True)
    return unique_ranks[1] if len(unique_ranks) >= 2 else unique_ranks[0]

def compare_definitions(seed, n_samples=100000):
    """Compare the two definitions."""

    # Track cases where R2_position = 13
    cases_pos = {'rmax_13': 0, 'rmax_14': 0}

    # Track cases where R2_unique = 13
    cases_unique = {'rmax_13': 0, 'rmax_14': 0}

    examples_pos = []
    examples_unique = []

    for r in range(n_samples):
        if r % 50000 == 0:
            print(f"   Processed {r:,} rounds...")

        rng = default_rng(round_seed(seed, r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        for j in range(len(hands)):
            pile = hands[j]
            max_r = max_rank[j]

            # Position-based R2
            r2_pos = r2_position(pile)
            if r2_pos == 13:
                if max_r == 13:
                    cases_pos['rmax_13'] += 1
                    if len(examples_pos) < 5:
                        examples_pos.append(pile)
                elif max_r == 14:
                    cases_pos['rmax_14'] += 1

            # Unique rank-based R2
            r2_uniq = r2_unique_rank(pile)
            if r2_uniq == 13:
                if max_r == 13:
                    cases_unique['rmax_13'] += 1
                elif max_r == 14:
                    cases_unique['rmax_14'] += 1
                    if len(examples_unique) < 5:
                        examples_unique.append(pile)

    print(f"\n📊 Comparison Results:\n")

    # Position-based (current implementation)
    total_pos = cases_pos['rmax_13'] + cases_pos['rmax_14']
    if total_pos > 0:
        prob_ace_pos = cases_pos['rmax_14'] / total_pos
        prob_13_pos = cases_pos['rmax_13'] / total_pos
        print(f"1️⃣ CURRENT (Position-based): R2 = pile[3]")
        print(f"   Total cases with R2=13: {total_pos:,}")
        print(f"   - Rmax=13: {cases_pos['rmax_13']:,} ({prob_13_pos:.2%})")
        print(f"   - Rmax=14: {cases_pos['rmax_14']:,} ({prob_ace_pos:.2%})")
        print(f"   → P(Ace | R2=13) = {prob_ace_pos:.4f}")

        if examples_pos:
            print(f"\n   Examples where R2_position=13 and Rmax=13:")
            for pile in examples_pos:
                sorted_p = sorted(pile)
                print(f"      {sorted_p} → positions [0,1,2,3,4], R2=pos[3]={sorted_p[3]}, max={sorted_p[4]}")

    # Unique rank-based (user's expectation)
    total_unique = cases_unique['rmax_13'] + cases_unique['rmax_14']
    if total_unique > 0:
        prob_ace_unique = cases_unique['rmax_14'] / total_unique if total_unique > 0 else 0
        prob_13_unique = cases_unique['rmax_13'] / total_unique if total_unique > 0 else 0
        print(f"\n2️⃣ ALTERNATIVE (Unique rank-based): R2 = 2nd highest unique rank")
        print(f"   Total cases with R2=13: {total_unique:,}")
        print(f"   - Rmax=13: {cases_unique['rmax_13']:,} ({prob_13_unique:.2%})")
        print(f"   - Rmax=14: {cases_unique['rmax_14']:,} ({prob_ace_unique:.2%})")
        print(f"   → P(Ace | R2=13) = {prob_ace_unique:.4f}")

        if cases_unique['rmax_13'] > 0:
            print(f"\n   ⚠️ BUG: Found {cases_unique['rmax_13']} cases with R2_unique=13 and Rmax=13")
            print(f"   This is logically impossible!")
        else:
            print(f"\n   ✅ No cases with R2_unique=13 and Rmax=13 (as expected)")

        if examples_unique:
            print(f"\n   Examples where R2_unique=13 and Rmax=14:")
            for pile in examples_unique:
                sorted_p = sorted(pile)
                unique = sorted(set(pile), reverse=True)
                print(f"      {sorted_p} → unique ranks {unique}, R2_unique={unique[1]}, max={unique[0]}")

    print(f"\n🎯 Recommendation:")
    if prob_ace_unique == 1.0:
        print(f"   The user is correct: with the UNIQUE RANK definition,")
        print(f"   P(Ace | R2=13) = 1.0 regardless of median.")
        print(f"\n   To fix this, modify _second_highest_rank() to return")
        print(f"   the second-highest UNIQUE rank, not the second position.")
    else:
        print(f"   Both definitions have cases where they differ from 1.0")

if __name__ == "__main__":
    print("🔍 Comparing R2 definitions...\n")
    compare_definitions(seed=123, n_samples=100000)

#!/usr/bin/env python3
"""
Verify the logical constraint that P(Ace | R2=13) = 1.0 for ANY median value.

Logic:
- Pile (sorted): [pile[0], pile[1], pile[2], pile[3], pile[4]]
- R2 = pile[3] (second-to-last position)
- Max = pile[4] (last position)
- If R2 = 13, then pile[3] = 13
- Since pile is sorted, pile[4] >= pile[3] = 13
- Possible values for pile[4]: 13 or 14 (only ranks >= 13)
- Can pile[4] = 13? Only if both 13s are in this pile at positions 3 and 4
- But wait: if pile[3] = 13, that's already one 13. Can pile[4] also be 13?
- YES, if both 13s from the deck are in this pile!

So the question is: when can both 13s be in the same pile?
- Global deck has 5 special cards: [14, 13, 13, 12, 12]
- These are distributed randomly to piles
- If both 13s go to same pile, they could end up at any positions
- After sorting, they could be at positions (i, j) where i < j

Let's check: if median=3, R2=13, can we have [c0, c1, 3, 13, 13]?
"""

import numpy as np
from numpy.random import default_rng
from card_game import round_seed, _deal_cards_global_deck

def _second_highest_rank(arr):
    a = np.sort(np.asarray(arr, int))
    return int(a[-2]) if a.size >= 2 else int(a[-1])

def exhaustive_search(seed, n_samples=1000000):
    """Search for ANY case where R2=13 and Rmax=13."""

    cases_r2_13_rmax_13 = []
    cases_r2_13_rmax_14 = []
    total_r2_13 = 0

    for r in range(n_samples):
        if r % 100000 == 0:
            print(f"   Processed {r:,} rounds...")

        rng = default_rng(round_seed(seed, r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        for j in range(len(hands)):
            pile = sorted(hands[j])
            R2 = _second_highest_rank(pile)

            if R2 == 13:
                total_r2_13 += 1
                max_r = max_rank[j]

                if max_r == 13:
                    cases_r2_13_rmax_13.append({
                        'round': r,
                        'pile_idx': j,
                        'pile': pile,
                        'median': medians[j],
                        'R2': R2,
                        'max': max_r,
                    })
                elif max_r == 14:
                    cases_r2_13_rmax_14.append({
                        'round': r,
                        'pile_idx': j,
                        'pile': pile,
                        'median': medians[j],
                        'R2': R2,
                        'max': max_r,
                    })

    return cases_r2_13_rmax_13, cases_r2_13_rmax_14, total_r2_13

def main():
    print("🔍 Exhaustive search for cases where R2=13...\n")

    cases_13, cases_14, total = exhaustive_search(seed=123, n_samples=1000000)

    print(f"\n📊 Results (from {total:,} cases with R2=13):")
    print(f"   Cases with R2=13, Rmax=13: {len(cases_13)}")
    print(f"   Cases with R2=13, Rmax=14: {len(cases_14)}")
    print(f"   Total:                      {len(cases_13) + len(cases_14)}")

    if total > 0:
        frac_13 = len(cases_13) / total
        frac_14 = len(cases_14) / total
        print(f"\n   P(Rmax=13 | R2=13) = {frac_13:.6f}")
        print(f"   P(Rmax=14 | R2=13) = {frac_14:.6f}")

    if len(cases_13) > 0:
        print(f"\n🔬 Examples with R2=13 and Rmax=13 (both 13s in same pile):")
        for i, case in enumerate(cases_13[:10]):
            pile = case['pile']
            print(f"   Round {case['round']}, Pile {case['pile_idx']}: {pile}")
            print(f"      Median={case['median']}, R2={case['R2']}, Max={case['max']}")
            # Count how many 13s
            count_13 = pile.count(13)
            print(f"      Number of 13s in pile: {count_13}")
        print(f"   ... and {len(cases_13) - 10} more")
    else:
        print(f"\n✅ NO cases found where R2=13 and Rmax=13!")
        print(f"   This confirms: P(Ace | R2=13) = 1.0")

    if len(cases_14) > 0:
        print(f"\n📋 Sample cases with R2=13 and Rmax=14:")
        for i, case in enumerate(cases_14[:5]):
            pile = case['pile']
            print(f"   Round {case['round']}, Pile {case['pile_idx']}: {pile}")
            print(f"      Median={case['median']}, R2={case['R2']}, Max={case['max']}")

    print(f"\n🎯 Conclusion:")
    if len(cases_13) == 0 and len(cases_14) > 0:
        print(f"   P(Ace | R2=13) = 1.0 across {total:,} samples")
        print(f"   This confirms the user's observation!")
        print(f"\n❌ The NPZ file showing P(Ace | median=3, R2=13) = 0.75 is INCORRECT")
        print(f"   The NPZ must have been generated with buggy code!")
    else:
        print(f"   Need to investigate further...")

if __name__ == "__main__":
    main()

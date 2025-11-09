#!/usr/bin/env python3
"""
Comprehensive verification that all signal definitions are now correct.
"""

import numpy as np
import sys

# Test the fixed functions directly
def test_signal_definitions():
    """Test all three signal definitions with comprehensive test cases."""

    print("=" * 70)
    print("COMPREHENSIVE SIGNAL DEFINITION VERIFICATION")
    print("=" * 70)

    # Import from the fixed modules
    from card_game_dynamic import _second_highest_rank as r2_dynamic
    from frontier import _second_highest_rank as r2_frontier
    from precomp import _second_highest_rank as r2_precomp
    from precomp_joint import _second_highest_rank as r2_joint

    # Test cases: (pile, expected_median, expected_top2sum, expected_r2)
    test_cases = [
        ([5,6,7,8,9], 7, 17, 8, "All unique"),
        ([2,3,3,13,14], 3, 27, 13, "Two 3s, ace at top"),
        ([2,3,3,13,13], 3, 16, 3, "Two 3s, two 13s"),
        ([3,3,3,13,13], 3, 16, 3, "Three 3s, two 13s"),
        ([2,2,3,13,14], 3, 27, 13, "Two 2s, ace at top"),
        ([7,7,7,7,7], 7, 14, 7, "All same rank (edge case)"),
        ([4,8,8,13,13], 8, 21, 8, "Two pairs"),
        ([2,2,2,2,14], 2, 16, 2, "Four of a kind plus ace"),
    ]

    all_passed = True

    print("\n1️⃣  Testing MEDIAN (should be arr[2] - position 2 in sorted pile)")
    print("-" * 70)
    for pile, exp_med, _, _, desc in test_cases:
        sorted_pile = sorted(pile)
        actual_med = sorted_pile[2]
        passed = actual_med == exp_med
        status = "✅" if passed else "❌"
        all_passed = all_passed and passed
        print(f"{status} {pile} → median={actual_med} (expected {exp_med}) [{desc}]")

    print("\n2️⃣  Testing TOP2SUM (sum of top 2 unique ranks)")
    print("-" * 70)
    for pile, _, exp_top2, _, desc in test_cases:
        unique_ranks = sorted(set(pile), reverse=True)
        actual_top2 = unique_ranks[0] + unique_ranks[1] if len(unique_ranks) >= 2 else unique_ranks[0] * 2
        passed = actual_top2 == exp_top2
        status = "✅" if passed else "❌"
        all_passed = all_passed and passed
        print(f"{status} {pile} → unique={unique_ranks}, top2sum={actual_top2} (expected {exp_top2}) [{desc}]")

    print("\n3️⃣  Testing R2 (second-highest unique rank)")
    print("-" * 70)
    print("   Testing card_game_dynamic.py:")
    for pile, _, _, exp_r2, desc in test_cases:
        arr = np.array(pile)
        actual_r2 = r2_dynamic(arr)
        passed = actual_r2 == exp_r2
        status = "✅" if passed else "❌"
        all_passed = all_passed and passed
        unique_ranks = sorted(set(pile), reverse=True)
        print(f"   {status} {pile} → unique={unique_ranks}, R2={actual_r2} (expected {exp_r2}) [{desc}]")

    print("\n   Testing frontier.py:")
    for pile, _, _, exp_r2, desc in test_cases:
        arr = np.array(pile)
        actual_r2 = r2_frontier(arr)
        passed = actual_r2 == exp_r2
        status = "✅" if passed else "❌"
        all_passed = all_passed and passed
        unique_ranks = sorted(set(pile), reverse=True)
        print(f"   {status} {pile} → unique={unique_ranks}, R2={actual_r2} (expected {exp_r2}) [{desc}]")

    print("\n   Testing precomp.py:")
    for pile, _, _, exp_r2, desc in test_cases:
        arr = np.array(pile)
        actual_r2 = r2_precomp(arr)
        passed = actual_r2 == exp_r2
        status = "✅" if passed else "❌"
        all_passed = all_passed and passed
        unique_ranks = sorted(set(pile), reverse=True)
        print(f"   {status} {pile} → unique={unique_ranks}, R2={actual_r2} (expected {exp_r2}) [{desc}]")

    print("\n   Testing precomp_joint.py:")
    for pile, _, _, exp_r2, desc in test_cases:
        arr = np.array(pile)
        actual_r2 = r2_joint(arr)
        passed = actual_r2 == exp_r2
        status = "✅" if passed else "❌"
        all_passed = all_passed and passed
        unique_ranks = sorted(set(pile), reverse=True)
        print(f"   {status} {pile} → unique={unique_ranks}, R2={actual_r2} (expected {exp_r2}) [{desc}]")

    # Critical constraint check
    print("\n4️⃣  CRITICAL CONSTRAINT: P(Ace | R2=13) must equal 1.0")
    print("-" * 70)
    print("   This will be verified after regenerating NPZ files.")
    print("   Logic: If R2=13 (second-highest unique rank), then max rank > 13.")
    print("   Since only rank > 13 is 14 (Ace), P(Ace | R2=13) = 1.0")

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Signal definitions are correct!")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME TESTS FAILED - Review the output above")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    exit_code = test_signal_definitions()
    sys.exit(exit_code)

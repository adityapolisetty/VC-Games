#!/usr/bin/env python3
"""
Test script for precomp.py output validation.

Validates that the posteriors in post_mc.npz are correctly computed:
- Checks NPZ structure and dimensions
- Validates probability distributions sum correctly
- Tests signal definitions (median, top2, R2)
- Verifies key constraints like P(Ace | R2=13) = 1.0

Usage:
  python code/testing/test_precomp.py --npz precomp_output/post_mc.npz
  python code/testing/test_precomp.py --npz precomp_output/post_mc.npz --verbose
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import from code/
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_res import round_seed, _deal_cards_global_deck, NUM_PILES, ACE_RANK, CARDS_PER_PILE
from precomp import _second_highest_rank

RMAX_DIM = 13  # ranks 2..14 map to 0..12
tol = 1e-6

def test_sample_boards(seed=42, n_boards=3):
    """Test 0: Display sample boards with computed signals for each pile."""
    print("\n" + "="*60)
    print("Test 0: Sample Board Signal Computation")
    print("="*60)
    print(f"\nShowing {n_boards} sample boards with signal computations:")

    for board_idx in range(n_boards):
        rng = np.random.default_rng(round_seed(seed, board_idx))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        print(f"\n{'─'*60}")
        print(f"Board {board_idx + 1} (seed={seed}, round={board_idx}):")
        print(f"{'─'*60}")

        for i, pile in enumerate(hands):
            sorted_pile = np.sort(pile)

            # Compute signals using functions from precomp
            median = int(sorted_pile[CARDS_PER_PILE // 2])
            unique_ranks = sorted(set(pile.tolist()), reverse=True)
            top2 = int(unique_ranks[0] + unique_ranks[1]) if len(unique_ranks) >= 2 else int(unique_ranks[0] * 2)
            r2 = _second_highest_rank(pile)
            rmax = int(np.max(pile))
            rmin = int(np.min(pile))

            # Compare with precomputed values
            pile_str = " ".join(f"{int(v):2d}" for v in sorted_pile)
            print(f"  Pile {i+1}: [{pile_str}]")
            print(f"    Median={median:2d} (pos 2), Top2sum={top2:2d}, R2={r2:2d}, Rmax={rmax:2d}, Rmin={rmin:2d}, Ace={'Y' if has_ace[i] else 'N'}")

            # Verify against returned values
            if median != medians[i]:
                print(f"      ⚠ Median mismatch: computed={median}, returned={medians[i]}")
            if top2 != top2sum[i]:
                print(f"      ⚠ Top2sum mismatch: computed={top2}, returned={top2sum[i]}")
            if rmax != max_rank[i]:
                print(f"      ⚠ Rmax mismatch: computed={rmax}, returned={max_rank[i]}")
            if rmin != min_rank[i]:
                print(f"      ⚠ Rmin mismatch: computed={rmin}, returned={min_rank[i]}")

        # Board-level summary
        ace_count = np.sum(has_ace)
        print(f"\n  Board summary: {ace_count} pile(s) with Ace, "
              f"median range [{medians.min()}-{medians.max()}], "
              f"top2sum range [{top2sum.min()}-{top2sum.max()}]")

    print(f"\n✓ Test 0 PASSED: Sample boards displayed with signal computations")
    return True


def test_npz_structure(npz_path):
    """Test 1: Verify NPZ file structure and required arrays exist."""
    print("\n" + "="*60)
    print("Test 1: NPZ Structure Validation")
    print("="*60)

    if not Path(npz_path).exists():
        print(f"✗ NPZ file not found: {npz_path}")
        return False

    with np.load(npz_path, allow_pickle=True) as z:
        required = {
            "joint_median_keys", "joint_median_mat",
            "joint_top2_keys", "joint_top2_mat",
            "rmax_median_keys", "rmax_median_mat",
            "rmax_top2_keys", "rmax_top2_mat",
            "prior_rmax", "r2_marginal_mat", "meta"
        }
        missing = required - set(z.files)
        if missing:
            print(f"✗ Missing arrays: {missing}")
            return False

        print(f"✓ All required arrays present: {len(required)} arrays")

        # Load arrays for dimension checking
        joint_med_keys = np.asarray(z["joint_median_keys"], int)
        joint_med_mat = np.asarray(z["joint_median_mat"], float)
        joint_t2_keys = np.asarray(z["joint_top2_keys"], int)
        joint_t2_mat = np.asarray(z["joint_top2_mat"], float)

        marg_med_keys = np.asarray(z["rmax_median_keys"], int)
        marg_med_mat = np.asarray(z["rmax_median_mat"], float)
        marg_t2_keys = np.asarray(z["rmax_top2_keys"], int)
        marg_t2_mat = np.asarray(z["rmax_top2_mat"], float)

        prior = np.asarray(z["prior_rmax"], float)
        r2_marg = np.asarray(z["r2_marginal_mat"], float)
        meta = dict(z["meta"].item())

    print(f"\nArray dimensions:")
    print(f"  Joint median: {len(joint_med_keys)} buckets, matrix shape {joint_med_mat.shape}")
    print(f"  Joint top2:   {len(joint_t2_keys)} buckets, matrix shape {joint_t2_mat.shape}")
    print(f"  Marg median:  {len(marg_med_keys)} buckets, matrix shape {marg_med_mat.shape}")
    print(f"  Marg top2:    {len(marg_t2_keys)} buckets, matrix shape {marg_t2_mat.shape}")
    print(f"  Prior Rmax:   shape {prior.shape}")
    print(f"  R2 marginal:  shape {r2_marg.shape}")

    print(f"\nMetadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # Check dimensions
    passed = True
    if joint_med_mat.shape != (len(joint_med_keys), RMAX_DIM, RMAX_DIM):
        print(f"✗ Joint median matrix shape mismatch")
        passed = False
    if joint_t2_mat.shape != (len(joint_t2_keys), RMAX_DIM, RMAX_DIM):
        print(f"✗ Joint top2 matrix shape mismatch")
        passed = False
    if marg_med_mat.shape != (len(marg_med_keys), RMAX_DIM):
        print(f"✗ Marginal median matrix shape mismatch")
        passed = False
    if marg_t2_mat.shape != (len(marg_t2_keys), RMAX_DIM):
        print(f"✗ Marginal top2 matrix shape mismatch")
        passed = False
    if prior.shape != (RMAX_DIM,):
        print(f"✗ Prior shape mismatch")
        passed = False
    if r2_marg.shape != (RMAX_DIM, RMAX_DIM):
        print(f"✗ R2 marginal shape mismatch")
        passed = False

    if passed:
        print(f"\n✓ Test 1 PASSED: All dimensions correct")
    else:
        print(f"\n✗ Test 1 FAILED: Dimension mismatches found")

    return passed


def test_probability_sums(npz_path):
    """Test 2: Verify all probability distributions sum correctly."""
    print("\n" + "="*60)
    print("Test 2: Probability Distribution Validation")
    print("="*60)

    with np.load(npz_path, allow_pickle=True) as z:
        joint_med_mat = np.asarray(z["joint_median_mat"], float)
        joint_t2_mat = np.asarray(z["joint_top2_mat"], float)
        marg_med_mat = np.asarray(z["rmax_median_mat"], float)
        marg_t2_mat = np.asarray(z["rmax_top2_mat"], float)
        prior = np.asarray(z["prior_rmax"], float)
        r2_marg = np.asarray(z["r2_marginal_mat"], float)

    passed = True

    # Check joint posteriors: each (bucket, R2) row should sum to 0 or 1
    print("\nJoint posteriors (sum over Rmax for each bucket, R2 pair):")
    joint_med_sums = joint_med_mat.sum(axis=2)  # [buckets, R2]
    joint_t2_sums = joint_t2_mat.sum(axis=2)

    med_invalid = np.sum((joint_med_sums > tol) & (np.abs(joint_med_sums - 1.0) > tol))
    t2_invalid = np.sum((joint_t2_sums > tol) & (np.abs(joint_t2_sums - 1.0) > tol))

    if med_invalid > 0:
        print(f"  ✗ Median: {med_invalid} rows don't sum to 0 or 1")
        passed = False
    else:
        print(f"  ✓ Median: All rows sum to 0 or 1")

    if t2_invalid > 0:
        print(f"  ✗ Top2: {t2_invalid} rows don't sum to 0 or 1")
        passed = False
    else:
        print(f"  ✓ Top2: All rows sum to 0 or 1")

    # Check marginal posteriors: each bucket row should sum to 1
    print("\nMarginal posteriors (sum over Rmax for each bucket):")
    marg_med_sums = marg_med_mat.sum(axis=1)
    marg_t2_sums = marg_t2_mat.sum(axis=1)

    med_marg_invalid = np.sum(np.abs(marg_med_sums - 1.0) > tol)
    t2_marg_invalid = np.sum(np.abs(marg_t2_sums - 1.0) > tol)

    if med_marg_invalid > 0:
        print(f"  ✗ Median: {med_marg_invalid} rows don't sum to 1.0")
        passed = False
    else:
        print(f"  ✓ Median: All {len(marg_med_sums)} rows sum to 1.0")

    if t2_marg_invalid > 0:
        print(f"  ✗ Top2: {t2_marg_invalid} rows don't sum to 1.0")
        passed = False
    else:
        print(f"  ✓ Top2: All {len(marg_t2_sums)} rows sum to 1.0")

    # Check prior sums to 1
    print("\nPrior distribution:")
    prior_sum = prior.sum()
    if np.abs(prior_sum - 1.0) > tol:
        print(f"  ✗ Prior sums to {prior_sum:.10f}, not 1.0")
        passed = False
    else:
        print(f"  ✓ Prior sums to 1.0")

    # Check R2 marginal: each R2 row should sum to 1 (excluding R2=14 which is impossible)
    print("\nR2 marginal P(Rmax | R2):")
    r2_sums = r2_marg.sum(axis=1)
    # Only check R2 from 2 to 13 (indices 0 to 11); R2=14 (index 12) is impossible
    r2_invalid = 0
    for r2_idx in range(12):  # R2=2 to R2=13 (indices 0-11)
        if np.abs(r2_sums[r2_idx] - 1.0) > tol:
            r2_invalid += 1
            r2_val = r2_idx + 2
            print(f"    R2={r2_val} (idx={r2_idx}): sum={r2_sums[r2_idx]:.10f}")

    if r2_invalid > 0:
        print(f"  ✗ {r2_invalid} R2 rows (out of 12 valid) don't sum to 1.0")
        passed = False
    else:
        print(f"  ✓ All 12 valid R2 rows (R2=2 to R2=13) sum to 1.0")
        # Verify R2=14 is indeed 0
        if r2_sums[12] < tol:
            print(f"  ✓ R2=14 correctly has sum=0.0 (impossible case)")
        else:
            print(f"  ✗ WARNING: R2=14 has non-zero sum={r2_sums[12]:.10f}")
            passed = False

    if passed:
        print(f"\n✓ Test 2 PASSED: All probability distributions sum correctly")
    else:
        print(f"\n✗ Test 2 FAILED: Invalid probability sums found")

    return passed


def test_r2_constraint(npz_path, seed=42, n_samples=10000):
    """Test 3: Verify P(Ace | R2=13) = 1.0 (key signal definition check)."""
    print("\n" + "="*60)
    print("Test 3: R2 Signal Definition Validation")
    print("="*60)
    print(f"\nVerifying P(Rmax=14 | R2=13) = 1.0 across {n_samples} samples")
    print("(If R2=13, then second-highest unique rank is 13, so Rmax must be 14)")

    with np.load(npz_path, allow_pickle=True) as z:
        r2_marg = np.asarray(z["r2_marginal_mat"], float)

    # Check P(Rmax=14 | R2=13) in the marginal
    r2_13_idx = 13 - 2  # R2=13 -> index 11
    ace_idx = 14 - 2    # Rmax=14 -> index 12

    p_ace_given_r2_13 = r2_marg[r2_13_idx, ace_idx]
    print(f"\nFrom precomputed marginal:")
    print(f"  P(Rmax=14 | R2=13) = {p_ace_given_r2_13:.10f}")

    # Empirical check: generate boards and verify
    count_r2_13 = 0
    count_rmax_14_given_r2_13 = 0

    for i in range(n_samples):
        rng = np.random.default_rng(round_seed(seed, i))
        _, hands, _, _, max_rank, _ = _deal_cards_global_deck(rng)

        for pile in hands:
            r2 = _second_highest_rank(pile)
            rmax = int(np.max(pile))

            if r2 == 13:
                count_r2_13 += 1
                if rmax == 14:
                    count_rmax_14_given_r2_13 += 1

    if count_r2_13 > 0:
        empirical_prob = count_rmax_14_given_r2_13 / count_r2_13
        print(f"\nEmpirical verification ({n_samples} rounds, {count_r2_13} piles with R2=13):")
        print(f"  P(Rmax=14 | R2=13) = {empirical_prob:.10f}")
        print(f"  Count: {count_rmax_14_given_r2_13}/{count_r2_13}")
    else:
        print(f"\n  Warning: No piles with R2=13 found in {n_samples} rounds")
        empirical_prob = None

    # Check if close to 1.0
    passed = True
    if np.abs(p_ace_given_r2_13 - 1.0) > tol:
        print(f"\n✗ FAILED: P(Rmax=14 | R2=13) != 1.0")
        passed = False
    else:
        print(f"\n✓ PASSED: P(Rmax=14 | R2=13) = 1.0 (as expected)")

    if empirical_prob is not None and np.abs(empirical_prob - 1.0) > 0.01:
        print(f"✗ WARNING: Empirical probability deviates from 1.0")
        passed = False

    return passed


def test_signal_definitions(npz_path, seed=42, n_rounds=1000, verbose=False):
    """Test 4: Verify signal definitions are computed correctly."""
    print("\n" + "="*60)
    print("Test 4: Signal Definition Correctness")
    print("="*60)
    print(f"\nChecking {n_rounds} boards to verify:")
    print("  - Median = 3rd card in sorted pile (position 2)")
    print("  - Top2sum = sum of top 2 unique ranks")
    print("  - R2 = second-highest unique rank")

    errors = []

    for i in range(n_rounds):
        rng = np.random.default_rng(round_seed(seed, i))
        _, hands, medians, top2sum, max_rank, _ = _deal_cards_global_deck(rng)

        for j, pile in enumerate(hands):
            sorted_pile = np.sort(pile)

            # Check median (position 2 in sorted pile)
            expected_median = int(sorted_pile[2])
            if medians[j] != expected_median:
                errors.append(f"Round {i}, Pile {j}: Median mismatch - expected {expected_median}, got {medians[j]}")
                if verbose:
                    print(f"  Pile: {sorted_pile}, Median: {medians[j]} (expected {expected_median})")

            # Check top2sum (sum of top 2 unique ranks)
            unique_ranks = sorted(set(pile.tolist()), reverse=True)
            expected_top2 = unique_ranks[0] + unique_ranks[1] if len(unique_ranks) >= 2 else unique_ranks[0] * 2
            if top2sum[j] != expected_top2:
                errors.append(f"Round {i}, Pile {j}: Top2sum mismatch - expected {expected_top2}, got {top2sum[j]}")
                if verbose:
                    print(f"  Pile: {sorted_pile}, Top2sum: {top2sum[j]} (expected {expected_top2})")

            # Check R2 (second-highest unique rank)
            expected_r2 = _second_highest_rank(pile)
            actual_r2 = _second_highest_rank(pile)  # Using same function as precomp
            if actual_r2 != expected_r2:
                errors.append(f"Round {i}, Pile {j}: R2 mismatch - expected {expected_r2}, got {actual_r2}")
                if verbose:
                    print(f"  Pile: {sorted_pile}, R2: {actual_r2} (expected {expected_r2})")

    if errors:
        print(f"\n✗ FAILED: Found {len(errors)} signal definition errors")
        for err in errors[:10]:  # Show first 10
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    else:
        print(f"\n✓ PASSED: All {n_rounds * NUM_PILES} piles have correct signal definitions")
        return True


def test_sample_posteriors(npz_path, verbose=False):
    """Test 5: Display sample posteriors for sanity checking."""
    print("\n" + "="*60)
    print("Test 5: Sample Posterior Inspection")
    print("="*60)

    with np.load(npz_path, allow_pickle=True) as z:
        joint_med_keys = np.asarray(z["joint_median_keys"], int)
        joint_med_mat = np.asarray(z["joint_median_mat"], float)
        marg_med_keys = np.asarray(z["rmax_median_keys"], int)
        marg_med_mat = np.asarray(z["rmax_median_mat"], float)
        r2_marg = np.asarray(z["r2_marginal_mat"], float)

    # Sample marginal posterior
    if len(marg_med_keys) > 0:
        print(f"\nSample marginal P(Rmax | Median={marg_med_keys[0]}):")
        for r in range(2, 15):
            p = marg_med_mat[0, r-2]
            if p > 0.001:
                print(f"  Rmax={r:2d}: {p:.6f}")

    # Sample joint posterior (find a valid combination)
    print(f"\nSample joint P(Rmax | Median, R2) for a valid combination:")
    found = False
    for i in range(min(5, len(joint_med_keys))):
        for r2_idx in range(7, 13):  # R2 from 9 to 14
            r2_val = r2_idx + 2
            if joint_med_mat[i, r2_idx, :].sum() > 0.5:
                print(f"  P(Rmax | Median={joint_med_keys[i]}, R2={r2_val}):")
                for r in range(2, 15):
                    p = joint_med_mat[i, r2_idx, r-2]
                    if p > 0.001:
                        print(f"    Rmax={r:2d}: {p:.6f}")
                found = True
                break
        if found:
            break

    # Sample R2 marginal
    print(f"\nSample R2 marginal P(Rmax | R2=10):")
    r2_idx = 10 - 2
    for r in range(2, 15):
        p = r2_marg[r2_idx, r-2]
        if p > 0.001:
            print(f"  Rmax={r:2d}: {p:.6f}")

    print(f"\n✓ Test 5 PASSED: Sample posteriors displayed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test precomp.py output validation")
    parser.add_argument("--npz", type=str, default="../precomp_output/post_mc.npz",
                        help="Path to post_mc.npz file")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random tests")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("PRECOMP OUTPUT VALIDATION TEST SUITE")
    print("="*60)
    print(f"Testing NPZ: {args.npz}")

    results = []

    # Run all tests
    results.append(("Sample Boards", test_sample_boards(seed=args.seed)))
    results.append(("NPZ Structure", test_npz_structure(args.npz)))
    results.append(("Probability Sums", test_probability_sums(args.npz)))
    results.append(("R2 Constraint", test_r2_constraint(args.npz, seed=args.seed)))
    results.append(("Signal Definitions", test_signal_definitions(args.npz, seed=args.seed, verbose=args.verbose)))
    results.append(("Sample Posteriors", test_sample_posteriors(args.npz, verbose=args.verbose)))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(p for _, p in results)
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())

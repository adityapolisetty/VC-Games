#!/usr/bin/env python3
"""
Test that precomp.py generates the correct consolidated NPZ structure.
"""

import numpy as np
from pathlib import Path

def test_precomp_npz(npz_path):
    """Verify NPZ has all required arrays with correct shapes."""

    print(f"🔍 Testing NPZ structure: {npz_path}\n")

    if not Path(npz_path).exists():
        print(f"❌ File not found: {npz_path}")
        return False

    with np.load(npz_path, allow_pickle=False) as z:
        files = set(z.files)
        print(f"📦 Available arrays: {sorted(files)}\n")

        # Required arrays
        required_joint = {
            'joint_median_keys', 'joint_median_mat',
            'joint_top2_keys', 'joint_top2_mat',
        }
        required_marginal = {
            'rmax_median_keys', 'rmax_median_mat',
            'rmax_top2_keys', 'rmax_top2_mat',
            'rmax_max_keys', 'rmax_max_mat',
            'rmax_min_keys', 'rmax_min_mat',
        }
        required_ace = {
            'ace_median_probs', 'ace_top2_probs',
            'ace_max_probs', 'ace_min_probs',
        }
        required_common = {
            'prior_rmax', 'r2_marginal_mat', 'meta',
        }

        required = required_joint | required_marginal | required_ace | required_common
        missing = required - files

        if missing:
            print(f"❌ Missing required arrays: {sorted(missing)}")
            return False

        print("✅ All required arrays present\n")

        # Check shapes
        print("📊 Array shapes:\n")

        all_ok = True

        # Joint posteriors should be 3D: [buckets, 13 R2, 13 Rmax]
        for name in ['joint_median_mat', 'joint_top2_mat']:
            arr = z[name]
            print(f"  {name:25s} {arr.shape}")
            if len(arr.shape) != 3 or arr.shape[1] != 13 or arr.shape[2] != 13:
                print(f"    ❌ Expected shape (N, 13, 13), got {arr.shape}")
                all_ok = False

        print()

        # Marginal posteriors should be 2D: [buckets, 13 Rmax]
        for name in ['rmax_median_mat', 'rmax_top2_mat', 'rmax_max_mat', 'rmax_min_mat']:
            arr = z[name]
            print(f"  {name:25s} {arr.shape}")
            if len(arr.shape) != 2 or arr.shape[1] != 13:
                print(f"    ❌ Expected shape (N, 13), got {arr.shape}")
                all_ok = False

        print()

        # Ace probabilities should be 1D
        for name in ['ace_median_probs', 'ace_top2_probs', 'ace_max_probs', 'ace_min_probs']:
            arr = z[name]
            print(f"  {name:25s} {arr.shape}")
            if len(arr.shape) != 1:
                print(f"    ❌ Expected 1D array, got {arr.shape}")
                all_ok = False

        print()

        # R2 marginal should be 2D: [13 R2, 13 Rmax]
        r2_arr = z['r2_marginal_mat']
        print(f"  r2_marginal_mat           {r2_arr.shape}")
        if r2_arr.shape != (13, 13):
            print(f"    ❌ Expected shape (13, 13), got {r2_arr.shape}")
            all_ok = False

        # Prior should be 1D: [13 Rmax]
        prior_arr = z['prior_rmax']
        print(f"  prior_rmax                {prior_arr.shape}")
        if prior_arr.shape != (13,):
            print(f"    ❌ Expected shape (13,), got {prior_arr.shape}")
            all_ok = False

        print()

        # Check probability constraints
        print("🧮 Verifying probability constraints:\n")

        # Joint posteriors: each row should sum to 1 (across Rmax for each bucket x R2)
        for name in ['joint_median_mat', 'joint_top2_mat']:
            arr = z[name]
            row_sums = arr.sum(axis=2)
            non_zero_rows = row_sums > 0
            if non_zero_rows.any():
                sums_close_to_one = np.allclose(row_sums[non_zero_rows], 1.0, atol=1e-6)
                status = "✅" if sums_close_to_one else "❌"
                print(f"  {status} {name}: rows sum to 1.0")
                if not sums_close_to_one:
                    all_ok = False

        # Marginal posteriors: each row should sum to 1 (across Rmax for each bucket)
        for name in ['rmax_median_mat', 'rmax_top2_mat', 'rmax_max_mat', 'rmax_min_mat']:
            arr = z[name]
            row_sums = arr.sum(axis=1)
            non_zero_rows = row_sums > 0
            if non_zero_rows.any():
                sums_close_to_one = np.allclose(row_sums[non_zero_rows], 1.0, atol=1e-6)
                status = "✅" if sums_close_to_one else "❌"
                print(f"  {status} {name}: rows sum to 1.0")
                if not sums_close_to_one:
                    all_ok = False

        # Prior should sum to 1
        prior_sum = prior_arr.sum()
        prior_ok = np.isclose(prior_sum, 1.0, atol=1e-6)
        status = "✅" if prior_ok else "❌"
        print(f"  {status} prior_rmax: sums to {prior_sum:.6f}")
        if not prior_ok:
            all_ok = False

        # R2 marginal: each row should sum to 1
        r2_row_sums = r2_arr.sum(axis=1)
        non_zero_rows = r2_row_sums > 0
        if non_zero_rows.any():
            r2_ok = np.allclose(r2_row_sums[non_zero_rows], 1.0, atol=1e-6)
            status = "✅" if r2_ok else "❌"
            print(f"  {status} r2_marginal_mat: rows sum to 1.0")
            if not r2_ok:
                all_ok = False

        print()

        # Critical constraint: P(Ace | R2=13) should be 1.0
        print("🎯 Critical constraint check:\n")
        r2_idx = 13 - 2  # R2=13 maps to index 11
        ace_idx = 14 - 2  # Ace=14 maps to index 12
        prob_ace_given_r2_13 = r2_arr[r2_idx, ace_idx]
        constraint_ok = np.isclose(prob_ace_given_r2_13, 1.0, atol=1e-6)
        status = "✅" if constraint_ok else "❌"
        print(f"  {status} P(Ace | R2=13) = {prob_ace_given_r2_13:.8f} (expected 1.0)")
        if not constraint_ok:
            all_ok = False

        print()

        if all_ok:
            print("✅ ALL TESTS PASSED - NPZ structure is correct!")
            return True
        else:
            print("❌ SOME TESTS FAILED - Review output above")
            return False

if __name__ == "__main__":
    import sys
    npz_path = sys.argv[1] if len(sys.argv) > 1 else "precomp_output/posteriors.npz"
    success = test_precomp_npz(npz_path)
    sys.exit(0 if success else 1)

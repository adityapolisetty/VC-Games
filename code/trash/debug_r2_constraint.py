#!/usr/bin/env python3
"""
Debug script to verify the logical constraint:
P(Ace | median=3, R2=13) must equal 1.0

If median=3 and R2=13 (second-highest rank), then the pile structure is:
[c1, c2, 3, 13, c5] where c1 <= c2 <= 3, and c5 > 13
Since only rank > 13 is 14 (Ace), we must have c5 = 14.
"""

import numpy as np
from pathlib import Path

def check_joint_posterior():
    """Load joint posterior and check the constraint."""

    # Load joint posterior NPZ
    post_path = Path("../output/post_joint.npz")
    if not post_path.exists():
        print(f"❌ File not found: {post_path}")
        print("   Run: python precomp_joint.py --seed 123 --rounds 500000 --out output/post_joint.npz --procs 8")
        return

    print(f"📂 Loading {post_path}")
    with np.load(post_path, allow_pickle=False) as z:
        print(f"   Available arrays: {z.files}")

        med_keys = z["joint_median_keys"]
        med_mat = z["joint_median_mat"]  # [K, 13(R2), 13(Rmax)]

        print(f"\n📊 Data shape:")
        print(f"   median keys: {med_keys.shape} = {list(med_keys)}")
        print(f"   median mat:  {med_mat.shape} (bucket, R2, Rmax)")

    # Find bucket index for median=3
    bucket_idx = np.where(med_keys == 3)[0]
    if len(bucket_idx) == 0:
        print(f"\n❌ No data for median=3")
        return
    bucket_idx = bucket_idx[0]
    print(f"\n🔍 Checking median=3 (bucket index {bucket_idx})")

    # Check P(Rmax | median=3, R2=13)
    # R2=13 maps to index 13-2=11
    # Ace=14 maps to index 14-2=12
    r2_idx = 13 - 2  # 11
    ace_idx = 14 - 2  # 12

    prob_vec = med_mat[bucket_idx, r2_idx, :]  # P(Rmax=k | median=3, R2=13) for all k
    prob_ace = med_mat[bucket_idx, r2_idx, ace_idx]  # P(Rmax=14 | median=3, R2=13)

    print(f"\n📈 P(Rmax | median=3, R2=13) distribution:")
    print(f"   Rmax  Probability")
    print(f"   ----  -----------")
    for rmax in range(2, 15):
        rmax_idx = rmax - 2
        prob = prob_vec[rmax_idx]
        marker = " ⭐" if rmax == 14 else ""
        marker += " ⚠️ BUG!" if prob > 0 and rmax < 14 else ""
        print(f"   {rmax:4d}  {prob:11.6f}{marker}")

    print(f"\n🎯 Logical constraint check:")
    print(f"   P(Ace | median=3, R2=13) = {prob_ace:.8f}")

    if np.isclose(prob_ace, 1.0, atol=1e-6):
        print(f"   ✅ PASS: Probability is 1.0 (within tolerance)")
    else:
        print(f"   ❌ FAIL: Expected 1.0, got {prob_ace:.8f}")
        print(f"\n   This is a logical bug because:")
        print(f"   - Pile structure: [c1, c2, 3, 13, c5] (sorted)")
        print(f"   - Since R2=13 is second-highest, c5 > 13")
        print(f"   - Only rank > 13 is 14 (Ace)")
        print(f"   - Therefore c5 must be 14 with probability 1.0")

    # Check if any impossible cases have non-zero probability
    print(f"\n🔬 Checking for impossible cases (Rmax < R2):")
    impossible_found = False
    for bucket_i in range(len(med_keys)):
        for r2 in range(2, 15):
            r2_idx = r2 - 2
            for rmax in range(2, r2):  # Rmax < R2 is impossible
                rmax_idx = rmax - 2
                prob = med_mat[bucket_i, r2_idx, rmax_idx]
                if prob > 1e-10:
                    print(f"   ⚠️ median={med_keys[bucket_i]}, R2={r2}, Rmax={rmax}: P={prob:.6f} (IMPOSSIBLE!)")
                    impossible_found = True

    if not impossible_found:
        print(f"   ✅ No impossible cases found (Rmax >= R2 always)")

    # Additional check: for any median and R2=13, check if Ace probability is reasonable
    print(f"\n📊 P(Ace | median=k, R2=13) for all medians:")
    print(f"   Median  P(Ace | median, R2=13)  Expected")
    print(f"   ------  ----------------------  --------")
    r2_idx = 13 - 2
    for i, med in enumerate(med_keys):
        prob_ace_given_med = med_mat[i, r2_idx, ace_idx]

        # Logical expectation:
        # If median <= 13 and R2 = 13, then pile is [c1, c2, median, ..., 13, c5]
        # Since 13 is second-highest, c5 > 13, so c5 must be 14
        expected = "1.0" if med <= 13 else "?"
        marker = ""
        if med <= 13 and not np.isclose(prob_ace_given_med, 1.0, atol=1e-6):
            marker = " ⚠️ BUG!"

        print(f"   {med:6d}  {prob_ace_given_med:22.6f}  {expected:8s}{marker}")


if __name__ == "__main__":
    check_joint_posterior()

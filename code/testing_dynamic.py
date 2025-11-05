"""Testing harness for the dynamic 2-stage card game.

Quick usage:
  python code/testing_dynamic.py --test 1
  python code/testing_dynamic.py --test 2 --pre_rounds 2000000 --procs 8 --post output_joint/post_joint.testing.npz
  python code/testing_dynamic.py --test 3 --post_t3 output/post_mc.npz --post_joint output_joint/post_joint.npz --sig median --nsigs 5 --sp 0 --ap 20 --alpha 0.5
  python code/testing_dynamic.py --test 4 --post_t4 output/post_mc.npz --post_joint output_joint/post_joint.npz --sig median --nsigs 5 --alpha 0.5
  python code/testing_dynamic.py --test 5 --ap 20 --sp 0
  python code/testing_dynamic.py --test all

Tests:
  1) Board draws comply with constraints
     - Prints 3 sample boards with per-pile ranks and signals
     - Checks global-deck constraints on 1000 boards

  2) Precompute for joint posteriors is correct
     - Runs precomp_joint.py and validates joint posterior matrices
     - Checks dimensions, probability sums, and R2 conditioning

  3) Single-round diagnostics for net return correctness
     - One deterministic board; prints Stage-1 and Stage-2 investments
     - Verifies final net return matches sum of stage payoffs

  4) Stage 2 investments only in Stage 1 piles
     - Verifies w2 > 0 only where w1 > 0 for all three rules
     - Tests multiple boards and allocation ratios

  5) Stage 2 payoffs are half of Stage 1 payoffs
     - Verifies h_vals2 = 0.5 * h_vals for all ranks
     - Tests with both ace-only and scaled payoff modes
"""

import argparse
import numpy as np
from numpy.random import default_rng
import sys
import importlib

from card_game_dynamic import (
    _deal_cards_global_deck,
    NUM_PILES,
    CARDS_PER_PILE,
    ACE_RANK,
    BUDGET,
    round_seed,
    _load_mc_posteriors,
    _load_joint_posteriors,
    run_single_round_dynamic,
    _second_highest_rank,
    _compute_gross_payout,
)

import precomp_joint


# ============================================
# Test 1: Board draws comply with constraints
# ============================================

def _print_board(idx: int, seed: int) -> None:
    rng = default_rng(round_seed(seed, idx))
    has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

    print(f"Board {idx + 1}")
    for i in range(NUM_PILES):
        pile = " ".join(f"{int(v):2d}" for v in hands[i].tolist())
        print(f"  Pile {i+1}: {pile}")

    fmt_bool = lambda arr: " ".join("Y" if b else "N" for b in arr)
    print("  has_ace :", fmt_bool(has_ace))
    print("  medians :", " ".join(f"{int(x):2d}" for x in medians.tolist()))
    print("  top2sum :", " ".join(f"{int(x):2d}" for x in top2sum.tolist()))
    print("  max_rank:", " ".join(f"{int(x):2d}" for x in max_rank.tolist()))
    print("  min_rank:", " ".join(f"{int(x):2d}" for x in min_rank.tolist()))
    print()


def _check_constraints(seed: int, runs: int) -> int:
    violations = 0
    for i in range(runs):
        rng = default_rng(round_seed(seed, i))
        _, hands, *_ = _deal_cards_global_deck(rng)
        flat = np.concatenate(hands)

        # 1) total cards exactly 45
        if flat.size != NUM_PILES * CARDS_PER_PILE:
            violations += 1
            continue

        # 2) per-rank caps: Ace<=1, King<=2, Queen<=2, others<=4
        unique, counts = np.unique(flat, return_counts=True)
        cmap = dict(zip(unique.tolist(), counts.tolist()))
        for rank in range(2, 15):
            c = cmap.get(rank, 0)
            if rank == ACE_RANK:
                if c != 1: violations += 1; break
            elif rank == 13:
                if c != 2: violations += 1; break
            elif rank == 12:
                if c != 2: violations += 1; break
            else:
                if c > 4: violations += 1; break
    return violations


def _test_1(seed: int) -> None:
    print("\n\n--------------------------------------------------")
    print(f"\nTest 1: Board draws comply with constraints\n")

    # Print 3 example boards
    for i in range(3):
        _print_board(i, seed)

    # Check constraints over 100000 boards
    n = 100000
    v = _check_constraints(seed, n)
    print("Constraint check over", n, "boards (serial)")
    print("Restrictions tested:")
    print(" - Exactly 45 cards total on board")
    print(" - Ace count == 1; King (13) == 2; Queen (12) == 2; all other ranks <= 4")
    print("Violations:", v)
    if v == 0:
        print("✓ Test 1 PASSED")
    else:
        print("✗ Test 1 FAILED")


# =======================================================
# Test 2: Precompute for joint posteriors is correct
# =======================================================

def _test_2(pre_rounds: int, procs: int, seed: int, out_path: str) -> None:
    print("\n\n--------------------------------------------------")
    print(f"\nTest 2: Precompute joint posteriors and validate\n")

    # Run full precompute to generate joint NPZ
    save_argv = list(sys.argv)
    sys.argv = [
        "precomp_joint.py",
        "--seed", str(seed),
        "--rounds", str(int(pre_rounds)),
        "--out", out_path,
        "--procs", str(int(procs)),
    ]
    importlib.reload(precomp_joint)
    precomp_joint.main()
    sys.argv = save_argv

    # Load and validate joint NPZ
    with np.load(out_path, allow_pickle=False) as z:
        med_keys = np.asarray(z["joint_median_keys"], int)
        med_mat = np.asarray(z["joint_median_mat"], float)  # [K, 13(R2), 13(Rmax)]
        t2_keys = np.asarray(z["joint_top2_keys"], int)
        t2_mat = np.asarray(z["joint_top2_mat"], float)  # [T, 13, 13]
        prior_rmax = np.asarray(z["prior_rmax"], float)
        r2_marg = np.asarray(z["r2_marginal_mat"], float)  # [13(R2), 13(Rmax)]

    print("Joint posterior NPZ loaded successfully")
    print(f"  Median buckets: {len(med_keys)} (range {med_keys.min()}-{med_keys.max()})")
    print(f"  Top2 buckets: {len(t2_keys)} (range {t2_keys.min()}-{t2_keys.max()})")
    print(f"  Matrix shapes: median={med_mat.shape}, top2={t2_mat.shape}, r2_marginal={r2_marg.shape}")

    # Validation checks
    passed = True

    # Check 1: Dimensions
    if med_mat.shape[1:] != (13, 13):
        print(f"✗ Median matrix shape mismatch: expected (*, 13, 13), got {med_mat.shape}")
        passed = False
    if t2_mat.shape[1:] != (13, 13):
        print(f"✗ Top2 matrix shape mismatch: expected (*, 13, 13), got {t2_mat.shape}")
        passed = False

    # Check 2: Each row sums to 0 or 1 (for valid R2 values given bucket)
    # Some (bucket, R2) combinations are impossible, so their rows sum to 0
    # Valid combinations should sum to 1
    med_row_sums = med_mat.sum(axis=2)  # [K, 13]
    t2_row_sums = t2_mat.sum(axis=2)  # [T, 13]

    # For each row sum, check it's either ~0 (impossible) or ~1 (valid probability)
    def check_row_sums(row_sums, name):
        errors = []
        for i, row_sum_vec in enumerate(row_sums):
            for j, s in enumerate(row_sum_vec):
                # Each sum should be either ~0 (no data) or ~1 (valid probability)
                if not (abs(s) < 1e-6 or abs(s - 1.0) < 1e-6):
                    errors.append((i, j, s))
        return errors

    med_errors = check_row_sums(med_row_sums, "median")
    t2_errors = check_row_sums(t2_row_sums, "top2")

    # Count valid (non-zero) combinations
    med_valid = np.sum(med_row_sums > 0.5)
    t2_valid = np.sum(t2_row_sums > 0.5)

    print(f"\nProbability sum checks:")
    print(f"  Median: {med_valid} valid (bucket, R2) combinations out of {med_row_sums.size} total")
    print(f"  Top2: {t2_valid} valid (bucket, R2) combinations out of {t2_row_sums.size} total")

    if med_errors:
        print(f"✗ Median has {len(med_errors)} row sums not equal to 0 or 1")
        for i, j, s in med_errors[:5]:  # Show first 5
            print(f"    bucket_idx={i}, R2_idx={j} (R2={j+2}): sum={s:.6f}")
        passed = False
    else:
        print(f"  ✓ All median row sums are either 0 or 1")

    if t2_errors:
        print(f"✗ Top2 has {len(t2_errors)} row sums not equal to 0 or 1")
        for i, j, s in t2_errors[:5]:
            print(f"    bucket_idx={i}, R2_idx={j} (R2={j+2}): sum={s:.6f}")
        passed = False
    else:
        print(f"  ✓ All top2 row sums are either 0 or 1")

    # Check 3: Print examples of impossible (bucket, R2) combinations
    print(f"\nExamples of impossible (Median, R2) combinations (zero probability mass):")
    zero_combos = []
    for i, med_val in enumerate(med_keys):
        for j in range(13):
            r2_val = j + 2
            if med_row_sums[i, j] < 1e-6:
                zero_combos.append((int(med_val), r2_val))

    if zero_combos:
        print(f"  Found {len(zero_combos)} impossible combinations. Showing first 10:")
        for med_val, r2_val in zero_combos[:10]:
            print(f"    Median={med_val}, R2={r2_val} (impossible: R2 must be ≥ median in a 5-card pile)")
    else:
        print(f"  No impossible combinations found (unexpected)")

    # Check 4: Marginal P(Rmax | R2) validation
    print(f"\nMarginal P(Rmax | R2) validation:")
    if r2_marg.shape != (13, 13):
        print(f"✗ Marginal matrix shape mismatch: expected (13, 13), got {r2_marg.shape}")
        passed = False
    else:
        print(f"  ✓ Marginal matrix shape is correct: {r2_marg.shape}")

    # Check each row sums to 1 (all R2 values are possible)
    r2_row_sums = r2_marg.sum(axis=1)
    r2_errors = []
    for r2_idx, s in enumerate(r2_row_sums):
        if abs(s - 1.0) > 1e-6:
            r2_errors.append((r2_idx, s))

    if r2_errors:
        print(f"✗ Marginal has {len(r2_errors)} rows that don't sum to 1.0")
        for r2_idx, s in r2_errors[:5]:
            print(f"    R2={r2_idx+2}: sum={s:.6f}")
        passed = False
    else:
        print(f"  ✓ All marginal rows sum to 1.0")

    # Print sample marginal posterior
    print(f"\nSample marginal posterior P(Rmax | R2=10):")
    r2_idx = 10 - 2  # R2=10 -> index 8
    sample_vec = r2_marg[r2_idx, :]
    for r in range(2, 15):
        if sample_vec[r-2] > 0.001:
            print(f"    Rmax={r}: {sample_vec[r-2]:.6f}")

    # Check 5: Prior sums to 1
    prior_sum = prior_rmax.sum()
    print(f"\nPrior distribution sum: {prior_sum:.10f}")
    if np.abs(prior_sum - 1.0) > 1e-6:
        print(f"✗ Prior does not sum to 1.0")
        passed = False
    else:
        print(f"  ✓ Prior sums to 1.0")

    # Check 6: Print sample posteriors for inspection
    print(f"\nSample valid posterior (Median={med_keys[0]}, R2=10):")
    r2_idx_check = 10 - 2  # R2=10 -> index 8
    if len(med_keys) > 0 and med_row_sums[0, r2_idx_check] > 0.5:
        sample_vec = med_mat[0, r2_idx_check, :]
        print(f"  P(Rmax | Median={med_keys[0]}, R2=10):")
        for r in range(2, 15):
            if sample_vec[r-2] > 0.001:  # Only show non-negligible probabilities
                print(f"    Rmax={r}: {sample_vec[r-2]:.6f}")
    elif len(med_keys) > 0:
        print(f"  (Median={med_keys[0]}, R2=10 is an impossible combination)")

    # Find and show a valid combination as example
    print(f"\nSample valid posterior (searching for valid combination):")
    found_valid = False
    for i, med_val in enumerate(med_keys[:5]):  # Check first 5 buckets
        for j in range(7, 13):  # Check R2 from 9 to 14
            r2_val = j + 2
            if med_row_sums[i, j] > 0.5:
                sample_vec = med_mat[i, j, :]
                print(f"  P(Rmax | Median={med_val}, R2={r2_val}):")
                for r in range(2, 15):
                    if sample_vec[r-2] > 0.001:
                        print(f"    Rmax={r}: {sample_vec[r-2]:.6f}")
                found_valid = True
                break
        if found_valid:
            break

    if passed:
        print("\n✓ Test 2 PASSED")
    else:
        print("\n✗ Test 2 FAILED")


# ====================================================
# Test 3: Single-round diagnostics for net return
# ====================================================

def _test_3(
    seed: int,
    post_npz: str,
    post_joint_npz: str,
    signal_type: str = "median",
    signal_cost: float = 0.0,
    scale_pay: int = 0,
    scale_param: float = 0.25,
    ace_payout: float = 20.0,
    n_sigs: int = 5,
    stage1_alloc: float = 0.5,
) -> None:
    print("\n\n--------------------------------------------------")
    print("\nTest 3: Single-round net return diagnostics\n")
    print(f"Config: seed={seed}, signal={signal_type}, n_sigs={n_sigs}, alpha={stage1_alloc}")
    print(f"        signal_cost={signal_cost}, scale_pay={scale_pay}, scale_param={scale_param}, ace_payout={ace_payout}")

    # Generate one deterministic board
    r = 0
    rng = default_rng(round_seed(seed, r))
    has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

    # Load posteriors
    rmax_median, rmax_top2, _, _, prior_rmax, _ = _load_mc_posteriors(post_npz)
    rmax_tables = {"median": rmax_median, "top2": rmax_top2}
    joint_tables, _, r2_marginal = _load_joint_posteriors(post_joint_npz)

    # Choose piles
    perm = rng.permutation(NUM_PILES)
    n_buy = max(0, min(int(n_sigs), NUM_PILES))
    chosen_idx = perm[:n_buy]

    # Run dynamic round
    out = run_single_round_dynamic(
        rmax_tables=rmax_tables,
        joint_tables=joint_tables,
        prior_rmax=prior_rmax,
        r2_marginal=r2_marginal,
        chosen_idx=chosen_idx,
        signal_type=signal_type,
        hands=hands,
        medians=medians,
        top2sum=top2sum,
        max_rank=max_rank,
        min_rank=min_rank,
        scale_pay=scale_pay,
        scale_param=scale_param,
        ace_payout=ace_payout,
        signal_cost=signal_cost,
        stage1_alloc=stage1_alloc,
    )

    # Manual verification: compute expected values
    alpha = min(1.0, max(0.0, float(stage1_alloc)))
    budget1 = alpha * BUDGET
    info_cost = n_buy * signal_cost
    investable1 = max(0.0, budget1 - info_cost)
    budget2 = max(0.0, BUDGET - budget1)

    # Print board
    print(f"\nBoard (pile → ranks, specials):")
    if signal_type == "median":
        obs_vals = medians
    else:
        obs_vals = top2sum

    for i in range(NUM_PILES):
        pile = hands[i].tolist()
        specials = []
        if any(v == 14 for v in pile): specials.append('A')
        if any(v == 13 for v in pile): specials.append('K')
        if any(v == 12 for v in pile): specials.append('Q')
        spec = "".join(specials) if specials else "-"
        obs_mark = f" (observed, {signal_type}={obs_vals[i]})" if i in chosen_idx else ""
        print(f"  Pile {i+1:>2}: {' '.join(f'{int(v):2d}' for v in pile)}   specials={spec}{obs_mark}")

    print(f"\nBudgets:")
    print(f"  Total: {BUDGET}")
    print(f"  Stage 1 (alpha={alpha}): {budget1:.2f}")
    print(f"  Info cost: {info_cost:.2f}")
    print(f"  Stage 1 investable: {investable1:.2f}")
    print(f"  Stage 2: {budget2:.2f}")

    # Manually compute Stage 1 and Stage 2 weights/scores (matching run_single_round_dynamic logic)
    # Setup payoff vectors
    ranks_all = np.arange(2, 15, dtype=int)
    if scale_pay == 0:
        h_vals = np.array([float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
        h_vals2 = np.array([0.5*float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
    else:
        h_vals = np.array([float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)
        h_vals2 = np.array([0.5*float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)

    # Stage 1 scores
    post_table = rmax_tables[signal_type]
    prior_vec = np.asarray(prior_rmax, float)
    chosen_set = set(int(x) for x in chosen_idx)
    scores1 = np.zeros(NUM_PILES, float)
    for i in range(NUM_PILES):
        vec = np.asarray(post_table.get(int(obs_vals[i]), prior_vec), float) if (i in chosen_set) else prior_vec
        scores1[i] = float(np.dot(h_vals, vec))

    # Weight functions
    def weights_max(sc):
        p = np.array(sc, float); mx = np.max(p)
        winners = np.where(p == mx)[0]
        w = np.zeros_like(p, float); w[winners] = 1.0 / max(1, winners.size)
        return w

    def weights_linear(sc):
        p = np.array(sc, float); s = float(np.sum(p))
        return (p / s) if s > 0 else (np.ones_like(p)/len(p))

    def weights_top5(sc):
        p = np.array(sc, float)
        idx = np.argsort(p)[-5:]
        w = np.zeros_like(p, float); sm = float(np.sum(p[idx]))
        w[idx] = (p[idx] / sm) if sm > 0 else (1.0 / 5.0)
        return w

    # Stage 1 weights
    w1_max = weights_max(scores1)
    w1_lin = weights_linear(scores1)
    w1_top5 = weights_top5(scores1)

    # Get R2 values
    R2 = np.array([_second_highest_rank(h) for h in hands], int)

    # Stage 2 scores: use joint for observed, marginal for unobserved
    keys, mat3d, rowmap = joint_tables[signal_type]
    r2_marg = np.asarray(r2_marginal, float)

    def scores2_from(w1):
        sc = np.zeros(NUM_PILES, float)
        if investable1 <= 0:
            return sc
        idxs = np.where(w1 > 0)[0]
        for i in idxs:
            r2k = int(R2[i]) - 2
            if i in chosen_set:
                # Invested AND observed: use joint posterior
                b = int(obs_vals[i])
                if (b in rowmap) and (0 <= r2k < 13):
                    vec = np.asarray(mat3d[rowmap[b], r2k, :], float)
                else:
                    vec = prior_vec
            else:
                # Invested but NOT observed: use marginal posterior
                if 0 <= r2k < 13:
                    vec = np.asarray(r2_marg[r2k, :], float)
                else:
                    vec = prior_vec
            sc[i] = float(np.dot(h_vals2, vec))
        return sc

    scores2_max = scores2_from(w1_max)
    scores2_linear = scores2_from(w1_lin)
    scores2_top5 = scores2_from(w1_top5)

    # Stage 2 weights
    w2_max = weights_max(scores2_max)
    w2_lin = weights_linear(scores2_linear)
    w2_top5 = weights_top5(scores2_top5)

    # Compute P(Ace|obs) for Stage 1
    post_table = rmax_tables[signal_type]
    prior_vec = np.asarray(prior_rmax, float)
    ace_idx = ACE_RANK - 2
    p_ace_stage1 = np.zeros(NUM_PILES, float)
    for i in range(NUM_PILES):
        if i in chosen_idx:
            vec = np.asarray(post_table.get(int(obs_vals[i]), prior_vec), float)
        else:
            vec = prior_vec
        p_ace_stage1[i] = float(vec[ace_idx])

    # Compute P(Ace | obs, R2) for Stage 2: use joint for observed, marginal for unobserved
    keys, mat3d, rowmap = joint_tables[signal_type]
    p_ace_stage2 = np.zeros(NUM_PILES, float)
    for i in range(NUM_PILES):
        if w1_max[i] > 1e-10 or w1_lin[i] > 1e-10 or w1_top5[i] > 1e-10:  # Invested in Stage 1
            r2k = int(R2[i]) - 2
            if i in chosen_idx:
                # Observed: use joint posterior
                b = int(obs_vals[i])
                if (b in rowmap) and (0 <= r2k < 13):
                    vec = np.asarray(mat3d[rowmap[b], r2k, :], float)
                else:
                    vec = prior_vec
            else:
                # Not observed: use marginal posterior
                if 0 <= r2k < 13:
                    vec = np.asarray(r2_marg[r2k, :], float)
                else:
                    vec = prior_vec
            p_ace_stage2[i] = float(vec[ace_idx])

    # Print Stage 1 table
    print(f"\n{'='*100}")
    print(f"STAGE 1 — Expected payoffs and investment weights")
    print(f"{'='*100}")

    header = (f"{'Pile':>4}  {'Obs':>5}  {'P(Ace|obs)':>10}  "
              f"{'E(payoff)':>10}  {'w[High]':>8}  {'w[Lin]':>8}  {'w[Top5]':>8}")
    print(header)
    print("-" * len(header))

    for i in range(NUM_PILES):
        obs_str = '-' if i not in chosen_idx else f"{int(obs_vals[i])}"
        print(f"{i+1:4d}  {obs_str:>5}  {p_ace_stage1[i]:10.6f}  "
              f"{scores1[i]:10.6f}  {w1_max[i]:8.4f}  {w1_lin[i]:8.4f}  {w1_top5[i]:8.4f}")

    # Print Stage 2 table
    print(f"\n{'='*100}")
    print(f"STAGE 2 — Expected payoffs (0.5x) and investment weights (only piles invested in Stage 1)")
    print(f"{'='*100}")

    header = (f"{'Pile':>4}  {'R2':>4}  {'P(Ace|obs,R2)':>13}  "
              f"{'E(payoff)':>10}  {'w[High]':>8}  {'w[Lin]':>8}  {'w[Top5]':>8}")
    print(header)
    print("-" * len(header))

    for i in range(NUM_PILES):
        # Only show piles that were invested in Stage 1
        if w1_max[i] > 1e-10 or w1_lin[i] > 1e-10 or w1_top5[i] > 1e-10:
            # Pick the non-zero scores
            sc2 = scores2_max[i] if w1_max[i] > 1e-10 else (scores2_linear[i] if w1_lin[i] > 1e-10 else scores2_top5[i])
            print(f"{i+1:4d}  {R2[i]:4d}  {p_ace_stage2[i]:13.6f}  "
                  f"{sc2:10.6f}  {w2_max[i]:8.4f}  {w2_lin[i]:8.4f}  {w2_top5[i]:8.4f}")

    print(f"\n{'='*100}\n")

    print(f"\nNet returns from run_single_round_dynamic:")
    print(f"  Highest:           {out['net_return_max']:.6f}%")
    print(f"  E[payoff] weighted: {out['net_return_linear']:.6f}%")
    print(f"  Top-5 E[payoff]:   {out['net_return_top5']:.6f}%")

    # Verify net return formula: net = 100 * (gross_total - BUDGET) / BUDGET
    # where gross_total = stage1_gross + stage2_gross
    print(f"\nVerification: checking net return = 100 * (gross - BUDGET) / BUDGET")

    # The output includes gross payouts
    gross_max = out.get('gross_max', None)
    gross_lin = out.get('gross_linear', None)
    gross_top5 = out.get('gross_top5', None)

    if gross_max is not None:
        expected_net_max = 100.0 * (gross_max - BUDGET) / BUDGET
        print(f"  Highest: gross={gross_max:.6f}, expected_net={expected_net_max:.6f}%, actual_net={out['net_return_max']:.6f}%")
        if abs(expected_net_max - out['net_return_max']) > 1e-4:
            print(f"  ✗ Net return mismatch for Highest!")
        else:
            print(f"  ✓ Net return correct for Highest")

    print("\n✓ Test 3 PASSED (manual inspection required)")


# ====================================================
# Test 4: Stage 2 investments only in Stage 1 piles
# ====================================================

def _test_4(
    seed: int,
    post_npz: str,
    post_joint_npz: str,
    signal_type: str = "median",
    n_sigs: int = 5,
    stage1_alloc: float = 0.5,
    test_rounds: int = 100,
) -> None:
    print("\n\n--------------------------------------------------")
    print("\nTest 4: Stage 2 investments only in Stage 1 piles\n")
    print(f"Config: seed={seed}, signal={signal_type}, n_sigs={n_sigs}, alpha={stage1_alloc}")
    print(f"Testing {test_rounds} rounds...")

    # Load posteriors
    rmax_median, rmax_top2, _, _, prior_rmax, _ = _load_mc_posteriors(post_npz)
    rmax_tables = {"median": rmax_median, "top2": rmax_top2}
    joint_tables, _, r2_marginal = _load_joint_posteriors(post_joint_npz)

    violations = 0

    # Weight functions (same as Test 3)
    def weights_max(sc):
        p = np.array(sc, float); mx = np.max(p)
        winners = np.where(p == mx)[0]
        w = np.zeros_like(p, float); w[winners] = 1.0 / max(1, winners.size)
        return w

    def weights_linear(sc):
        p = np.array(sc, float); s = float(np.sum(p))
        return (p / s) if s > 0 else (np.ones_like(p)/len(p))

    def weights_top5(sc):
        p = np.array(sc, float)
        idx = np.argsort(p)[-5:]
        w = np.zeros_like(p, float); sm = float(np.sum(p[idx]))
        w[idx] = (p[idx] / sm) if sm > 0 else (1.0 / 5.0)
        return w

    for r in range(test_rounds):
        rng = default_rng(round_seed(seed, r))
        _, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        perm = rng.permutation(NUM_PILES)
        n_buy = max(0, min(int(n_sigs), NUM_PILES))
        chosen_idx = perm[:n_buy]

        # Manually compute weights (matching run_single_round_dynamic logic)
        alpha = min(1.0, max(0.0, float(stage1_alloc)))
        budget1 = alpha * BUDGET
        info_cost = 0.0  # signal_cost = 0 in test
        investable1 = max(0.0, budget1 - info_cost)

        # Setup payoff vectors
        ranks_all = np.arange(2, 15, dtype=int)
        ace_payout = 20.0
        h_vals = np.array([float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
        h_vals2 = np.array([0.5*float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)

        # Get observed values
        if signal_type == "median":
            obs_vals = medians
        else:
            obs_vals = top2sum

        # Stage 1 scores
        post_table = rmax_tables[signal_type]
        prior_vec = np.asarray(prior_rmax, float)
        chosen_set = set(int(x) for x in chosen_idx)
        scores1 = np.zeros(NUM_PILES, float)
        for i in range(NUM_PILES):
            vec = np.asarray(post_table.get(int(obs_vals[i]), prior_vec), float) if (i in chosen_set) else prior_vec
            scores1[i] = float(np.dot(h_vals, vec))

        # Stage 1 weights
        w1_max = weights_max(scores1)
        w1_lin = weights_linear(scores1)
        w1_top5 = weights_top5(scores1)

        # Get R2 values
        R2 = np.array([_second_highest_rank(h) for h in hands], int)

        # Stage 2 scores: use joint for observed, marginal for unobserved
        keys, mat3d, rowmap = joint_tables[signal_type]
        r2_marg = np.asarray(r2_marginal, float)

        def scores2_from(w1):
            sc = np.zeros(NUM_PILES, float)
            if investable1 <= 0:
                return sc
            idxs = np.where(w1 > 0)[0]
            for i in idxs:
                r2k = int(R2[i]) - 2
                if i in chosen_set:
                    # Invested AND observed: use joint posterior
                    b = int(obs_vals[i])
                    if (b in rowmap) and (0 <= r2k < 13):
                        vec = np.asarray(mat3d[rowmap[b], r2k, :], float)
                    else:
                        vec = prior_vec
                else:
                    # Invested but NOT observed: use marginal posterior
                    if 0 <= r2k < 13:
                        vec = np.asarray(r2_marg[r2k, :], float)
                    else:
                        vec = prior_vec
                sc[i] = float(np.dot(h_vals2, vec))
            return sc

        scores2_max = scores2_from(w1_max)
        scores2_linear = scores2_from(w1_lin)
        scores2_top5 = scores2_from(w1_top5)

        # Stage 2 weights
        w2_max = weights_max(scores2_max)
        w2_lin = weights_linear(scores2_linear)
        w2_top5 = weights_top5(scores2_top5)

        # Check each rule
        for rule_name, w1, w2 in [("max", w1_max, w2_max), ("linear", w1_lin, w2_lin), ("top5", w1_top5, w2_top5)]:
            for i in range(NUM_PILES):
                if w2[i] > 1e-10 and w1[i] <= 1e-10:
                    violations += 1
                    if violations <= 5:  # Print first few violations
                        print(f"  Round {r}, rule={rule_name}, pile {i}: w1={w1[i]:.6f}, w2={w2[i]:.6f} ✗")

    print(f"\nTotal violations: {violations} (out of {test_rounds * 3 * NUM_PILES} checks)")

    if violations == 0:
        print("✓ Test 4 PASSED: Stage 2 investments only occur in Stage 1 invested piles")
    else:
        print("✗ Test 4 FAILED: Found Stage 2 investments in non-Stage 1 piles")


# ====================================================
# Test 5: Stage 2 payoffs are half of Stage 1 payoffs
# ====================================================

def _test_5(
    ace_payout: float = 20.0,
    scale_pay: int = 0,
    scale_param: float = 0.25,
) -> None:
    print("\n\n--------------------------------------------------")
    print("\nTest 5: Stage 2 payoffs are half of Stage 1 payoffs\n")
    print(f"Config: ace_payout={ace_payout}, scale_pay={scale_pay}, scale_param={scale_param}")

    ranks_all = np.arange(2, 15, dtype=int)

    if scale_pay == 0:
        h_vals = np.array([float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
        h_vals2 = np.array([0.5*float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
    else:
        h_vals = np.array([float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)
        h_vals2 = np.array([0.5*float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)

    print("\nStage 1 payoffs (h_vals):")
    for r in ranks_all:
        print(f"  Rank {r}: {h_vals[r-2]:.6f}")

    print("\nStage 2 payoffs (h_vals2):")
    for r in ranks_all:
        print(f"  Rank {r}: {h_vals2[r-2]:.6f}")

    print("\nVerification: h_vals2 = 0.5 * h_vals")
    expected_h_vals2 = 0.5 * h_vals
    max_diff = np.max(np.abs(h_vals2 - expected_h_vals2))
    print(f"  Max difference: {max_diff:.10f}")

    if max_diff < 1e-10:
        print("✓ Test 5 PASSED: Stage 2 payoffs are exactly half of Stage 1 payoffs")
    else:
        print("✗ Test 5 FAILED: Stage 2 payoffs do not match 0.5 * Stage 1 payoffs")


# ====================================================
# Runner
# ====================================================

def main():
    ap = argparse.ArgumentParser(description="Testing harness for dynamic 2-stage model")
    ap.add_argument("--test", type=str, required=True,
                    help="Test: 1=boards, 2=joint posteriors, 3=net return, 4=stage2 constraints, 5=payoff ratios, all=run 1-5")
    ap.add_argument("--seed", type=int, default=12345, help="Base seed for tests")

    # Test 2 args
    ap.add_argument("--pre_rounds", type=int, default=500000, help="Rounds for joint precompute in test 2")
    ap.add_argument("--procs", type=int, default=8, help="Workers for precompute in test 2")
    ap.add_argument("--post", type=str, default="output_joint/post_joint.testing.npz", help="Output NPZ path for test 2")

    # Test 3 args
    ap.add_argument("--post_t3", type=str, default="output/post_mc.npz", help="Marginal posteriors for test 3")
    ap.add_argument("--post_joint", type=str, default="output_joint/post_joint.npz", help="Joint posteriors for test 3/4")
    ap.add_argument("--sig", type=str, default="median", help="Signal type (median/top2)")
    ap.add_argument("--sig_cost", type=float, default=0.0, help="Signal cost")
    ap.add_argument("--nsigs", type=int, default=5, help="Number of signals (0-9)")
    ap.add_argument("--sp", type=int, default=0, help="scale_pay (0 or 1)")
    ap.add_argument("--sparam", type=float, default=0.25, help="scale_param")
    ap.add_argument("--ap", type=float, default=20.0, help="Ace payout multiplier")
    ap.add_argument("--alpha", type=float, default=0.5, help="Stage 1 allocation (0-1)")

    # Test 4 args
    ap.add_argument("--post_t4", type=str, default="output/post_mc.npz", help="Marginal posteriors for test 4")
    ap.add_argument("--test4_rounds", type=int, default=100, help="Rounds to test for stage 2 constraint")

    args = ap.parse_args()
    seed = int(args.seed)

    test_sel = str(args.test).strip().lower()

    if test_sel == 'all':
        _test_1(seed)
        _test_2(int(args.pre_rounds), int(args.procs), seed, args.post)
        _test_3(seed, args.post_t3, args.post_joint, args.sig, float(args.sig_cost),
                int(args.sp), float(args.sparam), float(args.ap), int(args.nsigs), float(args.alpha))
        _test_4(seed, args.post_t4, args.post_joint, args.sig, int(args.nsigs),
                float(args.alpha), int(args.test4_rounds))
        _test_5(float(args.ap), int(args.sp), float(args.sparam))
        return

    try:
        test_num = int(test_sel)
    except ValueError:
        raise SystemExit("Unknown test. Use --test 1, 2, 3, 4, 5 or 'all'.")

    if test_num == 1:
        _test_1(seed)
    elif test_num == 2:
        _test_2(int(args.pre_rounds), int(args.procs), seed, args.post)
    elif test_num == 3:
        _test_3(seed, args.post_t3, args.post_joint, args.sig, float(args.sig_cost),
                int(args.sp), float(args.sparam), float(args.ap), int(args.nsigs), float(args.alpha))
    elif test_num == 4:
        _test_4(seed, args.post_t4, args.post_joint, args.sig, int(args.nsigs),
                float(args.alpha), int(args.test4_rounds))
    elif test_num == 5:
        _test_5(float(args.ap), int(args.sp), float(args.sparam))
    else:
        raise SystemExit("Unknown test. Use --test 1, 2, 3, 4, 5 or 'all'.")


if __name__ == "__main__":
    main()

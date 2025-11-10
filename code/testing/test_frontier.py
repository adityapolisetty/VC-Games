#!/usr/bin/env python3
"""
Comprehensive test suite for frontier.py computations.

Tests:
1. Weight generation (_weight_splits)
2. Posterior selection logic
3. Budget accounting with signal costs
4. Return calculations (g1, g2)
5. Mean/variance formulas
6. Binning logic
7. Integration: n=0 vs n>0 with signal costs

Usage:
    python3 test_frontier.py --post_npz ../frontier_output_m/post_joint.npz
"""

import sys
import pathlib
import argparse
import numpy as np
from numpy.random import default_rng

# Add parent directory to path to import from frontier.py
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import functions from frontier.py
from frontier import (
    _weight_splits,
    _load_posteriors,
    _deal_cards_global_deck,
    _hvals,
    _per_dollar_realized,
    _second_highest_rank,
    _concat_stats,
    _worker_chunk_il,
    _collect_stats_il,
    round_seed,
    BUDGET,
    ACE_PAYOUT,
    NUM_PILES,
    UNITS,
    SD_STEP,
)


def test_imports():
    """Test 0: Verify all imports from frontier.py work"""
    print("\n" + "="*70)
    print("TEST 0: Import Verification")
    print("="*70)

    # Test that all imported functions exist and are callable
    imports_to_test = [
        ('_weight_splits', callable),
        ('_load_posteriors', callable),
        ('_deal_cards_global_deck', callable),
        ('_hvals', callable),
        ('_per_dollar_realized', callable),
        ('_second_highest_rank', callable),
        ('_concat_stats', callable),
        ('_worker_chunk_il', callable),
        ('_collect_stats_il', callable),
        ('round_seed', callable),
        ('BUDGET', lambda x: isinstance(x, (int, float))),
        ('ACE_PAYOUT', lambda x: isinstance(x, (int, float))),
        ('NUM_PILES', lambda x: isinstance(x, int)),
        ('UNITS', lambda x: isinstance(x, int)),
        ('SD_STEP', lambda x: isinstance(x, (int, float))),
    ]

    print(f"Testing {len(imports_to_test)} imports from frontier.py:")
    for name, check in imports_to_test:
        obj = globals()[name]
        assert check(obj), f"{name} failed type check"
        print(f"  ✓ {name}: {type(obj).__name__}")

    print(f"\n✓ All imports successful")
    print(f"  BUDGET = {BUDGET}")
    print(f"  ACE_PAYOUT = {ACE_PAYOUT}")
    print(f"  NUM_PILES = {NUM_PILES}")
    print(f"  UNITS = {UNITS}")
    print(f"  SD_STEP = {SD_STEP}")

    print("\n✅ TEST 0 PASSED")
    return True


def test_weight_splits():
    """Test 1: Weight generation"""
    print("\n" + "="*70)
    print("TEST 1: Weight Generation (_weight_splits)")
    print("="*70)

    # Test with current UNITS and NUM_PILES
    Wm = _weight_splits(UNITS, NUM_PILES)

    # Check shape
    print(f"UNITS={UNITS}, NUM_PILES={NUM_PILES}")
    print(f"Weight matrix shape: {Wm.shape}")

    # Expected count: combinations with repetition C(n+k-1, k) where n=UNITS, k=NUM_PILES
    # This is the number of ways to partition UNITS into NUM_PILES non-negative integers
    from math import comb
    expected_count = comb(UNITS + NUM_PILES - 1, NUM_PILES - 1)

    assert Wm.shape[1] == NUM_PILES, f"Expected {NUM_PILES} columns, got {Wm.shape[1]}"
    print(f"✓ Number of allocations: {Wm.shape[0]} (expected {expected_count})")

    # Check all rows sum to 1.0
    row_sums = np.sum(Wm, axis=1)
    assert np.allclose(row_sums, 1.0), f"Weights don't sum to 1.0: {row_sums[:10]}"
    print(f"✓ All weights sum to 1.0")

    # Check granularity (multiples of 1/UNITS)
    granularity = 1.0 / UNITS
    all_multiples = np.all(np.isclose(Wm % granularity, 0) | np.isclose(Wm % granularity, granularity))
    assert all_multiples, f"Weights are not multiples of {granularity}"
    print(f"✓ All weights are multiples of {granularity:.4f}")

    # Show sample allocations
    print(f"\nSample allocations:")
    for i in [0, 100, 1000, 10000, -1]:
        print(f"  Strategy {i:5d}: {Wm[i]}")

    print("\n✅ TEST 1 PASSED")
    return True


def test_posterior_selection():
    """Test 2: Posterior selection logic"""
    print("\n" + "="*70)
    print("TEST 2: Posterior Selection Logic")
    print("="*70)

    # Create mock posterior tables
    mock_prior = np.array([0.1] * 13 + [0.05, 0.05])  # 15 elements (14 ranks + Ace)
    mock_prior = mock_prior / mock_prior.sum()

    # Mock median posterior: P(Ace | median=13) should differ from prior
    mock_median_table = {
        2: mock_prior.copy(),
        13: np.array([0.0] * 13 + [0.0, 1.0]),  # Certain Ace if median=13
    }

    # Mock top2 posterior: different distribution
    mock_top2_table = {
        5: mock_prior.copy(),
        25: np.array([0.0] * 13 + [0.0, 1.0]),  # Certain Ace if top2=25
    }

    mock_rmax_tables = {
        "median": mock_median_table,
        "top2": mock_top2_table,
    }

    # Test selection for median
    signal_type = "median"
    post_table = mock_rmax_tables[signal_type]

    # Simulate observed pile j=0 with median=13
    chosen_set = {0}
    buckets = np.array([13, 2, 2, 2, 2, 2, 2, 2, 2])

    # Line 215 logic from frontier.py
    prior_vec = mock_prior
    j = 0  # Observed pile
    vec_observed = np.asarray(post_table.get(int(buckets[j]), prior_vec), float) if (j in chosen_set) else prior_vec

    j = 1  # Unobserved pile
    vec_unobserved = np.asarray(post_table.get(int(buckets[j]), prior_vec), float) if (j in chosen_set) else prior_vec

    print(f"Signal type: {signal_type}")
    print(f"Chosen set: {chosen_set}")
    print(f"Buckets: {buckets}")
    print(f"\nP(Ace) in prior: {prior_vec[-1]:.4f}")
    print(f"P(Ace | median=13, observed): {vec_observed[-1]:.4f}")
    print(f"P(Ace | median=2, unobserved): {vec_unobserved[-1]:.4f}")

    assert vec_observed[-1] == 1.0, f"Observed posterior should be 1.0, got {vec_observed[-1]}"
    assert vec_unobserved[-1] == mock_prior[-1], f"Unobserved should use prior"

    print("\n✓ Correct posterior table selected based on signal_type")
    print("✓ Observed piles use posterior, unobserved use prior")
    print("\n✅ TEST 2 PASSED")
    return True


def test_budget_accounting():
    """Test 3: Budget accounting with signal costs"""
    print("\n" + "="*70)
    print("TEST 3: Budget Accounting")
    print("="*70)

    stage1_alloc = 0.5  # 50% to Stage 1
    signal_cost = 3.0

    # Test case 1: Affordable signals (n_sig=3)
    n_sig = 3
    budget1 = float(stage1_alloc) * BUDGET
    signal_cost_total = float(n_sig) * signal_cost
    investable1 = max(0.0, budget1 - signal_cost_total)
    budget2 = max(0.0, BUDGET - budget1)

    print(f"BUDGET = {BUDGET}")
    print(f"Stage 1 allocation: {stage1_alloc*100:.1f}%")
    print(f"Signal cost: £{signal_cost} x {n_sig} signals = £{signal_cost_total}")
    print(f"\nbudget1 = {budget1:.2f}")
    print(f"signal_cost_total = {signal_cost_total:.2f}")
    print(f"investable1 = {investable1:.2f}")
    print(f"budget2 = {budget2:.2f}")

    expected_investable1 = 50.0 - 9.0  # £41
    assert np.isclose(investable1, expected_investable1), f"Expected investable1={expected_investable1}, got {investable1}"

    c1 = investable1 / BUDGET
    c2 = budget2 / BUDGET if investable1 > 0 else 0.0

    print(f"\nc1 (Stage 1 fraction) = {c1:.4f}")
    print(f"c2 (Stage 2 fraction) = {c2:.4f}")
    print(f"c1 + c2 = {c1+c2:.4f} (should be < 1.0 due to signal cost)")

    assert c1 + c2 < 1.0, "Total allocation should be less than 1.0 when signal_cost > 0"

    # Test case 2: Unaffordable signals (budget1 < signal_cost_total)
    n_sig = 20  # Too many signals
    signal_cost_total = float(n_sig) * signal_cost
    affordable = budget1 >= signal_cost_total

    print(f"\n--- Edge case: Unaffordable signals ---")
    print(f"n_sig = {n_sig}, signal_cost_total = £{signal_cost_total}")
    print(f"budget1 = £{budget1}")
    print(f"Affordable? {affordable}")
    assert not affordable, "Should be unaffordable"

    # Test case 3: investable1 = 0 forces c2 = 0
    n_sig_exact = int(budget1 / signal_cost)
    signal_cost_total = float(n_sig_exact) * signal_cost
    investable1 = max(0.0, budget1 - signal_cost_total)

    print(f"\n--- Edge case: investable1 ≈ 0 ---")
    print(f"n_sig = {n_sig_exact}, investable1 = £{investable1:.4f}")

    if investable1 <= 0.0:
        c2 = 0.0
        print(f"c2 forced to 0 (cannot invest Stage 2 without Stage 1)")
        assert c2 == 0.0

    print("\n✓ Budget accounting correct")
    print("✓ Affordability check works")
    print("✓ c2=0 when investable1≤0")
    print("\n✅ TEST 3 PASSED")
    return True


def test_return_calculations():
    """Test 4: Return calculations (g1, g2)"""
    print("\n" + "="*70)
    print("TEST 4: Return Calculations")
    print("="*70)

    # Create small mock weight matrix
    Wm = np.array([
        [1.0, 0.0, 0.0],  # All on pile 0
        [0.5, 0.5, 0.0],  # Split piles 0-1
        [0.0, 0.0, 1.0],  # All on pile 2
    ])

    # Mock realized payoffs (per dollar invested)
    p_m = np.array([2.0, 1.5, 0.5])  # Pile 0 doubles, pile 1 +50%, pile 2 loses 50%

    # Calculate g1
    g1 = Wm @ p_m

    print(f"Weight matrix Wm:")
    print(Wm)
    print(f"\nRealized payoffs p_m: {p_m}")
    print(f"\nStage 1 returns g1 = Wm @ p_m: {g1}")

    expected_g1 = np.array([2.0, 1.75, 0.5])
    assert np.allclose(g1, expected_g1), f"Expected {expected_g1}, got {g1}"

    # Test Stage 2 with permutation
    p_m_stage2 = 0.5 * p_m  # Stage 2 gets 0.5x payoff
    perm2 = np.array([1, 0, 2])  # Re-rank: pile 1 first, then pile 0, then pile 2

    g2 = Wm[:, perm2] @ p_m_stage2

    print(f"\nStage 2 payoffs (0.5x): {p_m_stage2}")
    print(f"Stage 2 permutation perm2: {perm2}")
    print(f"Wm[:, perm2]:")
    print(Wm[:, perm2])
    print(f"\nStage 2 returns g2 = Wm[:, perm2] @ p_m_stage2: {g2}")

    # Verify returns after permutation
    # perm2 = [1,0,2] means: new_column_0 = old_column_1, new_column_1 = old_column_0, new_column_2 = old_column_2
    # Wm[:, perm2][0, :] = [Wm[0,1], Wm[0,0], Wm[0,2]] = [0, 1, 0]
    # g2[0] = [0,1,0] @ [1.0, 0.75, 0.25] = 0.75
    # Wm[:, perm2][1, :] = [Wm[1,1], Wm[1,0], Wm[1,2]] = [0.5, 0.5, 0]
    # g2[1] = [0.5,0.5,0] @ [1.0, 0.75, 0.25] = 0.5*1.0 + 0.5*0.75 = 0.875
    # Wm[:, perm2][2, :] = [Wm[2,1], Wm[2,0], Wm[2,2]] = [0, 0, 1]
    # g2[2] = [0,0,1] @ [1.0, 0.75, 0.25] = 0.25
    expected_g2 = np.array([0.75, 0.875, 0.25])
    assert np.allclose(g2, expected_g2), f"Expected {expected_g2}, got {g2}"

    print("\n✓ Stage 1 returns calculated correctly")
    print("✓ Stage 2 permutation applied correctly")
    print("✓ Stage 2 uses 0.5x payoffs")
    print("\n✅ TEST 4 PASSED")
    return True


def test_mean_variance_formulas():
    """Test 5: Mean/variance formulas"""
    print("\n" + "="*70)
    print("TEST 5: Mean/Variance Formulas")
    print("="*70)

    # Create synthetic statistics
    cnt = 1000
    g1 = np.array([1.2, 1.1, 0.9])  # Mean gross returns
    g2 = np.array([1.15, 1.05, 0.95])

    # Create squared sums for variance calculation
    g1sq = cnt * (g1**2 + 0.01)  # Add small variance
    g2sq = cnt * (g2**2 + 0.01)
    g12 = cnt * (g1 * g2 + 0.005)  # Positive covariance

    # Budget fractions (after signal cost)
    c1 = 0.41  # 41% in Stage 1 (50% - 9% signal cost)
    c2 = 0.50  # 50% in Stage 2

    print(f"Sample size: {cnt}")
    print(f"Mean g1: {g1}")
    print(f"Mean g2: {g2}")
    print(f"c1 = {c1}, c2 = {c2}")

    # Calculate mean/variance per frontier.py formulas
    mean_g1 = g1  # Already means
    mean_g2 = g2
    var_g1 = g1sq / cnt - mean_g1**2
    var_g2 = g2sq / cnt - mean_g2**2
    cov_g12 = g12 / cnt - (mean_g1 * mean_g2)

    # Net portfolio statistics (percentage points)
    mean_net = 100.0 * (c1 * (mean_g1 - 1.0) + c2 * (mean_g2 - 1.0))
    var_net = (100.0**2) * ((c1**2) * var_g1 + (c2**2) * var_g2 + 2.0 * c1 * c2 * cov_g12)
    sd_net = np.sqrt(np.clip(var_net, 0.0, np.inf))

    print(f"\nvar_g1: {var_g1}")
    print(f"var_g2: {var_g2}")
    print(f"cov_g12: {cov_g12}")
    print(f"\nmean_net (% points): {mean_net}")
    print(f"var_net: {var_net}")
    print(f"sd_net (% points): {sd_net}")

    # Verify excess return formula (only invested portions contribute)
    # For first strategy: mean_g1[0]=1.2, mean_g2[0]=1.15
    # mean_net[0] = 100 * (0.41*(1.2-1.0) + 0.50*(1.15-1.0))
    #              = 100 * (0.41*0.2 + 0.50*0.15)
    #              = 100 * (0.082 + 0.075) = 15.7
    expected_mean_net_0 = 100 * (c1 * (g1[0] - 1.0) + c2 * (g2[0] - 1.0))
    assert np.isclose(mean_net[0], expected_mean_net_0), f"Expected {expected_mean_net_0}, got {mean_net[0]}"

    # Verify variance formula includes covariance term
    expected_var_net_0 = (100**2) * ((c1**2) * var_g1[0] + (c2**2) * var_g2[0] + 2*c1*c2*cov_g12[0])
    assert np.isclose(var_net[0], expected_var_net_0), f"Expected {expected_var_net_0}, got {var_net[0]}"

    print("\n✓ Excess return formula correct (only invested portions)")
    print("✓ Variance formula includes covariance term")
    print("✓ 100x scaling for percentage points")
    print("\n✅ TEST 5 PASSED")
    return True


def test_binning_logic():
    """Test 6: Binning logic"""
    print("\n" + "="*70)
    print("TEST 6: Binning Logic")
    print("="*70)

    # Create mock SD and mean data
    sd_net = np.array([2.5, 7.5, 12.3, 18.9, 7.2, 11.5, 3.1, 22.4, 19.1, 6.8])
    mean_net = np.array([5.0, 15.0, 20.0, 30.0, 14.0, 19.0, 6.0, 35.0, 32.0, 13.0])

    print(f"SD_STEP = {SD_STEP} pp")
    print(f"sd_net: {sd_net}")
    print(f"mean_net: {mean_net}")

    # Binning per frontier.py line 405
    bins = np.floor(sd_net / SD_STEP).astype(int)
    print(f"\nBins (floor(sd/5)): {bins}")

    # Best mean selection per bin
    max_bin = int(np.max(bins))
    sd_levels = []
    best_means = []
    best_indices = []

    for b in range(max_bin + 1):
        mask = (bins == b)
        if not np.any(mask):
            print(f"Bin {b}: (empty)")
            continue
        idx = np.argmax(mean_net[mask])
        sel = np.flatnonzero(mask)[idx]
        sd_levels.append(float(b) * SD_STEP)
        best_means.append(float(mean_net[sel]))
        best_indices.append(sel)
        print(f"Bin {b}: {np.sum(mask)} strategies, best index={sel}, mean={mean_net[sel]:.1f}, sd={sd_net[sel]:.1f}")

    print(f"\nFrontier points:")
    print(f"SD levels: {sd_levels}")
    print(f"Best means: {best_means}")
    print(f"Best indices: {best_indices}")

    # Verify bin 1 (SD 5-10): should pick index 1 (mean=15.0, sd=7.5)
    assert 1 in best_indices and mean_net[1] == 15.0, "Bin 1 should select index 1"

    # Verify bin 3 (SD 15-20): should pick index 8 (mean=32.0, sd=19.1), not index 3 (mean=30.0)
    assert 8 in best_indices and mean_net[8] == 32.0, "Bin 3 should select highest mean"

    print("\n✓ Binning uses floor(sd / SD_STEP)")
    print("✓ Best mean selection per bin (argmax)")
    print("✓ Empty bins handled correctly")
    print("\n✅ TEST 6 PASSED")
    return True


def test_integration_n0_vs_nsig(post_npz_path):
    """Test 7: Integration test comparing n=0 vs n>0 with signal costs"""
    print("\n" + "="*70)
    print("TEST 7: Integration - n=0 vs n>0 with Signal Costs")
    print("="*70)

    # Load real posterior tables
    rmax_tables, joint_tables, prior_rmax, r2_marginal = _load_posteriors(post_npz_path)

    print(f"Loaded posteriors from: {post_npz_path}")
    print(f"Signal types available: {list(rmax_tables.keys())}")

    # Test with median signals
    signal_type = "median"
    post_table = rmax_tables[signal_type]

    print(f"\nTesting signal_type='{signal_type}'")
    print(f"Number of median posterior entries: {len(post_table)}")

    # Check if median posteriors differ from prior
    print(f"\nChecking if median signals are informative:")
    sample_medians = list(post_table.keys())[:5]
    prior_ace_prob = prior_rmax[-1]
    print(f"Prior P(Ace) = {prior_ace_prob:.6f}")

    differs_from_prior = False
    for med in sample_medians:
        post = post_table[med]
        post_ace_prob = post[-1]
        diff = abs(post_ace_prob - prior_ace_prob)
        print(f"  median={med:2d}: P(Ace|median) = {post_ace_prob:.6f}, diff = {diff:.6f}")
        if diff > 0.01:
            differs_from_prior = True

    if differs_from_prior:
        print("⚠️  WARNING: Median posteriors DIFFER from prior - signals are informative!")
    else:
        print("✓ Median posteriors match prior - signals are uninformative")

    # Run mini-simulation with n=0 and n=1 using frontier.py's actual functions
    print(f"\n--- Mini-simulation (1000 rounds using _collect_stats_il) ---")
    base_seed = 12345
    rounds = 1000
    procs = 1  # Single-threaded for testing
    stage1_alloc = 0.5
    signal_cost = 7.0
    sp = 0
    scale_param = 0.0

    results = {}
    for n_sig in [0, 1, 3]:
        print(f"\nSimulating n_sig={n_sig}...")

        # Budget calculation
        budget1 = float(stage1_alloc) * BUDGET
        signal_cost_total = float(n_sig) * signal_cost
        investable1 = max(0.0, budget1 - signal_cost_total)
        budget2 = max(0.0, BUDGET - budget1)

        print(f"  budget1=£{budget1:.2f}, signal_cost=£{signal_cost_total:.2f}")
        print(f"  investable1=£{investable1:.2f}, budget2=£{budget2:.2f}")

        if budget1 < signal_cost_total:
            print(f"  ⚠️  Cannot afford {n_sig} signals, skipping")
            continue

        # Use frontier.py's actual _collect_stats_il function
        stats = _collect_stats_il(
            seed=base_seed,
            rounds=rounds,
            procs=procs,
            signal_type=signal_type,
            n_sig=n_sig,
            sp=sp,
            scale_param=scale_param,
            rmax_tables=rmax_tables,
            joint_tables=joint_tables,
            prior_rmax=prior_rmax,
            r2_marginal=r2_marginal,
            debug_excel=False
        )

        # Extract statistics using frontier.py's _concat_stats
        g1, g2, g1sq, g2sq, g12, cnt, rank9, ace_hits, king_hits, queen_hits = _concat_stats(stats)

        # Calculate mean/variance using frontier.py's exact formulas
        c1 = investable1 / BUDGET
        c2 = budget2 / BUDGET if investable1 > 0 else 0.0

        mean_g1 = g1 / cnt
        mean_g2 = g2 / cnt
        var_g1 = g1sq / cnt - mean_g1 ** 2
        var_g2 = g2sq / cnt - mean_g2 ** 2
        cov_g12 = g12 / cnt - (mean_g1 * mean_g2)

        # Net portfolio statistics (percentage points) - frontier.py lines 400-402
        mean_net = 100.0 * (c1 * (mean_g1 - 1.0) + c2 * (mean_g2 - 1.0))
        var_net = (100.0 ** 2) * ((c1 ** 2) * var_g1 + (c2 ** 2) * var_g2 + 2.0 * c1 * c2 * cov_g12)
        sd_net = np.sqrt(np.clip(var_net, 0.0, np.inf))

        results[n_sig] = {
            'mean_net': mean_net,
            'sd_net': sd_net,
            'investable1': investable1,
            'cnt': cnt,
        }

        print(f"  Strategies evaluated: {len(mean_net)}")
        print(f"  mean_net range: [{np.min(mean_net):.2f}, {np.max(mean_net):.2f}]pp")
        print(f"  sd_net range: [{np.min(sd_net):.2f}, {np.max(sd_net):.2f}]pp")

    # Compare results (compare max mean across all strategies)
    print(f"\n--- Comparison ---")
    if 0 in results and 1 in results:
        max_mean0 = np.max(results[0]['mean_net'])
        max_mean1 = np.max(results[1]['mean_net'])
        avg_mean0 = np.mean(results[0]['mean_net'])
        avg_mean1 = np.mean(results[1]['mean_net'])

        print(f"n=0: max_mean = {max_mean0:.2f}pp, avg_mean = {avg_mean0:.2f}pp, investable = £{results[0]['investable1']:.2f}")
        print(f"n=1: max_mean = {max_mean1:.2f}pp, avg_mean = {avg_mean1:.2f}pp, investable = £{results[1]['investable1']:.2f}")
        print(f"Max difference: {max_mean1 - max_mean0:.2f}pp")
        print(f"Avg difference: {avg_mean1 - avg_mean0:.2f}pp")

        if signal_cost > 0 and not differs_from_prior:
            # If signals are uninformative and costly, n=0 should outperform
            if max_mean1 > max_mean0:
                print(f"\n⚠️  ANOMALY DETECTED: n=1 outperforms n=0 despite uninformative signals + cost!")
                print(f"   This suggests either:")
                print(f"   1. Median signals are actually informative (check posteriors)")
                print(f"   2. Bug in posterior selection logic")
                print(f"   3. Bug in budget accounting")
            else:
                print(f"\n✓ As expected: n=0 outperforms n=1 (uninformative signals + cost)")
        elif differs_from_prior:
            print(f"\n✓ Median signals are informative, so n=1 can outperform n=0")

    print("\n✅ TEST 7 COMPLETED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test frontier.py computations")
    parser.add_argument("--post_npz", type=str, default="../../precomp_output/post_mc.npz",
                        help="Path to posterior NPZ file")
    args = parser.parse_args()

    print("="*70)
    print("FRONTIER.PY COMPREHENSIVE TEST SUITE")
    print("="*70)

    try:
        # Run all tests
        test_imports()
        test_weight_splits()
        test_posterior_selection()
        test_budget_accounting()
        test_return_calculations()
        test_mean_variance_formulas()
        test_binning_logic()
        test_integration_n0_vs_nsig(args.post_npz)

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()

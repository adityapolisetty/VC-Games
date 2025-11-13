#!/usr/bin/env python3
"""
Profile frontier_v3.py to identify performance bottlenecks.

Run with: python3 profile_frontier_v3.py
"""

import cProfile
import pstats
import io
from pstats import SortKey

# Import the worker function we want to profile
from frontier_v3 import _worker_chunk_il, _load_posteriors, POST_NPZ_DEFAULT

def profile_worker():
    """Profile a single worker running a small number of rounds."""

    # Load posteriors (one-time cost, not in hot path)
    print("Loading posteriors...")
    rmax_tables, joint_tables, prior_rmax, r2_marginal = _load_posteriors(POST_NPZ_DEFAULT)

    # Profile parameters (small test)
    base_seed = 12345
    round_start = 0
    rounds_chunk = 10  # Just 10 rounds for profiling
    signal_type = "median"
    n_sig = 1  # 1 signal
    sp = 1  # scale_pay
    scale_param = 0.25

    print(f"\nProfiling {rounds_chunk} rounds with full Stage 1 + Stage 2 enumeration...")
    print("This will show where CPU time is actually spent.\n")

    # Create profiler
    profiler = cProfile.Profile()

    # Run the worker with profiling
    profiler.enable()
    stats = _worker_chunk_il(
        base_seed, round_start, rounds_chunk, signal_type, n_sig, sp, scale_param,
        rmax_tables, joint_tables, prior_rmax, r2_marginal, debug_excel=False
    )
    profiler.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)

    print("=" * 80)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 80)
    ps.sort_stats(SortKey.CUMULATIVE).print_stats(30)
    print(s.getvalue())

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    print("\n" + "=" * 80)
    print("TOP 30 FUNCTIONS BY TIME SPENT (excluding children)")
    print("=" * 80)
    ps.sort_stats(SortKey.TIME).print_stats(30)
    print(s.getvalue())

    # Summary statistics
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)
    print(f"Rounds simulated: {stats['count']}")
    print(f"Strategies evaluated: {len(stats['sum_g1'])} strategy pairs")
    print(f"Total strategies per round: {len(stats['sum_g1'])}")

    # Estimate time for full run
    total_time = ps.total_tt
    time_per_round = total_time / rounds_chunk
    rounds_full = 200000
    estimated_full = time_per_round * rounds_full / 3600  # hours

    print(f"\nTime for {rounds_chunk} rounds: {total_time:.2f} seconds")
    print(f"Time per round: {time_per_round:.3f} seconds")
    print(f"Estimated time for {rounds_full:,} rounds (1 core): {estimated_full:.1f} hours")
    print(f"Estimated time for {rounds_full:,} rounds (128 cores): {estimated_full/128:.1f} hours")

    print("\n" + "=" * 80)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)
    print("Look for:")
    print("1. Functions with high 'tottime' (time excluding subcalls)")
    print("2. Functions called millions of times (high ncalls)")
    print("3. Python loops vs vectorized numpy operations")
    print("=" * 80)


if __name__ == "__main__":
    profile_worker()

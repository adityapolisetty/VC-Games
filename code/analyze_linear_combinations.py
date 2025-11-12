#!/usr/bin/env python3
"""
analyze_linear_combinations.py — Expand frontier via linear combinations and save as NPZ

Assumption: Zero covariance between strategies
- Combined mean: μ_new = Σ(λ_i × μ_i)
- Combined variance: σ²_new = Σ(λ_i² × σ_i²)  [no cross terms]
- Combined SD: σ_new = √(σ²_new)

Output: NPZ file with expanded frontier in same format as frontier_v2.py
"""

import argparse
import json
import pathlib
import sys
from typing import List, Tuple, Dict, Any

import numpy as np

# Increase recursion limit for large frontier sizes
sys.setrecursionlimit(50000)


def _weight_splits(units: int, m: int) -> np.ndarray:
    """
    Enumerate all ways to split `units` into `m` parts (non-negative integers).
    Returns array of shape (n_combinations, m) with fractional weights.
    """
    out: List[List[int]] = []
    def rec(pos: int, remaining: int, cur: List[int]):
        if pos == m - 1:
            cur.append(remaining)
            out.append(cur.copy())
            cur.pop()
            return
        for t in range(remaining + 1):
            cur.append(t)
            rec(pos + 1, remaining - t, cur)
            cur.pop()
    rec(0, units, [])
    return (np.asarray(out, float) / float(units)).reshape((-1, m))


def load_frontier_data(npz_path: pathlib.Path) -> Dict[str, Any]:
    """
    Load minimal data from frontier NPZ file (only mean and SD).
    """
    with np.load(npz_path, allow_pickle=True) as data:
        result = {
            'sd_step': float(data['sd_step']),
            'sd_levels_by_n': data['sd_levels_by_n'],
            'best_means_by_n': data['best_means_by_n'],
            'meta': json.loads(str(data['meta'])),
        }
    return result


def expand_frontier_for_n_sig(orig_means: np.ndarray, orig_sds: np.ndarray,
                               units: int, sd_step: float, max_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand frontier for a single n_sig value using linear combinations.
    Only computes mean and SD.

    For large N, samples random combinations instead of enumerating all.
    """
    N = len(orig_means)
    if N == 0:
        return orig_sds, orig_means

    # Check if enumeration is feasible
    from math import comb
    n_combinations = comb(units + N - 1, N - 1)

    if n_combinations <= max_samples:
        # Small enough: enumerate all combinations
        mix_weights = _weight_splits(units, N)
    else:
        # Too large: sample random combinations using Dirichlet distribution
        rng = np.random.default_rng(42)
        alpha = np.ones(N)
        mix_weights = rng.dirichlet(alpha, size=max_samples)

    # Compute combined means and SDs (zero covariance assumption)
    comb_means = mix_weights @ orig_means
    variances = orig_sds ** 2
    comb_sds = np.sqrt((mix_weights**2) @ variances)

    # Combine all points (original + combinations)
    all_means = np.concatenate([orig_means, comb_means])
    all_sds = np.concatenate([orig_sds, comb_sds])

    # Bin by SD and keep best mean in each bin
    bins = np.floor(all_sds / sd_step).astype(int)
    max_bin = int(np.max(bins))

    exp_sds = []
    exp_means = []

    for b in range(max_bin + 1):
        mask = (bins == b)
        if not np.any(mask):
            continue
        best_idx = np.argmax(all_means[mask])
        sel_idx = np.flatnonzero(mask)[best_idx]
        exp_sds.append(float(b) * sd_step)
        exp_means.append(float(all_means[sel_idx]))

    return np.array(exp_sds), np.array(exp_means)


def expand_full_frontier(orig_data: Dict[str, Any], units: int) -> Dict[str, Any]:
    """
    Expand frontier for all n_sig values. Only computes mean and SD.
    """
    sd_step = orig_data['sd_step']
    n_sigs = len(orig_data['sd_levels_by_n'])

    new_sd_levels = []
    new_means = []

    for n_sig in range(n_sigs):
        orig_sds = np.array(orig_data['sd_levels_by_n'][n_sig], float)
        orig_means = np.array(orig_data['best_means_by_n'][n_sig], float)

        if len(orig_means) == 0:
            new_sd_levels.append(orig_sds)
            new_means.append(orig_means)
            continue

        exp_sds, exp_means = expand_frontier_for_n_sig(orig_means, orig_sds, units, sd_step)
        new_sd_levels.append(exp_sds)
        new_means.append(exp_means)

    new_meta = orig_data['meta'].copy()
    new_meta['expanded_via_linear_combinations'] = True
    new_meta['combination_units'] = int(units)

    return {
        'sd_step': sd_step,
        'sd_levels_by_n': np.array(new_sd_levels, dtype=object),
        'best_means_by_n': np.array(new_means, dtype=object),
        'meta': json.dumps(new_meta),
    }


def save_npz(out_path: pathlib.Path, data: Dict[str, Any]):
    """
    Save expanded frontier to NPZ file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import tempfile, os
    fd, tmp_path = tempfile.mkstemp(prefix=out_path.stem + '.', suffix='.npz', dir=str(out_path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **data)
        os.replace(tmp_path, str(out_path))
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def main():
    ap = argparse.ArgumentParser(description="Expand frontier via linear combinations")
    ap.add_argument("--npz", type=str, required=True, help="Path to original frontier NPZ file")
    ap.add_argument("--units", type=int, default=10, help="Weight granularity for combinations (default: 10)")
    ap.add_argument("--out_dir", type=str, default="frontier_expanded", help="Output directory")
    ap.add_argument("--suffix", type=str, default="_expanded", help="Suffix for output filename")

    args = ap.parse_args()

    npz_path = pathlib.Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    orig_data = load_frontier_data(npz_path)
    expanded_data = expand_full_frontier(orig_data, args.units)

    # Save
    out_path = pathlib.Path(args.out_dir) / f"{npz_path.stem}{args.suffix}.npz"
    save_npz(out_path, expanded_data)


if __name__ == "__main__":
    main()

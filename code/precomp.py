#!/usr/bin/env python3
"""
Precompute empirical posteriors for median and top2 signals.

Outputs (single NPZ in precomp_output/post_mc.npz):
- Joint posteriors: P(Rmax | signal, R2)
  - joint_median_keys, joint_median_mat
  - joint_top2_keys, joint_top2_mat
- Marginal posteriors: P(Rmax | signal)
  - rmax_median_keys, rmax_median_mat
  - rmax_top2_keys, rmax_top2_mat
- Common:
  - r2_marginal_mat: P(Rmax | R2)
  - prior_rmax: P(Rmax)

Usage (requires --procs >= 2):
  python precomp.py --seed 123 --rounds 500000 --procs 8
"""

import argparse
import os
import numpy as np
from numpy.random import default_rng
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sim_res import (round_seed, _deal_cards_global_deck, NUM_PILES, ACE_RANK)

RMAX_DIM = 13  # ranks 2..14 map to 0..12

def _second_highest_rank(arr: np.ndarray) -> int:
    """Return the second-highest UNIQUE rank value (not second position)."""
    unique_ranks = sorted(set(np.asarray(arr, int).tolist()), reverse=True)
    return int(unique_ranks[1]) if len(unique_ranks) >= 2 else int(unique_ranks[0])

def _acc_chunk(seed: int, start: int, rounds_chunk: int):
    # Joint posteriors: dict[bucket] -> counts[R2, Rmax]
    def _new_joint(): return np.zeros((RMAX_DIM, RMAX_DIM), dtype=np.int64)
    # Marginal posteriors: dict[bucket] -> counts[Rmax]
    def _new_marg(): return np.zeros(RMAX_DIM, dtype=np.int64)

    # Joint counts (for 2-stage mode)
    joint_med = {}
    joint_top2 = {}

    # Marginal counts
    marg_med = {}
    marg_top2 = {}

    prior = np.zeros((RMAX_DIM,), dtype=np.int64)
    # Marginal P(Rmax | R2) - aggregate across all buckets
    r2_marginal = np.zeros((RMAX_DIM, RMAX_DIM), dtype=np.int64)  # [R2, Rmax]
    piles = 0

    for i in range(int(rounds_chunk)):
        r = int(start) + int(i)
        rng = default_rng(round_seed(int(seed), r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
        for j in range(NUM_PILES):
            R2 = _second_highest_rank(hands[j])
            r2k = int(R2) - 2
            rmax = int(max_rank[j]) - 2
            m = int(medians[j])
            t = int(top2sum[j])

            piles += 1
            prior[rmax] += 1

            # R2 marginal: P(Rmax | R2)
            r2_marginal[r2k, rmax] += 1

            # Joint posteriors: P(Rmax | signal, R2)
            joint_med.setdefault(m, _new_joint())[r2k, rmax] += 1
            joint_top2.setdefault(t, _new_joint())[r2k, rmax] += 1

            # Marginal posteriors: P(Rmax | signal)
            marg_med.setdefault(m, _new_marg())[rmax] += 1
            marg_top2.setdefault(t, _new_marg())[rmax] += 1

    return joint_med, joint_top2, marg_med, marg_top2, prior, r2_marginal, piles

def _merge_joint(dlist):
    """Merge joint posterior dicts (2D arrays)."""
    out = {}
    for d in dlist:
        for k, v in d.items():
            out.setdefault(k, np.zeros((RMAX_DIM, RMAX_DIM), dtype=np.int64))
            out[k] += v
    return out

def _merge_marginal(dlist):
    """Merge marginal posterior dicts (1D arrays)."""
    out = {}
    for d in dlist:
        for k, v in d.items():
            out.setdefault(k, np.zeros(RMAX_DIM, dtype=np.int64))
            out[k] += v
    return out

def _accumulate(seed: int, rounds: int, procs: int):
    """Accumulate posteriors using parallel execution (requires procs >= 2)."""
    W = int(procs); base, rem = divmod(int(rounds), W)
    chunks = [(sum([base + (1 if i < rem else 0) for i in range(j)]), base + (1 if j < rem else 0)) for j in range(W)]
    chunks = [(s, c) for (s, c) in chunks if c > 0]

    # Collect parts from all workers
    joint_med_parts, joint_t2_parts = [], []
    marg_med_parts, marg_t2_parts = [], []
    prior_parts, r2_marg_parts = [], []
    piles_total = 0

    with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        futs = [ex.submit(_acc_chunk, int(seed), int(s), int(c)) for (s, c) in chunks]
        for (s, c), fut in zip(chunks, futs):
            joint_med, joint_t2, marg_med, marg_t2, prior, r2_marg, piles = fut.result()
            joint_med_parts.append(joint_med); joint_t2_parts.append(joint_t2)
            marg_med_parts.append(marg_med); marg_t2_parts.append(marg_t2)
            prior_parts.append(prior); r2_marg_parts.append(r2_marg)
            piles_total += int(piles)

    # Merge results from all workers
    joint_med = _merge_joint(joint_med_parts); joint_top2 = _merge_joint(joint_t2_parts)
    marg_med = _merge_marginal(marg_med_parts); marg_top2 = _merge_marginal(marg_t2_parts)
    prior_counts = np.sum(np.stack(prior_parts, axis=0), axis=0) if prior_parts else np.zeros(RMAX_DIM, np.int64)
    r2_marg_counts = np.sum(np.stack(r2_marg_parts, axis=0), axis=0) if r2_marg_parts else np.zeros((RMAX_DIM, RMAX_DIM), np.int64)
    total_piles = piles_total

    # Pack joint posteriors (3D: bucket x R2 x Rmax)
    def _pack_joint(d: dict[int, np.ndarray]):
        keys = sorted(d)
        if not keys:
            return np.array([], int), np.zeros((0, RMAX_DIM, RMAX_DIM), np.int64)
        mats = np.stack([d[k] for k in keys], axis=0)
        return np.array(keys, int), mats

    # Pack marginal posteriors (2D: bucket x Rmax)
    def _pack_marginal(d: dict[int, np.ndarray]):
        keys = sorted(d)
        if not keys:
            return np.array([], int), np.zeros((0, RMAX_DIM), np.int64)
        mats = np.stack([d[k] for k in keys], axis=0)
        return np.array(keys, int), mats

    # Pack joint posteriors
    joint_med_keys, joint_med_counts = _pack_joint(joint_med)
    joint_t2_keys,  joint_t2_counts  = _pack_joint(joint_top2)

    # Pack marginal posteriors
    marg_med_keys, marg_med_counts = _pack_marginal(marg_med)
    marg_t2_keys,  marg_t2_counts  = _pack_marginal(marg_top2)

    # Normalize joint posteriors (3D: row-wise over last axis)
    def _row_norm_3d(mat: np.ndarray) -> np.ndarray:
        if mat.size == 0:
            return mat.astype(float)
        s = mat.sum(axis=2, keepdims=True).astype(float)
        out = np.zeros_like(mat, float)
        mask = (s[...,0] > 0)
        out[mask] = mat[mask] / s[mask]
        return out

    # Normalize marginal posteriors (2D: row-wise over last axis)
    def _row_norm_2d(mat: np.ndarray) -> np.ndarray:
        if mat.size == 0:
            return mat.astype(float)
        s = mat.sum(axis=1, keepdims=True).astype(float)
        out = np.zeros_like(mat, float)
        mask = (s[:, 0] > 0)
        out[mask] = mat[mask] / s[mask]
        return out

    joint_med_prob = _row_norm_3d(joint_med_counts)
    joint_t2_prob  = _row_norm_3d(joint_t2_counts)
    marg_med_prob  = _row_norm_2d(marg_med_counts)
    marg_t2_prob   = _row_norm_2d(marg_t2_counts)

    prior_rmax = (prior_counts / float(prior_counts.sum())) if prior_counts.sum() > 0 else np.zeros(RMAX_DIM, float)

    # Normalize P(Rmax | R2) marginals: row-wise normalization
    r2_marg_prob = np.zeros_like(r2_marg_counts, dtype=float)
    for r2_idx in range(RMAX_DIM):
        row_sum = r2_marg_counts[r2_idx].sum()
        if row_sum > 0:
            r2_marg_prob[r2_idx] = r2_marg_counts[r2_idx] / float(row_sum)

    stats = dict(seed=int(seed), rounds=int(rounds), piles=int(total_piles), procs=int(procs))

    return (
        # Joint posteriors
        joint_med_keys, joint_med_prob, joint_t2_keys, joint_t2_prob,
        # Marginal posteriors
        marg_med_keys, marg_med_prob, marg_t2_keys, marg_t2_prob,
        # Common
        prior_rmax, r2_marg_prob, stats,
        # Counts for confidence intervals
        joint_med_counts, joint_t2_counts, marg_med_counts, marg_t2_counts, r2_marg_counts
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--out", type=str, default="../precomp_output/post_mc.npz")
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    print(f"\nRunning {args.rounds:,} rounds to compute posteriors (marginal + joint)")
    (
        joint_med_keys, joint_med_prob, joint_t2_keys, joint_t2_prob,
        marg_med_keys, marg_med_prob, marg_t2_keys, marg_t2_prob,
        prior_rmax, r2_marg_prob, stats,
        joint_med_counts, joint_t2_counts, marg_med_counts, marg_t2_counts, r2_marg_counts
    ) = _accumulate(args.seed, args.rounds, args.procs)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        # Joint posteriors: P(Rmax | signal, R2)
        joint_median_keys=joint_med_keys, joint_median_mat=joint_med_prob,
        joint_top2_keys=joint_t2_keys, joint_top2_mat=joint_t2_prob,
        # Marginal posteriors: P(Rmax | signal)
        rmax_median_keys=marg_med_keys, rmax_median_mat=marg_med_prob,
        rmax_top2_keys=marg_t2_keys, rmax_top2_mat=marg_t2_prob,
        # Common
        prior_rmax=prior_rmax,
        r2_marginal_mat=r2_marg_prob,
        meta=stats,
        # Counts for confidence intervals
        joint_median_counts=joint_med_counts,
        joint_top2_counts=joint_t2_counts,
        rmax_median_counts=marg_med_counts,
        rmax_top2_counts=marg_t2_counts,
        r2_marginal_counts=r2_marg_counts,
    )
    print(f"\nWrote posteriors to {out_path} | rounds={args.rounds}, piles={stats['piles']}")
    print(f"  - Joint: {len(joint_med_keys)} median buckets, {len(joint_t2_keys)} top2 buckets")
    print(f"  - Marginal: {len(marg_med_keys)} median, {len(marg_t2_keys)} top2")

if __name__ == "__main__":
    main()

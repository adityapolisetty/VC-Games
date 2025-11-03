#!/usr/bin/env python3
"""
Precompute joint empirical posteriors for the dynamic model:
P(Rmax | Stage‑1 bucket, R2) for bucket in {median, top2, max, min} and
second‑highest rank R2 in {2..14}.

Outputs (NPZ)
- joint_median_keys: int[K]
- joint_median_mat:  float[K, 13(R2), 13(Rmax)]
- joint_top2_keys:   int[T]
- joint_top2_mat:    float[T, 13, 13]
- joint_max_keys:    int[M]
- joint_max_mat:     float[M, 13, 13]
- joint_min_keys:    int[N]
- joint_min_mat:     float[N, 13, 13]
- prior_rmax:        float[13]

Usage
  python code/precomp_joint.py --seed 123 --rounds 500000 --out output/post_joint.npz --procs 8
"""

import argparse
import numpy as np
from numpy.random import default_rng
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from card_game import (round_seed, _deal_cards_global_deck, NUM_PILES, ACE_RANK)

RMAX_DIM = 13  # ranks 2..14 map to 0..12

def _second_highest_rank(arr: np.ndarray) -> int:
    a = np.sort(np.asarray(arr, int))
    return int(a[-2]) if a.size >= 2 else int(a[-1])

def _print_bar(cur: int, total: int) -> None:
    cur = int(cur); total = max(1, int(total))
    pct = int(100 * cur / total)
    barw = 30; filled = int(barw * cur / total)
    bar = "=" * filled + "-" * (barw - filled)
    print(f"\rProgress {cur}/{total} {pct:3d}% [{bar}]", end="", flush=True)

def _acc_chunk(seed: int, start: int, rounds_chunk: int):
    # dict[bucket] -> counts[R2, Rmax]
    def _new(): return np.zeros((RMAX_DIM, RMAX_DIM), dtype=np.int64)
    med = {}
    top2 = {}
    mx   = {}
    mn   = {}
    prior = np.zeros((RMAX_DIM,), dtype=np.int64)
    piles = 0
    for i in range(int(rounds_chunk)):
        r = int(start) + int(i)
        rng = default_rng(round_seed(int(seed), r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
        for j in range(NUM_PILES):
            R2 = _second_highest_rank(hands[j])
            r2k = int(R2) - 2
            rmax = int(max_rank[j]) - 2
            piles += 1
            prior[rmax] += 1
            # median
            m = int(medians[j]); med.setdefault(m, _new())[r2k, rmax] += 1
            # top2
            t = int(top2sum[j]); top2.setdefault(t, _new())[r2k, rmax] += 1
            # max
            M = int(max_rank[j]); mx.setdefault(M, _new())[r2k, rmax] += 1
            # min
            n = int(min_rank[j]); mn.setdefault(n, _new())[r2k, rmax] += 1
    return med, top2, mx, mn, prior, piles

def _merge(dlist):
    out = {}
    for d in dlist:
        for k, v in d.items():
            out.setdefault(k, np.zeros((RMAX_DIM, RMAX_DIM), dtype=np.int64))
            out[k] += v
    return out

def _accumulate(seed: int, rounds: int, procs: int):
    processed = 0; step_overall = max(1, int(rounds)//100)
    if procs and int(procs) > 1 and int(rounds) > 1:
        W = int(procs); base, rem = divmod(int(rounds), W)
        chunks = [(sum([base + (1 if i < rem else 0) for i in range(j)]), base + (1 if j < rem else 0)) for j in range(W)]
        chunks = [(s, c) for (s, c) in chunks if c > 0]
        med_parts, t2_parts, mx_parts, mn_parts, prior_parts, piles_total = [], [], [], [], [], 0
        with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
            futs = [ex.submit(_acc_chunk, int(seed), int(s), int(c)) for (s, c) in chunks]
            for (s, c), fut in zip(chunks, futs):
                m, t2, mx, mn, prior, piles = fut.result()
                med_parts.append(m); t2_parts.append(t2); mx_parts.append(mx); mn_parts.append(mn); prior_parts.append(prior); piles_total += int(piles)
                processed += int(c); _print_bar(processed, rounds)
        med = _merge(med_parts); top2 = _merge(t2_parts); mx = _merge(mx_parts); mn = _merge(mn_parts)
        prior_counts = np.sum(np.stack(prior_parts, axis=0), axis=0) if prior_parts else np.zeros(RMAX_DIM, np.int64)
        total_piles = piles_total
    else:
        med = {}; top2 = {}; mx = {}; mn = {}; prior_counts = np.zeros(RMAX_DIM, np.int64); total_piles = 0
        for r in range(int(rounds)):
            m, t2, x, n, prior, piles = _acc_chunk(int(seed), int(r), 1)
            for s, dst in ((m, med), (t2, top2), (x, mx), (n, mn)):
                for k, v in s.items(): dst.setdefault(k, np.zeros((RMAX_DIM,RMAX_DIM), np.int64)); dst[k] += v
            prior_counts += prior; total_piles += piles
            processed += 1
            if (processed % step_overall == 0) or (processed == int(rounds)):
                _print_bar(processed, rounds)

    def _pack(d: dict[int, np.ndarray]):
        keys = sorted(d)
        if not keys:
            return np.array([], int), np.zeros((0, RMAX_DIM, RMAX_DIM), np.int64)
        mats = np.stack([d[k] for k in keys], axis=0)
        return np.array(keys, int), mats

    med_keys, med_counts = _pack(med)
    t2_keys,  t2_counts  = _pack(top2)
    mx_keys,  mx_counts  = _pack(mx)
    mn_keys,  mn_counts  = _pack(mn)

    def _row_norm_3d(mat: np.ndarray) -> np.ndarray:
        if mat.size == 0:
            return mat.astype(float)
        s = mat.sum(axis=2, keepdims=True).astype(float)
        out = np.zeros_like(mat, float)
        mask = (s[...,0] > 0)
        out[mask] = mat[mask] / s[mask]
        return out

    med_prob = _row_norm_3d(med_counts)
    t2_prob  = _row_norm_3d(t2_counts)
    mx_prob  = _row_norm_3d(mx_counts)
    mn_prob  = _row_norm_3d(mn_counts)
    prior_rmax = (prior_counts / float(prior_counts.sum())) if prior_counts.sum() > 0 else np.zeros(RMAX_DIM, float)

    return (med_keys, med_prob, t2_keys, t2_prob, mx_keys, mx_prob, mn_keys, mn_prob, prior_rmax,
            dict(seed=int(seed), rounds=int(rounds), piles=int(total_piles), procs=int(procs)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--out", type=str, default="output/post_joint.npz")
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    print(f"Running {args.rounds:,} rounds to compute P(Rmax | Stage‑1 bucket, R2)")
    med_keys, med_prob, t2_keys, t2_prob, mx_keys, mx_prob, mn_keys, mn_prob, prior_rmax, stats = _accumulate(args.seed, args.rounds, args.procs)
    print()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             joint_median_keys=med_keys, joint_median_mat=med_prob,
             joint_top2_keys=t2_keys,  joint_top2_mat=t2_prob,
             joint_max_keys=mx_keys,   joint_max_mat=mx_prob,
             joint_min_keys=mn_keys,   joint_min_mat=mn_prob,
             prior_rmax=prior_rmax,
             meta=stats)
    print(f"Wrote joint posteriors to {out_path} | rounds={args.rounds}, piles={stats['piles']}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Precompute empirical posterior curves from sampled boards (deterministic).

Outputs
- rmax_median_keys: int[]              sorted median buckets observed
- rmax_median_mat:  float[K, 13]       rows are P(Rmax | median = key)
- rmax_top2_keys:   int[]              sorted top2sum buckets observed
- rmax_top2_mat:    float[T, 13]       rows are P(Rmax | top2sum = key)
- rmax_max_keys:    int[]              sorted max-rank buckets observed
- rmax_max_mat:     float[M, 13]       rows are P(Rmax | max = key)
- rmax_min_keys:    int[]              sorted min-rank buckets observed
- rmax_min_mat:     float[N, 13]       rows are P(Rmax | min = key)
- ace_median_probs: float[K]           last column of rmax_median_mat
- ace_top2_probs:   float[T]           last column of rmax_top2_mat
- ace_max_probs:    float[M]           last column of rmax_max_mat
- ace_min_probs:    float[N]           last column of rmax_min_mat
- prior_rmax:      float[13]           unconditional P(Rmax)
- meta:            dict                seed, rounds, piles, procs

Usage
  python precompute_posteriors.py --seed 123 --rounds 200000 --out output/post_mc.npz --procs 8

Method
- For each round r, use round_seed(seed, r), sample one board via
  _deal_cards_global_deck, then update per-pile histograms by bucket.
- Parallelized across disjoint round-id chunks; results are merged.
"""

import argparse
import numpy as np
from numpy.random import default_rng
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Import deal logic and helpers to exactly match simulation
from card_game import (round_seed, _deal_cards_global_deck, NUM_PILES, ACE_RANK)

# Rank domain 2..14 (Ace). RMAX_DIM indexes r by (r-2).
RMAX_DIM = 13


def _print_bar(cur: int, total: int) -> None:
    """Simple textual progress bar (updates in-place)."""
    cur = int(cur); total = max(1, int(total))
    pct = int(100 * cur / total)
    barw = 30
    filled = int(barw * cur / total)
    bar = "=" * filled + "-" * (barw - filled)
    print(f"\rProgress {cur}/{total} {pct:3d}% [{bar}]", end="", flush=True)


def _accumulate_chunk(seed: int, start: int, rounds_chunk: int):
    """Accumulate bucketed Rmax counts for rounds [start, start+rounds_chunk)."""
    med_counts: dict[int, np.ndarray] = {}
    t2_counts:  dict[int, np.ndarray] = {}
    max_counts: dict[int, np.ndarray] = {}
    min_counts: dict[int, np.ndarray] = {}
    prior_counts = np.zeros(RMAX_DIM, dtype=np.int64)
    piles = 0

    for i in range(int(rounds_chunk)):
        r = int(start) + int(i)
        rng = default_rng(round_seed(int(seed), r))
        has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
        for j in range(NUM_PILES):
            m = int(medians[j]); t = int(top2sum[j]); rmax = int(max_rank[j]); mx = int(max_rank[j]); mn = int(min_rank[j])
            piles += 1
            idx = rmax - 2
            med_counts.setdefault(m, np.zeros(RMAX_DIM, dtype=np.int64))[idx] += 1
            t2_counts.setdefault(t,  np.zeros(RMAX_DIM, dtype=np.int64))[idx] += 1
            max_counts.setdefault(mx, np.zeros(RMAX_DIM, dtype=np.int64))[idx] += 1
            min_counts.setdefault(mn, np.zeros(RMAX_DIM, dtype=np.int64))[idx] += 1
            prior_counts[idx] += 1
    return med_counts, t2_counts, max_counts, min_counts, prior_counts, piles


def _merge_counts(parts: list[dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for d in parts:
        for k, v in d.items():
            out.setdefault(k, np.zeros(RMAX_DIM, dtype=np.int64))
            out[k] += v
    return out


def _accumulate(seed: int, rounds: int, procs: int):
    """Accumulate empirical counts (parallel across round chunks)."""
    processed = 0
    step_overall = max(1, int(rounds) // 100)
    if procs and int(procs) > 1 and int(rounds) > 1:
        W = int(procs)
        base, rem = divmod(int(rounds), W)
        chunks = [(sum([base + (1 if i < rem else 0) for i in range(j)]), base + (1 if j < rem else 0)) for j in range(W)]
        chunks = [(s, c) for (s, c) in chunks if c > 0]
        med_parts, t2_parts, max_parts, min_parts, prior_parts, piles_total = [], [], [], [], [], 0
        with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
            futs = [ex.submit(_accumulate_chunk, int(seed), int(start), int(sz)) for (start, sz) in chunks]
            for (start, sz), fut in zip(chunks, futs):
                m, t2, mx, mn, pcounts, piles = fut.result()
                med_parts.append(m); t2_parts.append(t2); max_parts.append(mx); min_parts.append(mn); prior_parts.append(pcounts); piles_total += int(piles)
                processed += int(sz)
                _print_bar(processed, rounds)
        med_counts = _merge_counts(med_parts)
        t2_counts  = _merge_counts(t2_parts)
        max_counts = _merge_counts(max_parts)
        min_counts = _merge_counts(min_parts)
        prior_counts = np.sum(np.stack(prior_parts, axis=0), axis=0) if prior_parts else np.zeros(RMAX_DIM, np.int64)
        total_piles = piles_total
    else:
        med_counts: dict[int, np.ndarray] = {}
        t2_counts: dict[int, np.ndarray] = {}
        max_counts: dict[int, np.ndarray] = {}
        min_counts: dict[int, np.ndarray] = {}
        prior_counts = np.zeros(RMAX_DIM, dtype=np.int64)
        total_piles = 0
        # Manually inline the chunk loop to allow progress updates
        for i in range(int(rounds)):
            r = i
            m, t2, mx, mn, pcounts, piles = _accumulate_chunk(int(seed), int(r), 1)
            # Merge per-iteration results
            for d_src, d_dst in ((m, med_counts), (t2, t2_counts), (mx, max_counts), (mn, min_counts)):
                for k, v in d_src.items():
                    d_dst.setdefault(k, np.zeros(RMAX_DIM, dtype=np.int64))
                    d_dst[k] += v
            prior_counts += pcounts
            total_piles += int(piles)
            processed += 1
            if (processed % step_overall == 0) or (processed == int(rounds)):
                _print_bar(processed, rounds)

    # Pack dicts into aligned arrays sorted by bucket key
    def _pack(mat_dict: dict[int, np.ndarray]):
        keys = sorted(mat_dict)
        mat = np.stack([mat_dict[k] for k in keys], axis=0) if keys else np.zeros((0, RMAX_DIM), np.int64)
        return np.array(keys, dtype=int), mat

    med_keys, med_mat_counts = _pack(med_counts)
    t2_keys,  t2_mat_counts  = _pack(t2_counts)
    max_keys, max_mat_counts = _pack(max_counts)
    min_keys, min_mat_counts = _pack(min_counts)

    def _row_normalize(mat: np.ndarray) -> np.ndarray:
        """Row-normalize counts to probabilities (safe for zero rows)."""
        if mat.size == 0:
            return mat.astype(float)
        s = mat.sum(axis=1, keepdims=True).astype(float)
        out = np.zeros_like(mat, float)
        nz = s[:, 0] > 0
        out[nz] = mat[nz] / s[nz]
        return out

    med_mat_prob = _row_normalize(med_mat_counts)
    t2_mat_prob  = _row_normalize(t2_mat_counts)
    max_mat_prob = _row_normalize(max_mat_counts)
    min_mat_prob = _row_normalize(min_mat_counts)

    # Ensure extremely rare edge bucket exists deterministically:
    # top2sum == 5 implies Rmax == 3 with probability 1 (pile {3,2,2,2,2}).
    # If it was not observed empirically, add a one-hot row now.
    if t2_keys.size == 0:
        t2_keys = np.array([], dtype=int)
    if 5 not in set(t2_keys.tolist()):
        vec = np.zeros(RMAX_DIM, float)
        vec[3 - 2] = 1.0  # r=3 maps to index 1
        if t2_mat_prob.size == 0:
            t2_mat_prob = vec.reshape(1, -1)
            t2_keys = np.array([5], dtype=int)
        else:
            t2_mat_prob = np.vstack([t2_mat_prob, vec])
            t2_keys = np.concatenate([t2_keys, np.array([5], dtype=int)])
        order = np.argsort(t2_keys)
        t2_keys = t2_keys[order]
        t2_mat_prob = t2_mat_prob[order]

    # Guarantee 2-D shape for single-bucket cases so column slicing works
    def _ensure_2d(a: np.ndarray) -> np.ndarray:
        arr = np.asarray(a, float)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    med_mat_prob = _ensure_2d(med_mat_prob)
    t2_mat_prob  = _ensure_2d(t2_mat_prob)
    max_mat_prob = _ensure_2d(max_mat_prob)
    min_mat_prob = _ensure_2d(min_mat_prob)

    # P(Ace | bucket) is the last column in Rmax-prob mats
    ace_col = ACE_RANK - 2
    ace_median_probs = med_mat_prob[:, ace_col] if med_mat_prob.size else np.zeros((0,), float)
    ace_top2_probs   = t2_mat_prob[:,  ace_col] if t2_mat_prob.size  else np.zeros((0,), float)
    ace_max_probs    = max_mat_prob[:,  ace_col] if max_mat_prob.size  else np.zeros((0,), float)
    ace_min_probs    = min_mat_prob[:,  ace_col] if min_mat_prob.size  else np.zeros((0,), float)
    prior_rmax = (prior_counts / float(prior_counts.sum())) if prior_counts.sum() > 0 else np.zeros(RMAX_DIM, float)

    stats = dict(seed=int(seed), rounds=int(rounds), piles=int(total_piles), procs=int(procs))

    return (
        med_keys, med_mat_prob,
        t2_keys, t2_mat_prob,
        max_keys, max_mat_prob,
        min_keys, min_mat_prob,
        ace_median_probs, ace_top2_probs, ace_max_probs, ace_min_probs,
        prior_rmax,
        stats,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--out", type=str, default="output/post_mc.npz")
    ap.add_argument("--procs", type=int, default=8, help="workers for precompute (default: 8)")
    args = ap.parse_args()

    # Intro message
    print(
        f"Running {args.rounds:,} simulations of board draws to compute Prob(Max rank observed | signal value) under our board restrictions"
    )

    (
        med_keys, med_mat_prob,
        t2_keys, t2_mat_prob,
        max_keys, max_mat_prob,
        min_keys, min_mat_prob,
        ace_median_probs, ace_top2_probs, ace_max_probs, ace_min_probs,
        prior_rmax,
        stats,
    ) = _accumulate(args.seed, args.rounds, args.procs)

    # Finish the progress line
    print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        rmax_median_keys=med_keys,
        rmax_median_mat=med_mat_prob,
        rmax_top2_keys=t2_keys,
        rmax_top2_mat=t2_mat_prob,
        rmax_max_keys=max_keys,
        rmax_max_mat=max_mat_prob,
        rmax_min_keys=min_keys,
        rmax_min_mat=min_mat_prob,
        ace_median_probs=ace_median_probs,
        ace_top2_probs=ace_top2_probs,
        ace_max_probs=ace_max_probs,
        ace_min_probs=ace_min_probs,
        prior_rmax=prior_rmax,
        meta=stats,
    )

    print(f"Wrote empirical posteriors to {out_path} | rounds={args.rounds}, piles={stats['piles']}")

    # Quick visibility: list which signal buckets are present in the NPZ
    try:
        with np.load(out_path, allow_pickle=False) as z:
            for name in [
                'rmax_median_keys',
                'rmax_top2_keys',
                'rmax_max_keys',
                'rmax_min_keys',
            ]:
                if name in z.files:
                    vals = sorted(set(z[name].astype(int).tolist()))
                    print(name, vals)
                else:
                    print(name, 'MISSING')
    except Exception as e:
        print('Warning: could not reopen NPZ to list keys:', e)


if __name__ == "__main__":
    main()

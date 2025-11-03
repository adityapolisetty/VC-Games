"""Testing harness for card_game deck logic.

Currently implemented sections:
  1. Stress-test _deal_cards_global_deck and confirm deck constraints.
Run a section via `python code/unit_testing.py --test 1 ...`.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from numpy.random import default_rng

from card_game import _deal_cards_global_deck, NUM_PILES, CARDS_PER_PILE, ACE_RANK, round_seed


KING_RANK = 13
QUEEN_RANK = 12


def _trial_chunk(start: int, size: int, seed: int, total_cards_expected: int, allowed: np.ndarray):
    max_observed = np.zeros(13, dtype=int)
    violation_count = 0
    sample = []

    for offset in range(size):
        idx = start + offset
        rng = default_rng(round_seed(seed, idx))
        _, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

        flat = np.concatenate(hands)
        if flat.size != total_cards_expected:
            violation_count += 1
            if len(sample) < 5:
                sample.append((idx, "card_total", flat.size))
            continue

        unique, counts = np.unique(flat, return_counts=True)
        counts_map = dict(zip(unique.tolist(), counts.tolist()))

        for rank in range(2, 15):
            observed = counts_map.get(rank, 0)
            ridx = rank - 2
            if observed > max_observed[ridx]:
                max_observed[ridx] = observed
            if observed > allowed[ridx]:
                violation_count += 1
                if len(sample) < 5:
                    sample.append((idx, rank, observed))

    return max_observed, violation_count, sample


def run_trials(runs: int, seed: int, procs: int):
    total_cards_expected = NUM_PILES * CARDS_PER_PILE
    allowed = np.full(13, 4, dtype=int)
    allowed[ACE_RANK - 2] = 1
    allowed[KING_RANK - 2] = 2
    allowed[QUEEN_RANK - 2] = 2

    procs = max(1, int(procs))

    if procs == 1:
        max_observed, violation_count, sample = _trial_chunk(0, int(runs), seed, total_cards_expected, allowed)
        return allowed, max_observed, violation_count, sample

    chunks = []
    runs = int(runs)
    base, rem = divmod(runs, procs)
    start = 0
    for i in range(procs):
        size = base + (1 if i < rem else 0)
        if size > 0:
            chunks.append((start, size))
            start += size

    max_observed = np.zeros(13, dtype=int)
    violation_total = 0
    sample = []

    with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        futures = [
            ex.submit(_trial_chunk, start, size, seed, total_cards_expected, allowed)
            for (start, size) in chunks
        ]
        for fut in futures:
            chunk_max, chunk_count, chunk_sample = fut.result()
            max_observed = np.maximum(max_observed, chunk_max)
            violation_total += chunk_count
            for entry in chunk_sample:
                if len(sample) < 5:
                    sample.append(entry)

    return allowed, max_observed, violation_total, sample


def plot_results(allowed: np.ndarray, observed: np.ndarray, out_path: Path):
    ranks = np.arange(2, 15)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ranks - 0.15, allowed, width=0.3, label="Allowed", color="#CCCCCC")
    ax.bar(ranks + 0.15, observed, width=0.3, label="Observed max", color="#141450")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count on board")
    ax.set_xticks(ranks)
    ax.set_title("_deal_cards_global_deck counts across runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


SEPARATOR = "=" * 80


def _banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)


def run_section_1(runs: int, seed: int, out: Path, procs: int) -> None:
    _banner("Section 1 — Deal generator constraint check (start)")
    print(f"Runs: {runs:,} | Seed: {seed} | Workers: {procs}")
    allowed, observed, violation_count, sample = run_trials(runs, seed, procs)

    if violation_count:
        msg = "\n".join(
            f"Run {idx}: rank={rank} count={count}" if kind != "card_total" else f"Run {idx}: total cards {count}"
            for idx, kind, count in sample
        )
        raise AssertionError(
            f"Detected {violation_count} violations of deck constraints. Sample:\n{msg}"
        )

    print("All runs satisfied deck constraints.")
    print(f"Observed maximum count per rank: {observed.tolist()}")
    plot_results(allowed, observed, out)
    print(f"Saved comparison plot to {out}")
    _banner("Section 1 — Deal generator constraint check (end)")


def main():
    ap = argparse.ArgumentParser(description="Unit testing harness for card_game.")
    ap.add_argument("--test", type=int, default=None, help="Test section number (currently only 1). Omit to run all.")
    ap.add_argument("--runs", type=int, default=1_000_000, help="Number of deals to simulate (default: 1,000,000)")
    ap.add_argument("--seed", type=int, default=12345, help="Base RNG seed")
    ap.add_argument("--out", type=Path, default=Path("deal_constraint_check.png"), help="Output plot path (section 1)")
    ap.add_argument("--procs", type=int, default=8, help="Workers for section 1 (default: 8 cores)")
    args = ap.parse_args()

    sections = {1: lambda: run_section_1(int(args.runs), int(args.seed), Path(args.out), int(args.procs))}

    if args.test is None:
        for idx in sorted(sections):
            sections[idx]()
    else:
        if args.test not in sections:
            raise SystemExit("Unknown --test value. Currently supported sections: 1")
        sections[args.test]()


if __name__ == "__main__":
    main()

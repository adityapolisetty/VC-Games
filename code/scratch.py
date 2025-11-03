"""
Simple one-board flow to exercise card_game_v7 end-to-end with explicit settings:
 - Free signals (signal_cost = 0)
 - Ace-only payout (scale_pay = 0)
 - Ace payoff = 20x

It samples one board, builds a single permutation of piles, chooses the first
n piles as observed (chosen_idx), and computes net returns for each weight
rule (highest, linear, top-5) for both signal types (median, top2).

Usage (examples):
  python3 unit_testing_v7.py                  # defaults: seed=12345, n_sig=9
  python3 unit_testing_v7.py --seed 999 --n_sig 5
"""

from __future__ import annotations

import argparse
import numpy as np

import card_game_v7 as v7


def flow_single_board(seed: int, n_sig: int,
                      ace_payout: float = 20.0,
                      scale_pay: int = 0,
                      scale_param: float = 0.25,
                      signal_cost: float = 0.0):
    """
    Sample one board and compute net returns for both signal types using
    the v7 pipeline (no RNG inside run_single_round). Returns a dict.
    """
    # Precompute highest-card posteriors once (same as main flow)
    rmax_median, rmax_top2, prior_rmax = v7.build_rmax_posteriors()

    # Deterministic per-round RNG and board sampling
    rng = np.random.default_rng(v7.round_seed(seed, 0))
    has_ace, hands, medians, top2sum, max_rank = v7._deal_cards_global_deck(rng)
    pi = rng.permutation(v7.NUM_PILES)
    chosen_idx = pi[:int(n_sig)]

    cfg = dict(scale_pay=int(scale_pay), scale_param=float(scale_param),
               ace_payout=float(ace_payout), signal_cost=float(signal_cost))

    out = {}
    for st in ("median", "top2"):
        res = v7.run_single_round(
            rmax_median, rmax_top2, prior_rmax,
            chosen_idx=chosen_idx, signal_type=st,
            hands=hands, medians=medians, top2sum=top2sum,
            scale_pay=cfg["scale_pay"], scale_param=cfg["scale_param"],
            ace_payout=cfg["ace_payout"], signal_cost=cfg["signal_cost"],
        )
        out[st] = res

    return dict(
        seed=int(seed), n_sig=int(n_sig), chosen_idx=chosen_idx.tolist(),
        medians=list(map(int, medians)), top2sum=list(map(int, top2sum)),
        results=out,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n_sig", type=int, default=9)
    ap.add_argument("--mono", action="store_true", help="Run monotonicity test (card_game.py, requires --post_npz)")
    ap.add_argument("--post_npz", type=str, default=None, help="Empirical posteriors NPZ for monotonicity test")
    ap.add_argument("--rounds", type=int, default=1000, help="Rounds for expectation monotonicity test")
    args = ap.parse_args()

    if args.mono:
        # Expectation monotonicity test using empirical posteriors over many rounds.
        if not args.post_npz:
            raise SystemExit("--mono requires --post_npz pointing to precomputed posteriors NPZ")
        import card_game as cg
        rmax_m, rmax_t2, prior_r, _ = cg._load_mc_posteriors(args.post_npz)
        cfg = dict(scale_pay=0, scale_param=0.25, ace_payout=20.0, signal_cost=7.0)
        R = int(args.rounds)
        N = int(cg.NUM_PILES)
        tol = 1e-6
        for st in ("top2", "top2"):
            means = []
            for n in range(N + 1):
                vals = np.empty(R, float)
                for r in range(R):
                    rng = np.random.default_rng(cg.round_seed(args.seed, r))
                    has_ace, hands, medians, top2sum, max_rank = cg._deal_cards_global_deck(rng)
                    pi = rng.permutation(cg.NUM_PILES)
                    out = cg.run_single_round(
                        rmax_median=rmax_m, rmax_top2=rmax_t2, prior_rmax=prior_r,
                        chosen_idx=pi[:n], signal_type=st,
                        hands=hands, medians=medians, top2sum=top2sum,
                        scale_pay=cfg["scale_pay"], scale_param=cfg["scale_param"],
                        ace_payout=cfg["ace_payout"], signal_cost=cfg["signal_cost"]
                    )
                    vals[r] = out["net_return_max"]
                means.append(float(np.mean(vals)))
            diffs = np.diff(means)
            if np.any(diffs < -tol):
                raise AssertionError(f"Mean net_return_max not monotone for {st}: means={means}")
            print(f"Monotonicity OK for {st}. Mean series: {[round(x,4) for x in means]}")
        return

    report = flow_single_board(seed=args.seed, n_sig=args.n_sig,
                               ace_payout=20.0, scale_pay=0,
                               scale_param=0.25, signal_cost=0.0)

    print(f"Seed={report['seed']}, n_sig={report['n_sig']}")
    print(f"Chosen piles (prefix of permutation): {report['chosen_idx']}")
    print("Per-pile signals:")
    print("  medians:", report["medians"][:9], "... total", len(report["medians"]))
    print("  top2sum:", report["top2sum"], "... total", len(report["top2sum"]))

    for st in ("median", "top2"):
        r = report["results"][st]
        print(f"\nSignal type: {st}")
        print(f"  Highest expected payoff: {r['net_return_max']:.4f}%")
        print(f"  E[payoff] weighted:     {r['net_return_linear']:.4f}%")
        print(f"  Top-5 E[payoff] wgt:    {r['net_return_top5']:.4f}%")


if __name__ == "__main__":
    main()

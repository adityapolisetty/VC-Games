"""Testing harness for the card game (v7).

Quick usage:
  python code/testing.py --test 1
  python code/testing.py --test 2 --pre_rounds 2000000 --procs 8 --post output/post_mc.testing.npz
  python code/testing.py --test 3 --post_t3 output/post_mc.npz --sig median --nsigs 9 --sp 0 --ap 20
  python code/testing.py --test 4 --t4_rounds 10 --t4_workers 4
  python code/testing.py --test all   # run tests 1→4 sequentially

Tests:
  1) Boards + constraints
     - Prints 3 sample boards (seeded) with per‑pile ranks and signals.
     - Checks global‑deck constraints on 1000 boards.

  2) Precompute + sample
     - Runs precompute_posteriors.py and prints random posterior rows (median/top2/min).

  3) Single‑round diagnostics
     - One deterministic board; prints per‑pile P(Ace|obs), E[payoff], and weights; shows net returns.
     - Supports payoff scaling (sp/sparam), ace payout (ap), signal type and #signals (nsigs).

  4) Seeding checks
     - Verifies per‑round determinism, chunking invariance and within‑round grid consistency.
"""

import argparse
import numpy as np
from numpy.random import default_rng

from card_game import (
    _deal_cards_global_deck,
    NUM_PILES,
    CARDS_PER_PILE,
    ACE_RANK,
    round_seed,
    realize_payout,
    _load_mc_posteriors,
    run_single_round,
)
import sys, importlib
import precompute_posteriors as precomp


# -------------------------------------
# Test 1: print sample boards and check constraints
# -------------------------------------


def _print_board(idx: int, seed: int) -> None:
    rng = default_rng(round_seed(seed, idx))
    has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)

    print(f"Board {idx + 1}")
    for i in range(NUM_PILES):
        pile = " ".join(f"{int(v):2d}" for v in hands[i].tolist())
        print(f"  Pile {i}: {pile}")

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
    print(f"\nTest 1:\n\n")

    # Print 3 example boards
    for i in range(3):
        _print_board(i, seed)

    # Check constraints over 1000 boards
    n = 100000
    v = _check_constraints(seed, n)
    print("Constraint check over", n, "boards (serial)")
    print("Restrictions tested:")
    print(" - Exactly 45 cards total on board")
    print(" - Ace count == 1; King (13) == 2; Queen (12) == 2; all other ranks <= 4")
    print("Violations:", v)



# -------------------------------------
# Test 2: print random posterior rows (precompute tables)
# -------------------------------------
def _test_2(pre_rounds: int, procs: int, seed: int, out_path: str) -> None:

    print("\n\n--------------------------------------------------")
    print(f"\nTest 2 — Precompute posterior probability tables and sample rows\n")
    # Run full precompute to generate NPZ
    save_argv = list(sys.argv)
    sys.argv = [
        "precompute_posteriors.py",
        "--seed", str(seed),
        "--rounds", str(int(pre_rounds)),
        "--procs", str(int(procs)),
        "--out", out_path,
    ]
    importlib.reload(precomp)
    precomp.main()
    sys.argv = save_argv

    # Load NPZ and print random posterior rows (median/top2/min)
    with np.load(out_path, allow_pickle=False) as z:
        pools = []
        if {'rmax_median_keys','rmax_median_mat'}.issubset(z.files):
            pools.append(("median", z['rmax_median_keys'].astype(int), np.asarray(z['rmax_median_mat'], float)))
        if {'rmax_top2_keys','rmax_top2_mat'}.issubset(z.files):
            pools.append(("top2",   z['rmax_top2_keys'].astype(int),   np.asarray(z['rmax_top2_mat'], float)))
        if {'rmax_min_keys','rmax_min_mat'}.issubset(z.files):
            pools.append(("min",    z['rmax_min_keys'].astype(int),    np.asarray(z['rmax_min_mat'], float)))

    flat = []
    for label, keys, mat in pools:
        m = mat if mat.ndim == 2 else mat.reshape(1, -1)
        for i, val in enumerate(keys.tolist()):
            flat.append((label, int(val), m[i]))
    if not flat:
        raise SystemExit("Posterior NPZ has no rows to display.")
    rng = default_rng(seed + 999)
    idxs = rng.choice(len(flat), size=min(5, len(flat)), replace=False)
    ranks = list(range(2, 15))
    print("Labels: median=Median rank; top2=Top-2 rank sum; max=Highest rank; min=Lowest rank")
    for j, sel in enumerate(idxs, 1):
        label, val, row = flat[int(sel)]
        print(f"[{j}] P(Rmax=R | {label}={val})")
        for r in ranks:
            print(f"  R={r}: {row[r-2]:.8f}")


# -------------------------------------
# Test 3: run_single_round diagnostics on one sample board
# -------------------------------------
def _test_3(
    seed: int,
    post_npz: str,
    signal_type: str = "median",
    signal_cost: float = 0.0,
    scale_pay: int = 0,
    scale_param: float = 0.25,
    ace_payout: float = 20.0,
    n_sigs: int = 9,
) -> None:

    # One deterministic board
    r = 0
    _, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(default_rng(round_seed(seed, r)))

    # Load posteriors via the same loader as simulation
    rmax_median, rmax_top2, rmax_max, rmax_min, prior_rmax, _ = _load_mc_posteriors(post_npz)

    # Select signal table and observed values
    table_map = {"median": rmax_median, "top2": rmax_top2, "max": rmax_max, "min": rmax_min}
    obs_map   = {"median": np.asarray(medians, int), "top2": np.asarray(top2sum, int), "max": np.asarray(max_rank, int), "min": np.asarray(min_rank, int)}
    if signal_type not in table_map:
        raise SystemExit(f"Unknown signal_type '{signal_type}'. Use one of median/top2/max/min.")
    post_table = table_map[signal_type]
    obs_vals   = obs_map[signal_type]

    # Expected-payoff score setup (same as run_single_round)
    ranks_all = np.arange(2, 15, dtype=int)
    if scale_pay == 0:
        h_vals = np.array([float(ace_payout) if r == ACE_RANK else 0.0 for r in ranks_all], float)
    else:
        h_vals = np.array([float(ace_payout) * (float(scale_param) ** max(0, ACE_RANK - r)) for r in ranks_all], float)

    # Choose which piles are observed (buy n_sigs signals): prefix 0..n_sigs-1
    n_buy = max(0, min(int(n_sigs), NUM_PILES))
    chosen_idx = np.arange(n_buy, dtype=int)

    scores = np.zeros(NUM_PILES, float)
    for i in range(NUM_PILES):
        if i < n_buy:
            vec = np.asarray(post_table.get(int(obs_vals[i]), prior_rmax), float)
        else:
            vec = np.asarray(prior_rmax, float)
        scores[i] = float(np.dot(h_vals, vec))

    # Weight rules
    def weights_max(sc: np.ndarray) -> np.ndarray:
        p = np.array(sc, float); mx = np.max(p)
        winners = np.where(p == mx)[0]
        w = np.zeros_like(p, float); w[winners] = 1.0 / winners.size
        return w

    def weights_linear(sc: np.ndarray) -> np.ndarray:
        p = np.array(sc, float); s = float(np.sum(p))
        return (p / s) if s > 0 else (np.ones_like(p)/len(p))

    def weights_top5(sc: np.ndarray) -> np.ndarray:
        p = np.array(sc, float)
        idx = np.argsort(p)[-5:]
        w = np.zeros_like(p, float)
        sm = float(np.sum(p[idx]))
        w[idx] = (p[idx] / sm) if sm > 0 else (1.0 / 5.0)
        return w

    w_max  = weights_max(scores)
    w_lin  = weights_linear(scores)
    w_top5 = weights_top5(scores)

    # Call the exact function under test for net returns using the same chosen_idx
    out = run_single_round(
        rmax_median=rmax_median, rmax_top2=rmax_top2, rmax_max=rmax_max, rmax_min=rmax_min, prior_rmax=prior_rmax,
        chosen_idx=chosen_idx, signal_type=signal_type,
        hands=hands, medians=medians, top2sum=top2sum, max_rank=max_rank, min_rank=min_rank,
        scale_pay=scale_pay, scale_param=scale_param,
        ace_payout=ace_payout, signal_cost=signal_cost,
    )

    # Print board and per-pile table
    print("\n\n--------------------------------------------------")
    print("\nTest 3 — One board diagnostics\n\n")
    print(f"Config: seed={seed}, signal={signal_type}, signal_cost={signal_cost}, scale_pay={scale_pay}, scale_param={scale_param}, ace_payout={ace_payout}, n_sigs={n_buy}")
    print("Board (pile --------------> ranks, specials)")
    for i in range(NUM_PILES):
        pile = hands[i].tolist()
        specials = []
        if any(v == 14 for v in pile): specials.append('A')
        if any(v == 13 for v in pile): specials.append('K')
        if any(v == 12 for v in pile): specials.append('Q')
        spec = "".join(specials) if specials else "-"
        print(f"       Pile {i:>2}: {' '.join(f'{int(v):2d}' for v in pile)}   specials={spec}")

    print("\nPer-pile expected payoff and weights (observing first n_sigs piles for this signal)\n")
    
    # Columns differ slightly for sp==1 where non‑Ace ranks have nonzero pay.
    if scale_pay == 1:
        header = (
            f"{'Pile':>4}  {'Obs':>5}  {'P(Ace|obs)':>10}  {'AcePay':>7}  "
            f"{'P(King|obs)':>11}  {'KingPay':>7}  {'P(Queen|obs)':>12}  {'QueenPay':>9}  "
            f"{'P(Jack|obs)':>11}  {'JackPay':>8}  {'AKQJ sum':>10}  "
            f"{'E(Payoff)':>10}  {'w[High]':>8}  {'w[Lin]':>8}  {'w[Top5]':>8}"
        )
    else:
        header = (
            f"{'Pile':>4}  {'Obs':>5}  {'P(Ace|obs)':>10}  {'AcePay':>7}  "
            f"{'E(Payoff)':>10}  {'w[High]':>8}  {'w[Lin]':>8}  {'w[Top5]':>8}"
        )
    print(header)
    print("-" * len(header))

    ace_idx = ACE_RANK - 2
    k_idx = 13 - 2
    q_idx = 12 - 2
    j_idx = 11 - 2
    for i in range(NUM_PILES):
        obs = int(obs_vals[i]) if i < n_buy else None
        vec = np.asarray(post_table.get(int(obs_vals[i]), prior_rmax), float) if i < n_buy else np.asarray(prior_rmax, float)
        p_ace = float(vec[ace_idx])
        if scale_pay == 1:
            p_king = float(vec[k_idx])
            p_queen = float(vec[q_idx])
            p_jack = float(vec[j_idx])
            kpay = float(h_vals[k_idx])
            qpay = float(h_vals[q_idx])
            jpay = float(h_vals[j_idx])
            akqj_sum = float(p_ace * h_vals[ace_idx] + p_king * kpay + p_queen * qpay + p_jack * jpay)
            obs_str = '-' if obs is None else f"{obs:d}"
            row = (
                f"{i:4d}  {obs_str:>5}  {p_ace:10.6f}  {ace_payout:7.2f}  "
                f"{p_king:11.6f}  {kpay:7.2f}  {p_queen:12.6f}  {qpay:9.2f}  "
                f"{p_jack:11.6f}  {jpay:8.2f}  {akqj_sum:10.6f}  "
                f"{scores[i]:10.6f}  {w_max[i]:8.4f}  {w_lin[i]:8.4f}  {w_top5[i]:8.4f}"
            )
        else:
            obs_str = '-' if obs is None else f"{obs:d}"
            row = (
                f"{i:4d}  {obs_str:>5}  {p_ace:10.6f}  {ace_payout:7.2f}  {scores[i]:10.6f}  "
                f"{w_max[i]:8.4f}  {w_lin[i]:8.4f}  {w_top5[i]:8.4f}"
            )
        print(row)

    print("\nNet returns (%) from run_single_round:")
    print(
        f"  Highest: {out['net_return_max']:.6f}  |  "
        f"Top-5 E[payoff]: {out['net_return_top5']:.6f}  |  "
        f"E[payoff] weighted: {out['net_return_linear']:.6f}"

    )


# -------------------------------------
# Test 4: RNG seeding determinism and grid consistency
# -------------------------------------
def _test_4(seed: int, t4_rounds: int = 10, t4_workers: int = 4) -> None:
    R = int(max(1, t4_rounds))
    W = int(max(1, t4_workers))

    print("\n\n--------------------------------------------------")
    print("\nTest 4 — RNG seeding determinism checks")
    print(f"Config: seed={seed}, rounds={R}, workers={W}")

    # Part A: per‑round determinism (same seed -> identical board and permutation)
    ok_A = True
    for r in range(min(R, 12)):
        rng1 = default_rng(round_seed(seed, r))
        _, h1, *_ = _deal_cards_global_deck(rng1)
        pi1 = rng1.permutation(NUM_PILES)

        rng2 = default_rng(round_seed(seed, r))
        _, h2, *_ = _deal_cards_global_deck(rng2)
        pi2 = rng2.permutation(NUM_PILES)

        same_hands = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(h1, h2))
        same_perm  = np.array_equal(pi1, pi2)
        if not (same_hands and same_perm):
            ok_A = False
            print(f"  MISMATCH at r={r}: hands_equal={same_hands}, perm_equal={same_perm}")
            break
    print("\nPart A (per‑round determinism):", "OK" if ok_A else "FAILED")
    print("Rationale: Using round_seed(seed, r) multiple times for the same round must reproduce the same board and pile permutation but different ones across rounds. Important for parallel correctness.")

    # Print sample boards and permutations for first few rounds
    print("\nPart A — sample boards and permutations (first 3 rounds):")
    for r in range(min(R, 3)):
        rngs = default_rng(round_seed(seed, r))
        _, hands_s, *_ = _deal_cards_global_deck(rngs)
        pi_s = rngs.permutation(NUM_PILES)
        print(f"  r={r}: perm = {pi_s.tolist()}")
        for i in range(NUM_PILES):
            pile = " ".join(f"{int(v):2d}" for v in hands_s[i].tolist())
            print(f"       Pile {i:>2}: {pile}")

    # Part B: chunking invariance (parallel-style chunking vs serial)
    serial_perm_first = []
    for r in range(R):
        rng = default_rng(round_seed(seed, r))
        _ = _deal_cards_global_deck(rng)
        pi = rng.permutation(NUM_PILES)
        serial_perm_first.append(int(pi[0]))

    # Emulate chunking logic from simulate_experiment
    base = R // W; rem = R % W
    chunk_sizes = [base + (1 if i < rem else 0) for i in range(W)]
    starts = []
    s = 0
    for c in chunk_sizes:
        if c > 0:
            starts.append((s, c))
            s += c
    parallel_perm_first = [None] * R
    for start, sz in starts:
        for i in range(sz):
            r = int(start) + int(i)
            rng = default_rng(round_seed(seed, r))
            _ = _deal_cards_global_deck(rng)
            pi = rng.permutation(NUM_PILES)
            parallel_perm_first[r] = int(pi[0])
    ok_B = np.array_equal(np.asarray(serial_perm_first), np.asarray(parallel_perm_first))
    print("\nPart B (chunking invariance):", "OK" if ok_B else "FAILED")
    print("Rationale: Each round uses only r-based seeds, so splitting rounds into chunks/workers should not change the per-round permutations or results.")
    if not ok_B:
        diffs = [i for i,(a,b) in enumerate(zip(serial_perm_first, parallel_perm_first)) if a!=b]
        print("  First mismatches at indices:", diffs[:10])
    # Print actual sequences (first-pile index per round) and sample full perms
    print("\nPart B — serial vs parallel first-pile index per round:")
    print("  serial  :", serial_perm_first)
    print("  parallel:", parallel_perm_first)
    print("\nPart B — sample full permutations (first up to 3 rounds):")
    for r in range(min(R, 3)):
        rng_ser = default_rng(round_seed(seed, r))
        _ = _deal_cards_global_deck(rng_ser)
        pi_ser = rng_ser.permutation(NUM_PILES)
        rng_par = default_rng(round_seed(seed, r))
        _ = _deal_cards_global_deck(rng_par)
        pi_par = rng_par.permutation(NUM_PILES)
        print(f"  r={r}: serial={pi_ser.tolist()}  |  parallel={pi_par.tolist()}")

    # Part C: within-round grid consistency (same board/permutation across st, n_sig)
    r_probe = 0 if R == 1 else min(7, R-1)
    rng_base = default_rng(round_seed(seed, r_probe))
    _, hands_ref, *_ = _deal_cards_global_deck(rng_base)
    perm_ref = rng_base.permutation(NUM_PILES)
    sts = ["median", "top2", "max", "min"]
    ns_list = [0, max(1, NUM_PILES//2), NUM_PILES-1]
    ok_C = True
    for st in sts:
        for nn in ns_list:
            rngT = default_rng(round_seed(seed, r_probe))
            _, hands_T, *_ = _deal_cards_global_deck(rngT)
            perm_T = rngT.permutation(NUM_PILES)
            if (not all(np.array_equal(np.asarray(a), np.asarray(b)) for a,b in zip(hands_ref, hands_T))) or (not np.array_equal(perm_ref, perm_T)):
                ok_C = False
                print(f"  MISMATCH for st={st}, n_sig={nn} at r={r_probe}")
                break
        if not ok_C:
            break
    print("\nPart C (within‑round grid consistency):", "OK" if ok_C else "FAILED")
    print("Rationale: Within a fixed round, we re-seed before each (signal_type, n_sigs, param combo); the board/permutation should remain identical and only the observed set changes.")

    # Param-grid demo (shared-base): identical perms across combos
    SC_SET = [0.0, 5.0]; SP_SET = [0, 1]; S_SET = [0.25]; AP_SET = [20.0, 30.0]
    combos_demo = [dict(signal_cost=sc_, scale_pay=sp_, scale_param=s_, ace_payout=ap_)
                    for sc_ in SC_SET for sp_ in SP_SET for s_ in S_SET for ap_ in AP_SET]
    shared_perms = []
    for raw in combos_demo:
        rng_sh = default_rng(round_seed(seed, r_probe))
        _ = _deal_cards_global_deck(rng_sh)
        shared_perms.append(rng_sh.permutation(NUM_PILES))
    same_shared = all(np.array_equal(shared_perms[0], p) for p in shared_perms[1:])
    print(f"  Shared-base seeding: perms identical across {len(combos_demo)} combos? {same_shared}")

    # Part D: sample outputs to eyeball
    print("\nPart D - Sample (r=0) permutation and chosen_idx examples:")
    print("  Rationale: Order of piles signalled in the same simulation round should be the same. Crucial for comparability.")
    rngs = default_rng(round_seed(seed, 0))
    _ = _deal_cards_global_deck(rngs)
    pi0 = rngs.permutation(NUM_PILES)
    print("  perm[0..8] =", pi0.tolist())
    for nn in [0, 1, 2, 3, 4, min(5, NUM_PILES), 6, 7, 8, NUM_PILES]:
        print(f"  n_sig={nn}: chosen_idx =", pi0[:nn].tolist())
    print("Part D: OK")



# -------------------------------------
# Runner
# -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Minimal testing harness")
    ap.add_argument("--test", type=str, required=True, help="Test: 1=boards+constraints, 2=precompute+sample, 3=single-round diag, 4=seeding checks, all=run 1-4")
    ap.add_argument("--pre_rounds", type=int, default=10000000, help="Rounds for precompute tables in test 2")
    ap.add_argument("--procs", type=int, default=8, help="Workers for precompute in test 2")
    ap.add_argument("--post", type=str, default="output/post_mc.testing.npz", help="Output NPZ path for test 2")
    ap.add_argument("--post_t3", type=str, default="output/post_mc.npz", help="Input for test 3")
    ap.add_argument("--seed", type=int, default=12345, help="Base seed for tests (default: 12345)")

    # Optional knobs for Test 3
    ap.add_argument("--sig", type=str, default="median", help="Signal type for test 3 (median/top2/max/min)")
    ap.add_argument("--sig_cost", type=float, default=0.0, help="Signal cost for test 3")
    ap.add_argument("--nsigs", type=int, default=9, help="Number of signals to buy (0..9) for test 3")
    ap.add_argument("--sp", type=int, default=0, help="scale_pay for test 3 (0 or 1)")
    ap.add_argument("--sparam", type=float, default=0.25, help="scale_param for test 3")
    ap.add_argument("--ap", type=float, default=20.0, help="Ace payout multiplier for test 3")

    # Optional knobs for Test 4
    ap.add_argument("--t4_rounds", type=int, default=10, help="Rounds to probe for seeding determinism (test 4)")
    ap.add_argument("--t4_workers", type=int, default=4, help="Worker count to emulate chunking (test 4)")
    args = ap.parse_args()

    seed = int(args.seed)

    # Allow --test all and string values
    test_sel = str(args.test).strip().lower()
    if test_sel == 'all':
        _test_1(seed)
        _test_2(int(args.pre_rounds), int(args.procs), seed, args.post)
        _test_3(
            seed=seed,
            post_npz=args.post_t3,
            signal_type=args.sig,
            signal_cost=float(args.sig_cost),
            n_sigs=int(args.nsigs),
            scale_pay=int(args.sp),
            scale_param=float(args.sparam),
            ace_payout=float(args.ap),
        )
        _test_4(seed, int(args.t4_rounds), int(args.t4_workers))
        return
    else:
        try:
            args.test = int(test_sel)
        except ValueError:
            raise SystemExit("Unknown test. Use --test 1, 2, 3, 4 or 'all'.")

    if args.test == 1:
        _test_1(seed)
        
    elif args.test == 2:
        _test_2(int(args.pre_rounds), int(args.procs), seed, args.post)
        
    elif args.test == 3:
        _test_3(
            seed=seed,
            post_npz=args.post_t3,
            signal_type=args.sig,
            signal_cost=float(args.sig_cost),
            n_sigs=int(args.nsigs),
            scale_pay=int(args.sp),
            scale_param=float(args.sparam),
            ace_payout=float(args.ap),
        )
    
    elif args.test == 4:
        _test_4(seed, int(args.t4_rounds), int(args.t4_workers))
        return

    else:
        raise SystemExit("Unknown test. Use --test 1, 2, 3, 4 or 'all'.")


if __name__ == "__main__":
    main()

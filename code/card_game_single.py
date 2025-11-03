#!/usr/bin/env python3
# card_game_single.py
# Simplified single-run version with command-line parameters
# No sweep infrastructure - just one simulation per run

import numpy as np, argparse, pathlib, os, sys, tempfile, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.random import SeedSequence, default_rng
from time import perf_counter
from math import comb

# -----------------------
# Parameters / Defaults
# -----------------------
NUM_PILES       = 9
CARDS_PER_PILE  = 5
RANKS           = np.arange(2, 15)          # 2..14, where 14 is Ace
ACE_RANK        = 14
COPIES_PER_RANK = 4                         # one standard deck
BUDGET          = 100.0
ACE_SUCCESS_PROB= 1.0

# Prior: expected # of Ace piles on board = 2 (uniform K in {0..4} => E[K]=2)
P0_SINGLE_PILE  = 2.0 / NUM_PILES

# Canonical fill-ins when a switch is OFF
CANON_SCALE_PARAM_DEFAULT = 0.25   # used when scale_pay == 0

# -----------------------
# Helpers: formatting + IDs
# -----------------------
def _fmt_float_id(x: float, nd=4) -> str:
    s = f"{x:.{nd}f}"
    return s.replace(".", "p")

def canonicalize_params(raw):
    """Convert raw parameters to canonical form with stable ID."""
    sc  = float(raw["signal_cost"])
    sp  = int(raw["scale_pay"])
    s   = float(raw["scale_param"])
    ap  = float(raw["ace_payout"])

    s_norm  = s  if sp == 1 else CANON_SCALE_PARAM_DEFAULT

    key = (("sc", sc), ("sp", sp), ("s", s_norm), ("ap", ap))
    parts = []
    for k, v in key:
        parts.append(f"{k}{_fmt_float_id(v) if isinstance(v,float) else v}")
    key_id = "_".join(parts)
    return (
        {"signal_cost": sc, "scale_pay": sp, "scale_param": s_norm, "ace_payout": ap},
        key,
        key_id
    )

# -----------------------
# Exact posterior helpers
# -----------------------
def _all_multicounts(total, n_ranks, cap):
    c = [0]*n_ranks
    def rec(i, remaining):
        if i == n_ranks-1:
            if 0 <= remaining <= cap:
                c[i] = remaining
                yield tuple(c)
            return
        max_take = min(cap, remaining)
        for t in range(max_take+1):
            c[i] = t
            yield from rec(i+1, remaining - t)
    yield from rec(0, total)

def _median_of_multiset(counts, ranks):
    target = 3
    cum = 0
    for r, cnt in zip(ranks, counts):
        cum += cnt
        if cum >= target:
            return int(r)
    return int(ranks[-1])

def _top2sum_of_multiset(counts, ranks):
    remaining = 2
    s = 0
    for r, cnt in zip(ranks[::-1], counts[::-1]):
        take = min(cnt, remaining)
        if take > 0:
            s += int(r) * take
            remaining -= take
            if remaining == 0:
                break
    return int(s)

def _posterior_exact():
    """Compute exact Bayesian posteriors for all three signal types."""
    ranks_na = np.arange(2,14, dtype=int)

    like_na_m = {}; like_na_s = {}; like_na_t = {}; w_na_tot = 0
    for cnt in _all_multicounts(total=5, n_ranks=len(ranks_na), cap=4):
        w = 1
        for c in cnt: w *= comb(4, c)
        if w == 0: continue
        med = _median_of_multiset(cnt, ranks_na)
        sos = int(np.dot(np.array(cnt, int), (ranks_na**2)))
        t2  = _top2sum_of_multiset(cnt, ranks_na)
        like_na_m[med] = like_na_m.get(med, 0) + w
        like_na_s[sos] = like_na_s.get(sos, 0) + w
        like_na_t[t2 ] = like_na_t.get(t2 , 0) + w
        w_na_tot += w

    like_a_m = {}; like_a_s = {}; like_a_t = {}; w_a_tot = 0
    for cnt in _all_multicounts(total=4, n_ranks=len(ranks_na), cap=4):
        w = 1
        for c in cnt: w *= comb(4, c)
        if w == 0: continue
        cnt_all   = np.concatenate([np.array(cnt, int), np.array([1], int)])
        ranks_all = np.concatenate([ranks_na, np.array([ACE_RANK], int)])
        med = _median_of_multiset(cnt_all, ranks_all)
        sos = int(np.dot(np.array(cnt, int), (ranks_na**2)) + ACE_RANK*ACE_RANK)
        t2  = _top2sum_of_multiset(cnt_all, ranks_all)
        like_a_m[med] = like_a_m.get(med, 0) + w
        like_a_s[sos] = like_a_s.get(sos, 0) + w
        like_a_t[t2 ] = like_a_t.get(t2 , 0) + w
        w_a_tot += w

    for d in (like_na_m, like_na_s, like_na_t):
        for k in list(d.keys()): d[k] = d[k] / w_na_tot
    for d in (like_a_m, like_a_s, like_a_t):
        for k in list(d.keys()): d[k] = d[k] / w_a_tot

    def to_post(like_na, like_a):
        post = {}
        all_x = set(like_na.keys()) | set(like_a.keys())
        for x in all_x:
            pa = like_a.get(x, 0.0); pn = like_na.get(x, 0.0)
            num = P0_SINGLE_PILE * pa
            den = P0_SINGLE_PILE * pa + (1.0 - P0_SINGLE_PILE) * pn
            post[x] = (num / den) if den > 0 else P0_SINGLE_PILE
        return post

    return (to_post(like_na_m, like_a_m),
            to_post(like_na_s, like_a_s),
            to_post(like_na_t, like_a_t))

def build_posteriors():
    return _posterior_exact()

# -----------------------
# Global-deck board deal (single-pass, no restarts)
# -----------------------
def _deal_cards_global_deck(rng):
    """Deal cards from a global deck to all piles."""
    has_ace = np.zeros(NUM_PILES, dtype=bool)
    K = int(rng.integers(0, 5))
    if K > 0:
        ace_piles = rng.choice(np.arange(NUM_PILES), size=K, replace=False)
        has_ace[ace_piles] = True

    pool = np.repeat(np.arange(2, 14, dtype=int), COPIES_PER_RANK)

    hands = [[] for _ in range(NUM_PILES)]
    for i in range(NUM_PILES):
        if has_ace[i]:
            hands[i].append(ACE_RANK)
        need = CARDS_PER_PILE - len(hands[i])

        # Validation: ensure enough cards in pool
        if need > pool.shape[0]:
            raise ValueError(f"Deck exhausted: need {need} cards but only {pool.shape[0]} remaining")

        idx = rng.choice(pool.shape[0], size=need, replace=False)
        draw = pool[idx]
        hands[i].extend(draw.tolist())
        pool = np.delete(pool, idx)

    medians   = np.empty(NUM_PILES, dtype=int)
    sumsq     = np.empty(NUM_PILES, dtype=int)
    top2sum   = np.empty(NUM_PILES, dtype=int)
    max_rank  = np.empty(NUM_PILES, dtype=int)

    for i in range(NUM_PILES):
        arr = np.array(sorted(hands[i]), dtype=int)
        medians[i]   = int(arr[CARDS_PER_PILE//2])
        sumsq[i]     = int(np.sum(arr.astype(int)**2))
        top2sum[i]   = int(arr[-1] + arr[-2])
        max_rank[i]  = int(arr[-1])

    return has_ace, [np.array(sorted(h), int) for h in hands], medians, sumsq, top2sum, max_rank

# -----------------------
# Posterior updates
# -----------------------
def update_posteriors_median(p_prior, medians, post_lookup, chosen_idx):
    p = np.array(p_prior, float)
    for i in chosen_idx:
        p[i] = float(post_lookup[int(medians[i])])
    s = float(np.sum(p))
    if s > 0:
        p = np.minimum(p, 1.0)  # cap individual probabilities
        if s > 4.0:
            p *= (4.0 / s)
    return p

def update_posteriors_sumsq(p_prior, sumsq, post_lookup, chosen_idx):
    p = np.array(p_prior, float)
    for i in chosen_idx:
        p[i] = float(post_lookup[int(sumsq[i])])
    s = float(np.sum(p))
    if s > 0:
        p = np.minimum(p, 1.0)
        if s > 4.0:
            p *= (4.0 / s)
    return p

def update_posteriors_top2(p_prior, top2sum, post_lookup, chosen_idx):
    p = np.array(p_prior, float)
    for i in chosen_idx:
        p[i] = float(post_lookup[int(top2sum[i])])
    s = float(np.sum(p))
    if s > 0:
        p = np.minimum(p, 1.0)
        if s > 4.0:
            p *= (4.0 / s)
    return p

# -----------------------
# Realized payout
# -----------------------
def realize_payout(rng, has_ace, w, max_rank, scale_pay, scale_param, ace_payout,
                   investable_budget):
    """
    Compute realized payout given portfolio weights.

    Invest only the remaining budget after paying for signals.
    - investable_budget = max(0, BUDGET - info_cost)
    - gross is computed on that remaining stake
    - gross_pct is expressed relative to the original BUDGET for comparability
    """
    has_ace = np.array(has_ace, int)
    w = np.array(w, float)
    gross = 0.0

    for i in range(NUM_PILES):
        # payoff per £ invested
        if scale_pay == 0:
            if has_ace[i] == 1:
                hit = (rng.random() < ACE_SUCCESS_PROB)
                payoff_per_pound = ace_payout if hit else 0.0
            else:
                payoff_per_pound = 0.0
        else:
            steps_down = max(0, ACE_RANK - int(max_rank[i]))
            scale = (float(scale_param) ** steps_down)
            hit = (rng.random() < ACE_SUCCESS_PROB)
            payoff_per_pound = (ace_payout * scale) if hit else 0.0

        # invest only the remaining budget
        gross += investable_budget * w[i] * payoff_per_pound

    # express gross as % of original BUDGET for consistent scaling
    gross_pct = 100.0 * (gross / BUDGET) if BUDGET > 0 else 0.0
    net_pct = 100.0 * ((gross - BUDGET) / BUDGET) if BUDGET > 0 else 0.0
    return gross_pct, net_pct

# -----------------------
# One round
# -----------------------
def run_single_round(rng, posters, n_signals, signal_type,
                     has_ace, medians, sumsq, top2sum, max_rank,
                     scale_pay, scale_param, ace_payout, signal_cost):
    """Execute one round of the simulation."""
    p_prior = np.full(NUM_PILES, P0_SINGLE_PILE, float)
    chosen = np.array([], int) if n_signals == 0 else np.array(
        rng.choice(np.arange(NUM_PILES), size=n_signals, replace=False), int
    )

    if signal_type == "median":
        p_post = update_posteriors_median(p_prior, medians, posters["median"], chosen)
    elif signal_type == "sumsq":
        p_post = update_posteriors_sumsq(p_prior, sumsq, posters["sumsq"], chosen)
    elif signal_type == "top2":
        p_post = update_posteriors_top2(p_prior, top2sum, posters["top2"], chosen)
    else:
        raise ValueError("signal_type must be median|sumsq|top2")

    def weights_max(p):
        p = np.array(p, float); mx = np.max(p)
        winners = np.where(p == mx)[0]
        w = np.zeros_like(p, float); w[winners] = 1.0 / len(winners)
        return w
    def weights_linear(p):
        p = np.array(p, float); s = float(np.sum(p))
        return (p / s) if s > 0 else np.ones_like(p)/len(p)
    def weights_square(p):
        p = np.array(p, float); q = p*p; s = float(np.sum(q))
        return (q / s) if s > 0 else np.ones_like(p)/len(p)

    w_max = weights_max(p_post)
    w_lin = weights_linear(p_post)
    w_sq  = weights_square(p_post)

    # pay for information first; invest the remainder only
    info_cost = n_signals * float(signal_cost)
    investable_budget = max(0.0, BUDGET - info_cost)

    gp_max, n_max = realize_payout(
        rng, has_ace, w_max, max_rank,
        scale_pay, scale_param, ace_payout,
        investable_budget
    )
    gp_lin, n_lin = realize_payout(
        rng, has_ace, w_lin, max_rank,
        scale_pay, scale_param, ace_payout,
        investable_budget
    )
    gp_sq, n_sq = realize_payout(
        rng, has_ace, w_sq, max_rank,
        scale_pay, scale_param, ace_payout,
        investable_budget
    )

    return dict(
        gross_pct_max=float(gp_max), net_pct_max=float(n_max),
        gross_pct_linear=float(gp_lin), net_pct_linear=float(n_lin),
        gross_pct_square=float(gp_sq), net_pct_square=float(n_sq)
    )

# -----------------------
# Worker
# -----------------------
def _worker_chunk(seed_int, signal_type, n_sig, rounds_chunk, posters, params):
    """Worker function for parallel processing of simulation rounds."""
    rng = default_rng(int(seed_int))
    gp_max = np.empty(rounds_chunk, float)
    gp_lin = np.empty(rounds_chunk, float)
    gp_sq  = np.empty(rounds_chunk, float)
    n_max = np.empty(rounds_chunk, float)
    n_lin = np.empty(rounds_chunk, float)
    n_sq  = np.empty(rounds_chunk, float)
    for i in range(rounds_chunk):
        has_ace, hands, medians, sumsq, top2sum, max_rank = _deal_cards_global_deck(rng)
        out = run_single_round(
            rng=rng, posters=posters, n_signals=n_sig, signal_type=signal_type,
            has_ace=has_ace, medians=medians, sumsq=sumsq, top2sum=top2sum,
            max_rank=max_rank,
            scale_pay=params["scale_pay"], scale_param=params["scale_param"],
            ace_payout=params["ace_payout"], signal_cost=params["signal_cost"],
        )
        gp_max[i] = out["gross_pct_max"]; gp_lin[i] = out["gross_pct_linear"]; gp_sq[i] = out["gross_pct_square"]
        n_max[i] = out["net_pct_max"]; n_lin[i] = out["net_pct_linear"]; n_sq[i] = out["net_pct_square"]
    return gp_max, gp_lin, gp_sq, n_max, n_lin, n_sq

# -----------------------
# Main simulate (single configuration)
# -----------------------
def simulate_experiment(seed_int, rounds, max_signals, procs, params):
    """Run full simulation for one parameter configuration."""
    root_ss = SeedSequence(int(seed_int))

    post_median, post_sumsq, post_top2 = build_posteriors()
    posters = {"median": post_median, "sumsq": post_sumsq, "top2": post_top2}

    def _pack_post(d):
        keys = sorted(d.keys())
        xs = np.array(keys, dtype=float)
        ys = np.array([float(d[k]) for k in keys], dtype=float)
        return xs, ys
    pm_x, pm_y   = _pack_post(post_median)
    pss_x, pss_y = _pack_post(post_sumsq)
    pt2_x, pt2_y = _pack_post(post_top2)

    signal_types = ["median", "sumsq", "top2"]
    sigtype_seeds = {st: root_ss.spawn(max_signals+1) for st in signal_types}

    dist    = {st: {} for st in signal_types}
    summary = {st: {} for st in signal_types}

    total_units = len(signal_types) * (max_signals + 1) * int(rounds)
    processed = 0
    step_overall = max(1, total_units // 100)

    def _print_bar(cur, total):
        cur = int(cur); total = max(1, int(total))
        pct = int(100 * cur / total)
        barw = 30; filled = int(barw * cur / total)
        bar = "=" * filled + "-" * (barw - filled)
        sys.stdout.write(f"\rProgress {cur}/{total} {pct:3d}% [{bar}]"); sys.stdout.flush()

    for st in signal_types:
        for n_sig in range(max_signals+1):
            if procs and int(procs) > 1 and rounds > 1:
                W = int(procs); base = rounds // W; rem  = rounds % W
                chunks = [base + (1 if i < rem else 0) for i in range(W)]
                chunks = [c for c in chunks if c > 0]
                chunk_seeds = [int(s.entropy) for s in sigtype_seeds[st][n_sig].spawn(len(chunks))]
                futures = []
                with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
                    for i, n_i in enumerate(chunks):
                        futures.append(ex.submit(_worker_chunk, chunk_seeds[i], st, n_sig, n_i, posters, params))
                    parts_gp_max, parts_gp_lin, parts_gp_sq = [], [], []
                    parts_n_max, parts_n_lin, parts_n_sq = [], [], []
                    for fut in as_completed(futures):
                        gp_m, gp_l, gp_s, n_m, n_l, n_s = fut.result()
                        parts_gp_max.append(gp_m); parts_gp_lin.append(gp_l); parts_gp_sq.append(gp_s)
                        parts_n_max.append(n_m); parts_n_lin.append(n_l); parts_n_sq.append(n_s)
                        processed += int(len(gp_m))
                        if (processed % step_overall == 0) or (processed >= total_units):
                            _print_bar(processed, total_units)
                gp_max_arr = np.concatenate(parts_gp_max); gp_lin_arr = np.concatenate(parts_gp_lin); gp_sq_arr = np.concatenate(parts_gp_sq)
                n_max_arr = np.concatenate(parts_n_max); n_lin_arr = np.concatenate(parts_n_lin); n_sq_arr = np.concatenate(parts_n_sq)
            else:
                rng_rounds = default_rng(int(sigtype_seeds[st][n_sig].entropy))
                gp_max = np.empty(rounds, float)
                gp_lin = np.empty(rounds, float)
                gp_sq  = np.empty(rounds, float)
                n_max = np.empty(rounds, float)
                n_lin = np.empty(rounds, float)
                n_sq  = np.empty(rounds, float)
                for r in range(rounds):
                    has_ace, hands, medians, sumsq, top2sum, max_rank = _deal_cards_global_deck(rng_rounds)
                    out = run_single_round(
                        rng=rng_rounds, posters=posters, n_signals=n_sig, signal_type=st,
                        has_ace=has_ace, medians=medians, sumsq=sumsq, top2sum=top2sum,
                        max_rank=max_rank,
                        scale_pay=params["scale_pay"], scale_param=params["scale_param"],
                        ace_payout=params["ace_payout"], signal_cost=params["signal_cost"],
                    )
                    gp_max[r] = out["gross_pct_max"]; gp_lin[r] = out["gross_pct_linear"]; gp_sq[r] = out["gross_pct_square"]
                    n_max[r] = out["net_pct_max"]; n_lin[r] = out["net_pct_linear"]; n_sq[r] = out["net_pct_square"]
                    processed += 1
                    if (processed % step_overall == 0) or (processed >= total_units):
                        _print_bar(processed, total_units)
                gp_max_arr = gp_max; gp_lin_arr = gp_lin; gp_sq_arr = gp_sq
                n_max_arr = n_max; n_lin_arr = n_lin; n_sq_arr = n_sq

            dist[st][n_sig] = dict(
                gross_pct_max=gp_max_arr, gross_pct_linear=gp_lin_arr, gross_pct_square=gp_sq_arr,
                net_pct_max=n_max_arr, net_pct_linear=n_lin_arr, net_pct_square=n_sq_arr
            )

            # Compute summary statistics with safety for small samples
            n_samples = len(gp_max_arr)
            summary[st][n_sig] = dict(
                mean_gross_pct_max=float(np.mean(gp_max_arr)),
                mean_gross_pct_linear=float(np.mean(gp_lin_arr)),
                mean_gross_pct_square=float(np.mean(gp_sq_arr)),
                sd_gross_pct_max=float(np.std(gp_max_arr, ddof=1)) if n_samples > 1 else 0.0,
                sd_gross_pct_linear=float(np.std(gp_lin_arr, ddof=1)) if n_samples > 1 else 0.0,
                sd_gross_pct_square=float(np.std(gp_sq_arr, ddof=1)) if n_samples > 1 else 0.0,
                med_gross_pct_max=float(np.median(gp_max_arr)),
                med_gross_pct_linear=float(np.median(gp_lin_arr)),
                med_gross_pct_square=float(np.median(gp_sq_arr)),
                mean_net_pct_max=float(np.mean(n_max_arr)),
                mean_net_pct_linear=float(np.mean(n_lin_arr)),
                mean_net_pct_square=float(np.mean(n_sq_arr)),
                sd_net_pct_max=float(np.std(n_max_arr, ddof=1)) if n_samples > 1 else 0.0,
                sd_net_pct_linear=float(np.std(n_lin_arr, ddof=1)) if n_samples > 1 else 0.0,
                sd_net_pct_square=float(np.std(n_sq_arr, ddof=1)) if n_samples > 1 else 0.0,
                med_net_pct_max=float(np.median(n_max_arr)),
                med_net_pct_linear=float(np.median(n_lin_arr)),
                med_net_pct_square=float(np.median(n_sq_arr))
            )
    try: print()
    except Exception: pass

    meta = dict(mode="dup", params=dict(params),
                post_median_x=pm_x, post_median_y=pm_y,
                post_sumsq_x=pss_x, post_sumsq_y=pss_y,
                post_top2_x=pt2_x, post_top2_y=pt2_y)
    return dist, summary, meta

# -----------------------
# Save results
# -----------------------
def save_npz(out_path: pathlib.Path, rounds, max_signals, dist, summary, meta, norm_params, raw_params, key_tuple, key_id):
    """Save simulation results to compressed NPZ file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sig_grid = np.arange(max_signals + 1, dtype=int)

    def stack(key):
        return (np.vstack([dist[key][s]["gross_pct_max"]    for s in sig_grid]),
                np.vstack([dist[key][s]["gross_pct_linear"] for s in sig_grid]),
                np.vstack([dist[key][s]["gross_pct_square"] for s in sig_grid]),
                np.vstack([dist[key][s]["net_pct_max"]    for s in sig_grid]),
                np.vstack([dist[key][s]["net_pct_linear"] for s in sig_grid]),
                np.vstack([dist[key][s]["net_pct_square"] for s in sig_grid]))

    grosspct_median_max_mat, grosspct_median_linear_mat, grosspct_median_square_mat, netpct_median_max_mat, netpct_median_linear_mat, netpct_median_square_mat = stack("median")
    grosspct_sumsq_max_mat, grosspct_sumsq_linear_mat, grosspct_sumsq_square_mat, netpct_sumsq_max_mat, netpct_sumsq_linear_mat, netpct_sumsq_square_mat = stack("sumsq")
    grosspct_top2_max_mat, grosspct_top2_linear_mat, grosspct_top2_square_mat, netpct_top2_max_mat, netpct_top2_linear_mat, netpct_top2_square_mat = stack("top2")

    def pack_summary(key, stat_key):
        return np.array([summary[key][s][stat_key] for s in sig_grid], float)

    mean_grosspct_median_max    = pack_summary("median","mean_gross_pct_max")
    mean_grosspct_median_linear = pack_summary("median","mean_gross_pct_linear")
    mean_grosspct_median_square = pack_summary("median","mean_gross_pct_square")
    sd_grosspct_median_max      = pack_summary("median","sd_gross_pct_max")
    sd_grosspct_median_linear   = pack_summary("median","sd_gross_pct_linear")
    sd_grosspct_median_square   = pack_summary("median","sd_gross_pct_square")

    mean_netpct_median_max    = pack_summary("median","mean_net_pct_max")
    mean_netpct_median_linear = pack_summary("median","mean_net_pct_linear")
    mean_netpct_median_square = pack_summary("median","mean_net_pct_square")
    sd_netpct_median_max      = pack_summary("median","sd_net_pct_max")
    sd_netpct_median_linear   = pack_summary("median","sd_net_pct_linear")
    sd_netpct_median_square   = pack_summary("median","sd_net_pct_square")

    mean_grosspct_sumsq_max     = pack_summary("sumsq","mean_gross_pct_max")
    mean_grosspct_sumsq_linear  = pack_summary("sumsq","mean_gross_pct_linear")
    mean_grosspct_sumsq_square  = pack_summary("sumsq","mean_gross_pct_square")
    sd_grosspct_sumsq_max       = pack_summary("sumsq","sd_gross_pct_max")
    sd_grosspct_sumsq_linear    = pack_summary("sumsq","sd_gross_pct_linear")
    sd_grosspct_sumsq_square    = pack_summary("sumsq","sd_gross_pct_square")

    mean_netpct_sumsq_max     = pack_summary("sumsq","mean_net_pct_max")
    mean_netpct_sumsq_linear  = pack_summary("sumsq","mean_net_pct_linear")
    mean_netpct_sumsq_square  = pack_summary("sumsq","mean_net_pct_square")
    sd_netpct_sumsq_max       = pack_summary("sumsq","sd_net_pct_max")
    sd_netpct_sumsq_linear    = pack_summary("sumsq","sd_net_pct_linear")
    sd_netpct_sumsq_square    = pack_summary("sumsq","sd_net_pct_square")

    mean_grosspct_top2_max      = pack_summary("top2","mean_gross_pct_max")
    mean_grosspct_top2_linear   = pack_summary("top2","mean_gross_pct_linear")
    mean_grosspct_top2_square   = pack_summary("top2","mean_gross_pct_square")
    sd_grosspct_top2_max        = pack_summary("top2","sd_gross_pct_max")
    sd_grosspct_top2_linear     = pack_summary("top2","sd_gross_pct_linear")
    sd_grosspct_top2_square     = pack_summary("top2","sd_gross_pct_square")

    mean_netpct_top2_max      = pack_summary("top2","mean_net_pct_max")
    mean_netpct_top2_linear   = pack_summary("top2","mean_net_pct_linear")
    mean_netpct_top2_square   = pack_summary("top2","mean_net_pct_square")
    sd_netpct_top2_max        = pack_summary("top2","sd_net_pct_max")
    sd_netpct_top2_linear     = pack_summary("top2","sd_net_pct_linear")
    sd_netpct_top2_square     = pack_summary("top2","sd_net_pct_square")

    fd, tmp_path = tempfile.mkstemp(prefix=out_path.stem + '.', suffix='.npz', dir=str(out_path.parent))
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_path,
            mode="dup", id=key_id, key_tuple=np.array(key_tuple, dtype=object),
            params_raw=json.dumps(raw_params), params_norm=json.dumps(norm_params),
            rounds=int(rounds), budget=float(BUDGET), p0_single_pile=float(P0_SINGLE_PILE),
            sig_grid=sig_grid,
            grosspct_median_max_mat=grosspct_median_max_mat, grosspct_median_linear_mat=grosspct_median_linear_mat, grosspct_median_square_mat=grosspct_median_square_mat,
            grosspct_sumsq_max_mat=grosspct_sumsq_max_mat, grosspct_sumsq_linear_mat=grosspct_sumsq_linear_mat, grosspct_sumsq_square_mat=grosspct_sumsq_square_mat,
            grosspct_top2_max_mat=grosspct_top2_max_mat, grosspct_top2_linear_mat=grosspct_top2_linear_mat, grosspct_top2_square_mat=grosspct_top2_square_mat,
            netpct_median_max_mat=netpct_median_max_mat, netpct_median_linear_mat=netpct_median_linear_mat, netpct_median_square_mat=netpct_median_square_mat,
            netpct_sumsq_max_mat=netpct_sumsq_max_mat, netpct_sumsq_linear_mat=netpct_sumsq_linear_mat, netpct_sumsq_square_mat=netpct_sumsq_square_mat,
            netpct_top2_max_mat=netpct_top2_max_mat, netpct_top2_linear_mat=netpct_top2_linear_mat, netpct_top2_square_mat=netpct_top2_square_mat,
            mean_grosspct_median_max=mean_grosspct_median_max, mean_grosspct_median_linear=mean_grosspct_median_linear, mean_grosspct_median_square=mean_grosspct_median_square,
            sd_grosspct_median_max=sd_grosspct_median_max, sd_grosspct_median_linear=sd_grosspct_median_linear, sd_grosspct_median_square=sd_grosspct_median_square,
            mean_netpct_median_max=mean_netpct_median_max, mean_netpct_median_linear=mean_netpct_median_linear, mean_netpct_median_square=mean_netpct_median_square,
            sd_netpct_median_max=sd_netpct_median_max, sd_netpct_median_linear=sd_netpct_median_linear, sd_netpct_median_square=sd_netpct_median_square,
            mean_grosspct_sumsq_max=mean_grosspct_sumsq_max, mean_grosspct_sumsq_linear=mean_grosspct_sumsq_linear, mean_grosspct_sumsq_square=mean_grosspct_sumsq_square,
            sd_grosspct_sumsq_max=sd_grosspct_sumsq_max, sd_grosspct_sumsq_linear=sd_grosspct_sumsq_linear, sd_grosspct_sumsq_square=sd_grosspct_sumsq_square,
            mean_netpct_sumsq_max=mean_netpct_sumsq_max, mean_netpct_sumsq_linear=mean_netpct_sumsq_linear, mean_netpct_sumsq_square=mean_netpct_sumsq_square,
            sd_netpct_sumsq_max=sd_netpct_sumsq_max, sd_netpct_sumsq_linear=sd_netpct_sumsq_linear, sd_netpct_sumsq_square=sd_netpct_sumsq_square,
            mean_grosspct_top2_max=mean_grosspct_top2_max, mean_grosspct_top2_linear=mean_grosspct_top2_linear, mean_grosspct_top2_square=mean_grosspct_top2_square,
            sd_grosspct_top2_max=sd_grosspct_top2_max, sd_grosspct_top2_linear=sd_grosspct_top2_linear, sd_grosspct_top2_square=sd_grosspct_top2_square,
            mean_netpct_top2_max=mean_netpct_top2_max, mean_netpct_top2_linear=mean_netpct_top2_linear, mean_netpct_top2_square=mean_netpct_top2_square,
            sd_netpct_top2_max=sd_netpct_top2_max, sd_netpct_top2_linear=sd_netpct_top2_linear, sd_netpct_top2_square=sd_netpct_top2_square,
            post_median_x=meta["post_median_x"], post_median_y=meta["post_median_y"],
            post_sumsq_x=meta["post_sumsq_x"], post_sumsq_y=meta["post_sumsq_y"],
            post_top2_x=meta["post_top2_x"], post_top2_y=meta["post_top2_y"],
        )
        os.replace(tmp_path, str(out_path))
    except Exception:
        try: os.remove(tmp_path)
        except Exception: pass
        raise

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Card game simulation - single run with command-line parameters"
    )

    # Required arguments
    ap.add_argument("--seed", type=int, required=True,
                    help="Random seed for reproducibility")
    ap.add_argument("--rounds", type=int, required=True,
                    help="Number of simulation rounds")

    # Simulation parameters
    ap.add_argument("--max_signals", type=int, default=9,
                    help="Maximum number of signals to test (0 to max_signals)")
    ap.add_argument("--procs", type=int, default=1,
                    help="Number of parallel processes for intra-run parallelism")

    # Game parameters
    ap.add_argument("--signal_cost", type=float, default=7.0,
                    help="Cost per signal in £")
    ap.add_argument("--scale_pay", type=int, default=0, choices=[0, 1],
                    help="Enable payoff scaling: 0=off, 1=on")
    ap.add_argument("--scale_param", type=float, default=0.25,
                    help="Scaling parameter (used when scale_pay=1)")
    ap.add_argument("--ace_payout", type=float, default=20.0,
                    help="Payout multiplier for ace piles")

    # Output
    ap.add_argument("--out", type=str, default=None,
                    help="Output path; default=output/<canonical_id>.npz")

    args = ap.parse_args()

    # Input validation
    if args.rounds <= 0:
        ap.error("--rounds must be positive")
    if args.procs < 1:
        ap.error("--procs must be >= 1")
    if args.max_signals > NUM_PILES:
        ap.error(f"--max_signals cannot exceed {NUM_PILES}")
    if args.signal_cost < 0:
        ap.error("--signal_cost cannot be negative")
    if args.ace_payout <= 0:
        ap.error("--ace_payout must be positive")

    # Build parameter dict
    raw_params = dict(
        signal_cost=float(args.signal_cost),
        scale_pay=int(args.scale_pay),
        scale_param=float(args.scale_param),
        ace_payout=float(args.ace_payout),
    )

    # Canonicalize and generate stable ID
    norm_params, key_tuple, key_id = canonicalize_params(raw_params)

    # Determine output path
    out_path = pathlib.Path(args.out) if args.out else pathlib.Path("output") / f"{key_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Configuration: {key_id}")
    print(f"Parameters: {norm_params}")
    print(f"Running {args.rounds} rounds with up to {args.max_signals} signals...")

    t0 = perf_counter()
    dist, summary, meta = simulate_experiment(
        seed_int=int(args.seed),
        rounds=int(args.rounds),
        max_signals=int(args.max_signals),
        procs=int(args.procs),
        params=norm_params,
    )

    save_npz(out_path, args.rounds, args.max_signals, dist, summary, meta,
             norm_params, raw_params, key_tuple, key_id)

    t1 = perf_counter()
    print(f"\nWrote {out_path}")
    print(f"Total time: {t1 - t0:.2f}s")

if __name__ == "__main__":
    # Disable threading in numpy for better multiprocessing
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
    main()

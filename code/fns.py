# Standard imports required by type hints and body
import pathlib
import numpy as np
import tempfile
import os
import json
import zlib

NUM_PILES       = 9
CARDS_PER_PILE  = 5
RANKS           = np.arange(2, 15)          # 2..14, where 14 is Ace
ACE_RANK        = 14
COPIES_PER_RANK = 4                         # one standard deck
BUDGET          = 100.0
ACE_SUCCESS_PROB= 1.0

# Prior: expected # of Ace piles on board = 0.5 (uniform K in {0,1} => E[K]=0.5)
P0_SINGLE_PILE  = 0.5 / NUM_PILES

# Canonical fill-ins when a switch is OFF
CANON_SCALE_PARAM_DEFAULT = 0   # used when scale_pay == 0


# ---------------------------------------------
# Helpers: formatting + IDs - Verified all ok
# ---------------------------------------------

def _fmt_float_id(x: float, nd=4) -> str:
    s = f"{x:.{nd}f}"
    return s.replace(".", "p")

#  returns a parameter dict that rules out redundant configurations in parameter sweep
def canonicalize_params(raw):
    sc  = float(raw["signal_cost"])
    sp  = int(raw["scale_pay"])
    s   = float(raw["scale_param"])
    ap  = float(raw["ace_payout"])

    #  use grid of payoff scaling only when SCALE_PAYS is 1
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

def seed_for_id(base_seed: int, key_id: str) -> int:
    return int(np.uint32(zlib.adler32(key_id.encode("utf-8")) ^ np.uint32(base_seed)))


# -------------------------------------
# Helpers: pile composition and pile statistics utilities
# -------------------------------------

def _all_multicounts(total, n_ranks, cap):
    c = [0] * n_ranks
    def rec(i, remaining):
        if i == n_ranks - 1:
            if 0 <= remaining <= cap:
                c[i] = remaining
                yield tuple(c)
            return
        max_take = min(cap, remaining)
        for t in range(max_take + 1):
            c[i] = t
            yield from rec(i + 1, remaining - t)
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

# -------------------
# Save results
# -------------------

def save_npz(out_path: pathlib.Path, args, dist, summary, meta, norm_params, raw_params, key_tuple, key_id, hists=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sig_grid = np.arange(args.max_signals + 1, dtype=int)

    def pack_summary(key, stat_key):
        return np.array([summary[key][s][stat_key] for s in sig_grid], float)

    payload = dict(
        mode=str(meta.get("mode", "dup")), id=key_id, key_tuple=np.array(key_tuple, dtype=object),
        params_raw=json.dumps(raw_params), params_norm=json.dumps(norm_params),
        rounds=int(args.rounds), budget=float(BUDGET), p0_single_pile=float(P0_SINGLE_PILE),
        sig_grid=sig_grid,
    )
    if "stage1_alloc" in meta:
        payload["stage1_alloc"] = float(meta.get("stage1_alloc"))

    # Add per-regime stats for whatever regimes are present in summary
    regimes = list(summary.keys())
    for reg in regimes:
        try:
            payload[f"mean_net_return_{reg}_max"]    = pack_summary(reg, "mean_net_return_max")
            payload[f"mean_net_return_{reg}_linear"] = pack_summary(reg, "mean_net_return_linear")
            payload[f"mean_net_return_{reg}_top5"]   = pack_summary(reg, "mean_net_return_top5")
            payload[f"sd_net_return_{reg}_max"]      = pack_summary(reg, "sd_net_return_max")
            payload[f"sd_net_return_{reg}_linear"]   = pack_summary(reg, "sd_net_return_linear")
            payload[f"sd_net_return_{reg}_top5"]     = pack_summary(reg, "sd_net_return_top5")
        except KeyError:
            # Skip incomplete regimes gracefully
            pass

    # Add posterior meta curves if provided
    for k in ("post_median_x","post_median_y","post_top2_x","post_top2_y","post_max_x","post_max_y","post_min_x","post_min_y"):
        if k in meta:
            payload[k] = meta[k]

    # Histogram meta
    payload["hist_start"] = float(meta.get("hist_start", -100.0))
    payload["hist_step"]  = float(meta.get("hist_step", 1.0))
    payload["hist_n"]     = int(meta.get("hist_n", 0))

    # Add hist counts if present
    if isinstance(hists, dict):
        for reg, rules in hists.items():
            for rule_name, arr in rules.items():
                payload[f"hist_counts_{reg}_{rule_name}"] = arr

    fd, tmp_path = tempfile.mkstemp(prefix=out_path.stem + '.', suffix='.npz', dir=str(out_path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, str(out_path))
    except Exception:
        try: os.remove(tmp_path)
        except Exception: pass
        raise

# card_game.py
import numpy as np
import argparse
import pandas as pd

from web_wrangler import run_ui  # UI server only
from card_game import _deal_cards_global_deck, round_seed

# ------- game parameters -------
MIN_INV = {1: 1.0, 2: 5.0}  # Only 2 stages now
WALLET0 = 100.0
ACE_PAYOUT = 20.0  # Default ace payout multiplier
ACE_RANK = 14
# Flags: restrict to a single signal type and cost
SIGNAL_MODE = "median"  # or "top2"
SIGNAL_COST = 5.0
CARD_VALUES = np.arange(2, 15)  # 2..10, J=11, Q=12, K=13, A=14


# ===============================
# Deck and risks
# ===============================
def draw_deck(n_blue: int, n_red: int, seed: int | None = None) -> pd.DataFrame:
    """Compatibility wrapper returning a 9-pile blue-only board using global deck.

    Populates per-pile stats so the web UI can show actual signal values.
    Includes second-highest rank (R2) for dynamic two-stage model.
    """
    rng = np.random.default_rng(seed)
    has_ace, hands, medians, top2sum, max_rank, min_rank = _deal_cards_global_deck(rng)
    rows = []
    for i in range(len(hands)):
        # Extract second-highest rank from sorted hand
        hand_sorted = np.sort(hands[i])
        second_rank = int(hand_sorted[-2]) if len(hand_sorted) >= 2 else int(hand_sorted[-1])

        rows.append({
            "card_id": int(i),
            "color": "blue",
            "alive": True,
            "round": 0,
            # legacy field; keep for fallback (max rank)
            "N": int(max_rank[i]),
            # explicit fields consumed by UI
            "med": int(medians[i]),
            "sum2": int(top2sum[i]),
            "max_rank": int(max_rank[i]),
            "second_rank": second_rank,  # R2 for Stage 2 reveal
        })
    return pd.DataFrame(rows)


def apply_systemic_risk(df: pd.DataFrame, rng=None):
    """Apply block events. Return (df, blue_event, red_event)."""
    rng = np.random.default_rng() if rng is None else rng
    blue_event = rng.random() < 0.01
    red_event  = rng.random() < 0.10
    if blue_event:
        df.loc[(df["color"] == "blue") & df["alive"], "alive"] = False
    if red_event:
        df.loc[(df["color"] == "red") & df["alive"], "alive"] = False
    return df, blue_event, red_event


def apply_idiosyncratic_risk(df: pd.DataFrame, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    alive = df["alive"].to_numpy()
    is_blue = (df["color"].to_numpy() == "blue")
    is_red = ~is_blue

    p = np.zeros(len(df), float)
    p[is_blue] = 0.01
    p[is_red]  = np.minimum(0.03 * df["N"].to_numpy()[is_red], 1.0)

    fails = rng.random(len(df)) < p
    df.loc[alive & fails, "alive"] = False
    return df


def step_round(df: pd.DataFrame, round_idx: int, rng=None):
    """Run systemic then idiosyncratic; return (df, blue_event, red_event)."""
    rng = np.random.default_rng() if rng is None else rng
    df, b_evt, r_evt = apply_systemic_risk(df, rng)
    df = apply_idiosyncratic_risk(df, rng)
    df.loc[df["alive"], "round"] = round_idx
    return df, b_evt, r_evt


# ===============================
# Payoffs
# ===============================
def compute_payoffs_at_stage2(df: pd.DataFrame, ace_payout: float = ACE_PAYOUT) -> pd.DataFrame:
    """Compute payoffs after Stage 2 (end of game).

    Stage 1 investments pay full ace_payout for Ace.
    Stage 2 investments pay 0.5x ace_payout for Ace.
    Only highest card in pile determines payout (Ace-only model).
    """
    out = df[(df["alive"]) & (df["round"] == 2)].copy()

    # Ace-only model: only Aces pay out
    is_ace = out["N"] == ACE_RANK

    for c in ("inv1", "inv2"):
        if c not in out.columns:
            out[c] = 0.0

    # Stage 1 payout: full ace_payout
    out["payout1"] = np.where(is_ace, out["inv1"] * ace_payout, 0.0)

    # Stage 2 payout: 0.5x ace_payout
    out["payout2"] = np.where(is_ace, out["inv2"] * 0.5 * ace_payout, 0.0)

    out["stake"] = out[["inv1", "inv2"]].sum(axis=1)
    out["payout"] = out["payout1"] + out["payout2"]

    return out


# ===============================
# Signals
# ===============================
def make_signal_catalog_manual() -> pd.DataFrame:
    data = [
        (1, 1, "A–7 or 8–K"),
        (2, 2, "A/2–5/6–9/10–K"),
        (3, 4, "2–3 / 4–5 / 6–7 / 8–9 / 10–J / Q–K / A"),
        (4, 8, "Exact number (2–14)"),
    ]
    df = pd.DataFrame(data, columns=["signal_id", "cost", "info"])
    df["signal_name"] = [f"Signal {i}" for i in df["signal_id"]]
    return df.set_index("signal_id")[["signal_name", "cost", "info"]]


def reveal_signal(N: int, signal_id: int) -> str:
    if signal_id == 1:
        return "A–7" if (N == 14 or 2 <= N <= 7) else "8–K"
    if signal_id == 2:
        if N == 14:
            return "A"
        if 2 <= N <= 5:
            return "2–5"
        if 6 <= N <= 9:
            return "6–9"
        return "10–K"
    if signal_id == 3:
        if 2 <= N <= 3:
            return "2–3"
        if 4 <= N <= 5:
            return "4–5"
        if 6 <= N <= 7:
            return "6–7"
        if 8 <= N <= 9:
            return "8–9"
        if 10 <= N <= 11:
            return "10–J"
        if 12 <= N <= 13:
            return "Q–K"
        return "A"
    if signal_id == 4:
        return f"{N}"
    raise ValueError("signal_id must be 1..4")


def stage_buy_signals(
    deck: pd.DataFrame,
    purchases: dict[int, list[int]],
    budget: float,
    catalog: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, float, float]:
    catalog = make_signal_catalog_manual() if catalog is None else catalog
    deck = deck.copy()
    for col in ("s1", "s2", "s3", "s4"):
        if col not in deck.columns:
            deck[col] = pd.Series([None] * len(deck), dtype="object")
    if "signals_spend" not in deck.columns:
        deck["signals_spend"] = 0.0

    spend = 0.0
    for cid, sigs in purchases.items():
        idx = deck.index[deck["card_id"].eq(cid)]
        if idx.empty:
            continue
        i = idx[0]
        if not bool(deck.at[i, "alive"]):
            continue
        N = int(deck.at[i, "N"])
        for s in sigs:
            if s not in catalog.index:
                continue
            c = float(catalog.loc[s, "cost"])
            if spend + c > budget:
                break
            deck.at[i, f"s{s}"] = str(reveal_signal(N, s))
            deck.at[i, "signals_spend"] = float(deck.at[i, "signals_spend"]) + c
            spend += c
    return deck, spend, budget - spend


# ===============================
# Demo loop (UI in web_wrangler)
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VC card web game")
    parser.add_argument("--signal-mode", dest="signal_mode", choices=["median", "top2"], default=SIGNAL_MODE, help="Which single signal to offer")
    parser.add_argument("--signal-cost", dest="signal_cost", type=float, default=SIGNAL_COST, help="Cost of the signal in pounds")
    parser.add_argument("--no-open", dest="open_browser", action="store_false", help="Do not auto-open browser on stage 1")
    args = parser.parse_args()

    mode = args.signal_mode
    cost = float(args.signal_cost)
    open_first = args.open_browser
    ace_pay = ACE_PAYOUT
    # 9 piles, all blue
    df = draw_deck(n_blue=9, n_red=0, seed=42)
    for c in ("inv1", "inv2"):
        if c not in df.columns:
            df[c] = 0.0

    wallet = WALLET0
    sys_blue = False
    sys_red = False
    stage_history = []  # Track stage-wise stats

    # ---- Stage 1 ----
    act = run_ui(1, df, wallet, open_browser=open_first, signal_mode=mode, signal_cost=cost)
    if act is None:
        raise RuntimeError("Stage 1 UI returned None - did the server fail?")
    df, s_spent, _ = stage_buy_signals(df, {int(k): v for k, v in act["purchases"].items()}, budget=wallet)
    wallet = max(0.0, wallet - float(s_spent))

    inv_need = {int(k): max(float(v), MIN_INV[1]) for k, v in act["invest"].items() if float(v) > 0}
    stage1_stakes = sum(inv_need.values())
    need = stage1_stakes
    if need > wallet:
        left = wallet
        for cid in list(inv_need.keys()):
            take = min(inv_need[cid], left)
            inv_need[cid] = take
            left -= take
            if left <= 0:
                break
        wallet = 0.0
        stage1_stakes = sum(inv_need.values())  # Update to actual invested amount
    else:
        wallet -= need
    for cid, amt in inv_need.items():
        ix = df.index[df["card_id"].eq(cid)]
        if len(ix) and df.at[ix[0], "alive"] and amt > 0:
            df.at[ix[0], "inv1"] += amt
    df, b1, r1 = step_round(df, 1, rng=np.random.default_rng(101))
    sys_blue |= b1
    sys_red  |= r1

    # Record Stage 1 history
    stage_history.append({"signals": float(s_spent), "stakes": float(stage1_stakes)})

    # Track which piles were invested in Stage 1
    stage1_invested_ids = [int(cid) for cid in df[df["inv1"] > 0]["card_id"].tolist()]

    # ---- Stage 2 ----
    # Only allow investing in piles that were invested in Stage 1
    act = run_ui(2, df, wallet, signal_mode=mode, signal_cost=cost, stage1_invested=stage1_invested_ids, stage_history=stage_history)
    if act is None:
        raise RuntimeError("Stage 2 UI returned None - did the server fail?")
    df, s_spent, _ = stage_buy_signals(df, {int(k): v for k, v in act["purchases"].items()}, budget=wallet)
    wallet = max(0.0, wallet - float(s_spent))

    inv_need = {int(k): max(float(v), MIN_INV[2]) for k, v in act["invest"].items() if float(v) > 0}
    stage2_stakes = sum(inv_need.values())
    need = stage2_stakes
    if need > wallet:
        left = wallet
        for cid in list(inv_need.keys()):
            take = min(inv_need[cid], left)
            inv_need[cid] = take
            left -= take
            if left <= 0:
                break
        wallet = 0.0
        stage2_stakes = sum(inv_need.values())  # Update to actual invested amount
    else:
        wallet -= need
    for cid, amt in inv_need.items():
        ix = df.index[df["card_id"].eq(cid)]
        if len(ix) and df.at[ix[0], "alive"] and amt > 0:
            df.at[ix[0], "inv2"] += amt
    df, b2, r2 = step_round(df, 2, rng=np.random.default_rng(102))
    sys_blue |= b2
    sys_red  |= r2

    # Record Stage 2 history
    stage_history.append({"signals": float(s_spent), "stakes": float(stage2_stakes)})

    # ---- Compute Results and Show Performance ----
    pay = compute_payoffs_at_stage2(df, ace_payout=ace_pay)

    inv_sum = df.get("inv1", 0) + df.get("inv2", 0)
    total_invest = float(inv_sum.sum())
    total_payoff = float(pay["payout"].sum()) if len(pay) else 0.0
    n_invested = int((inv_sum > 0).sum())
    n_wiped = int(((inv_sum > 0) & (~df["alive"])).sum())  # invested cards that died
    n_red_invested = int((df["color"].eq("red") & (inv_sum > 0)).sum())
    # signals spent
    total_signals_spend = float(df.get("signals_spend", 0).fillna(0).sum())
    # avg signals per invested card
    def _row_sig_count(row):
        cnt = 0
        for k in (1, 2, 3, 4):
            v = row.get(f"s{k}")
            if pd.notna(v) and v is not None and str(v) != "None":
                cnt += 1
        return cnt
    if n_invested > 0:
        avg_signals = float(df.loc[(inv_sum > 0)].apply(_row_sig_count, axis=1).mean())
    else:
        avg_signals = 0.0

    net_return_abs = total_payoff - total_invest
    net_return_pct = (net_return_abs / total_invest * 100.0) if total_invest > 0 else 0.0
    sys_tag = "both" if (sys_blue and sys_red) else ("blue" if sys_blue else ("red" if sys_red else "none"))

    stats = {
        "player": "",                # filled by UI
        "wallet_left": wallet,       # remaining budget
        "invested": total_invest,    # total invested £
        "signals_spent": total_signals_spend,     # total spent on signals £
        "net_return": net_return_abs,             # £
        "net_return_pct": net_return_pct,         # %
        "n_invested": n_invested,
        "n_wiped": n_wiped,
        "sys_wipe": sys_tag,                       # "none" | "blue" | "red" | "both"
        "n_red_invested": n_red_invested,
        "avg_signals": avg_signals,
    }

    # ---- Show Results (triggered from Stage 2) ----
    # The UI will redirect to /results when Stage 2's "Show performance" is clicked
    _ = run_ui(2, df, wallet, results=stats, signal_mode=mode, signal_cost=cost)


    # Optional console dump
    if len(pay):
        print(
            pay[["card_id", "color", "N", "stake", "payout"]]
            .sort_values("payout", ascending=False)
            .to_string(index=False)
        )

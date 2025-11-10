# card_game.py
import numpy as np
import argparse
import pandas as pd

from web_wrangler import run_ui  # UI server only
from sim_res import _deal_cards_global_deck, round_seed

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
def draw_deck(n_cards: int, seed: int | None = None) -> pd.DataFrame:
    """Return a board with n_cards piles using global deck.

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


def step_round(df: pd.DataFrame, round_idx: int):
    """Increment round counter for all alive cards. No risk applied."""
    df.loc[df["alive"], "round"] = round_idx
    return df


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
    *,
    per_signal_cost: float | None = None,
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
            # Always use a fixed single-signal price when provided (new game version)
            c = float(per_signal_cost) if per_signal_cost is not None else float(catalog.loc[s, "cost"])
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
    # 9 piles
    df = draw_deck(n_cards=9, seed=42)
    for c in ("inv1", "inv2"):
        if c not in df.columns:
            df[c] = 0.0

    wallet = WALLET0
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
    df = step_round(df, 1)

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
    df = step_round(df, 2)

    # Record Stage 2 history
    stage_history.append({"signals": float(s_spent), "stakes": float(stage2_stakes)})

    # ---- Compute Results and Show Performance ----
    pay = compute_payoffs_at_stage2(df, ace_payout=ace_pay)

    inv_sum = df.get("inv1", 0) + df.get("inv2", 0)
    total_invest = float(inv_sum.sum())
    total_payoff = float(pay["payout"].sum()) if len(pay) else 0.0
    n_invested = int((inv_sum > 0).sum())
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

    # Calculate player portfolio weights (all 9 piles)
    player_weights = []
    for idx in range(9):
        if idx < len(df):
            inv_amt = float(inv_sum.iloc[idx]) if idx < len(inv_sum) else 0.0
            player_weights.append(round(inv_amt, 2))
        else:
            player_weights.append(0.0)

    # Count ace/king/queen hits in surviving invested cards
    ace_hits = 0
    king_hits = 0
    queen_hits = 0
    if len(pay) > 0:
        for _, row in pay.iterrows():
            card_inv = float(row.get("stake", 0))
            if card_inv > 0:
                max_rank = int(row.get("N", 0))
                if max_rank == 14:  # Ace
                    ace_hits += 1
                elif max_rank == 13:  # King
                    king_hits += 1
                elif max_rank == 12:  # Queen
                    queen_hits += 1

    stats = {
        "player": "",                # filled by UI
        "wallet_left": wallet,       # remaining budget
        "invested": total_invest,    # total invested £
        "signals_spent": total_signals_spend,     # total spent on signals £
        "net_return": net_return_abs,             # £
        "net_return_pct": net_return_pct,         # %
        "n_invested": n_invested,
        "avg_signals": avg_signals,
        "player_weights": player_weights,         # [9 pile weights]
        "ace_hits": ace_hits,
        "king_hits": king_hits,
        "queen_hits": queen_hits,
    }

    # ---- Show Results (triggered from Stage 2) ----
    # Start a persistent server to serve /results page
    # This server will keep running until user clicks "End Game"
    print("[web] Results ready. Server will stay up until 'End Game' is clicked.")
    _ = run_ui(stage=3, df=df, wallet=wallet, results=stats, signal_mode=mode, signal_cost=cost)


    # Optional console dump
    if len(pay):
        print(
            pay[["card_id", "N", "stake", "payout"]]
            .sort_values("payout", ascending=False)
            .to_string(index=False)
        )

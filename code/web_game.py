# card_game.py
import numpy as np
import argparse
import pandas as pd

from web_wrangler import run_ui  # UI server only
from sim_res import _deal_cards_global_deck, round_seed
from database import init_db, create_session, log_stage_action, log_game_results, close_session, mark_session_completed, delete_session
from simulate_policy import run_policy_simulation

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
        # Extract second-highest UNIQUE rank (matching frontier.py logic)
        hand_sorted = np.sort(hands[i])
        unique_ranks = sorted(set(hand_sorted.tolist()), reverse=True)
        second_rank = int(unique_ranks[1]) if len(unique_ranks) >= 2 else int(unique_ranks[0])

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

    # Initialize database (once at startup)
    init_db()

    # Game loop - restart after each game ends
    print("[game] Starting game server. Press Ctrl+C to stop.")
    while True:
        try:
            # 9 piles - generate random seed for each new game
            game_seed = np.random.randint(0, 1_000_000)
            df = draw_deck(n_cards=9, seed=game_seed)
            print(f"[game] New game started with seed: {game_seed}")
            print(f"[game] Board dealt (9 piles):")
            for _, row in df.iterrows():
                rank_label = {14: 'A', 13: 'K', 12: 'Q', 11: 'J'}.get(row['max_rank'], str(row['max_rank']))
                r2_label = {14: 'A', 13: 'K', 12: 'Q', 11: 'J'}.get(row['second_rank'], str(row['second_rank']))
                print(f"  Pile {row['card_id']}: max={rank_label}({row['max_rank']}), "
                      f"R2={r2_label}({row['second_rank']}), median={row['med']}, top2={row['sum2']}")
            for c in ("inv1", "inv2"):
                if c not in df.columns:
                    df[c] = 0.0

            wallet = WALLET0
            stage_history = []  # Track stage-wise stats
            total_signal_cost_stage1 = 0.0  # Explicit tracker for Stage 1 signals
            total_signal_cost_stage2 = 0.0  # Explicit tracker for Stage 2 signals
            session_id = None  # Will be created after Stage 1 submission

            # ---- Stage 1 ----
            act = run_ui(1, df, wallet, open_browser=open_first, signal_mode=mode, signal_cost=cost)
            if act is None:
                # Player closed browser or restarted - don't create DB session
                print("[game] Stage 1 returned None - game abandoned before submission")
                continue  # Start fresh game

            # Create database session AFTER Stage 1 submission (game has legitimate progress)
            team_name = act.get("player_name", "Anonymous")
            session_id = create_session(
                team_name=team_name,
                seed=game_seed,
                signal_mode=mode,
                signal_cost=cost
            )

            df, s_spent, _ = stage_buy_signals(
                df, {int(k): v for k, v in act["purchases"].items()},
                budget=wallet,
                per_signal_cost=cost
            )
            total_signal_cost_stage1 = float(s_spent)
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
            signal_count_stage1 = sum(len(v) for v in act.get("purchases", {}).values())
            stage_history.append({
                "signals": float(total_signal_cost_stage1),
                "signal_count": int(signal_count_stage1),
                "stakes": float(stage1_stakes)
            })

            # Log Stage 1 to database
            log_stage_action(
                session_id=session_id,
                stage=1,
                purchases=act.get("purchases", {}),
                investments=act.get("invest", {}),
                signals_spent=total_signal_cost_stage1,
                stakes_invested=stage1_stakes,
                budget_remaining=wallet
            )

            # Track which piles were invested in Stage 1
            stage1_invested_ids = [int(cid) for cid in df[df["inv1"] > 0]["card_id"].tolist()]

            # ---- Stage 2 ----
            print(f"[game] Stage 1 complete. Starting Stage 2 with wallet=£{wallet:.2f}")
            # Only allow investing in piles that were invested in Stage 1
            act = run_ui(2, df, wallet, signal_mode=mode, signal_cost=cost, stage1_invested=stage1_invested_ids, stage_history=stage_history)
            print(f"[game] Stage 2 UI returned, processing actions")
            if act is None:
                raise RuntimeError("Stage 2 UI returned None - did the server fail?")
            df, s_spent, _ = stage_buy_signals(
                df, {int(k): v for k, v in act["purchases"].items()},
                budget=wallet,
                per_signal_cost=cost
            )
            total_signal_cost_stage2 = float(s_spent)
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
            signal_count_stage2 = sum(len(v) for v in act.get("purchases", {}).values())
            stage_history.append({
                "signals": float(total_signal_cost_stage2),
                "signal_count": int(signal_count_stage2),
                "stakes": float(stage2_stakes)
            })

            # Log Stage 2 to database
            log_stage_action(
                session_id=session_id,
                stage=2,
                purchases=act.get("purchases", {}),
                investments=act.get("invest", {}),
                signals_spent=total_signal_cost_stage2,
                stakes_invested=stage2_stakes,
                budget_remaining=wallet
            )

            # ---- Compute Results and Show Performance ----
            pay = compute_payoffs_at_stage2(df, ace_payout=ace_pay)

            inv_sum = df.get("inv1", 0) + df.get("inv2", 0)
            total_invest = float(inv_sum.sum())
            total_payoff = float(pay["payout"].sum()) if len(pay) else 0.0
            n_invested = int((inv_sum > 0).sum())
            # signals spent - use explicit trackers as source of truth
            total_signals_spend = float(total_signal_cost_stage1 + total_signal_cost_stage2)
            # Verification: check DataFrame accumulation matches
            df_signals_total = float(df.get("signals_spend", 0).fillna(0).sum())
            if abs(total_signals_spend - df_signals_total) > 0.01:
                print(f"[WARNING] Signal cost mismatch: explicit={total_signals_spend}, df={df_signals_total}")
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

            # Net returns: use frontier formula for consistency
            # mean_net = 100 * (c1*(g1-1) + c2*(g2-1) - signal_cost_fraction)
            stage1_payoff = float(pay["payout1"].sum()) if len(pay) else 0.0
            stage2_payoff = float(pay["payout2"].sum()) if len(pay) else 0.0

            # Investable amounts per stage (after signal costs)
            investable_stage1 = stage1_stakes
            investable_stage2 = stage2_stakes

            # Fractions of total budget
            c1 = investable_stage1 / WALLET0 if WALLET0 > 0 else 0.0
            c2 = investable_stage2 / WALLET0 if WALLET0 > 0 else 0.0
            signal_cost_fraction = total_signals_spend / WALLET0 if WALLET0 > 0 else 0.0

            # Gross return multipliers per stage
            g1 = stage1_payoff / investable_stage1 if investable_stage1 > 0 else 0.0
            g2 = stage2_payoff / investable_stage2 if investable_stage2 > 0 else 0.0

            # Net return using frontier formula
            net_return_pct = 100.0 * (c1 * (g1 - 1.0) + c2 * (g2 - 1.0) - signal_cost_fraction)
            net_return_abs = total_payoff - total_invest

            # Stage 1 fraction (for strategic analysis)
            total_stakes = stage1_stakes + stage2_stakes
            stage1_fraction = stage1_stakes / total_stakes if total_stakes > 0 else 0.0

            # Calculate player portfolio weights (all 9 piles)
            player_weights = []
            for idx in range(9):
                if idx < len(df):
                    inv_amt = float(inv_sum.iloc[idx]) if idx < len(inv_sum) else 0.0
                    player_weights.append(round(inv_amt, 2))
                else:
                    player_weights.append(0.0)

            # Concentration index (Σw² / 10000) - risk proxy for MV frontier
            concentration_index = sum(w ** 2 for w in player_weights) / 10000.0

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
                "net_return_pct": net_return_pct,         # % (frontier formula)
                "n_invested": n_invested,
                "avg_signals": avg_signals,
                "player_weights": player_weights,         # [9 pile weights]
                "ace_hits": ace_hits,
                "king_hits": king_hits,
                "queen_hits": queen_hits,
                "concentration_index": concentration_index,  # Σw²/10000 for MV frontier
                "stage1_fraction": stage1_fraction,          # % of stakes in Stage 1
            }

            # Log results to database
            log_game_results(session_id=session_id, results=stats)

            # ---- Run Policy Simulation (10k rounds) ----
            print("[game] Running policy simulation (10k rounds)...")
            total_n_signals = signal_count_stage1 + signal_count_stage2
            # Calculate stage1_alloc from actual budget allocation
            # stage1_alloc = (stage1_stakes + signal_costs_stage1) / WALLET0
            budget_stage1 = stage1_stakes + total_signal_cost_stage1
            stage1_alloc = budget_stage1 / WALLET0 if WALLET0 > 0 else 0.5
            try:
                sim_returns, sim_metadata = run_policy_simulation(
                    n_signals=total_n_signals,
                    signal_type=mode,
                    signal_cost=cost,
                    stage1_alloc=stage1_alloc,  # Calculated from actual budget split
                    ace_payout=ACE_PAYOUT,
                    scale_pay=0,  # Hardcoded for now (ace-only payoff)
                    scale_param=0.0,
                    player_concentration=concentration_index,
                    rounds=10000,
                )
                stats["sim_returns"] = sim_returns.tolist()
                stats["sim_metadata"] = sim_metadata
                print(f"[game] Simulation complete: mean={sim_metadata['mean']:.2f}%, std={sim_metadata['std']:.2f}%")
            except Exception as e:
                print(f"[game] Policy simulation failed: {e}")
                stats["sim_returns"] = []
                stats["sim_metadata"] = {}

            # ---- Show Results (triggered from Stage 2) ----
            # Start a persistent server to serve /results page
            # This server will keep running until user clicks "End Game"
            print("[web] Results ready. Server will stay up until 'End Game' is clicked.")
            _ = run_ui(stage=3, df=df, wallet=wallet, results=stats, signal_mode=mode, signal_cost=cost, session_id=session_id)

            # Mark session as completed (player clicked End Game)
            mark_session_completed(session_id=session_id)

            # Close database session
            close_session(session_id=session_id)

            # Optional console dump
            if len(pay):
                print(
                    pay[["card_id", "N", "stake", "payout"]]
                    .sort_values("payout", ascending=False)
                    .to_string(index=False)
                )

        except KeyboardInterrupt:
            print("\n[game] Server stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] Game encountered an error: {e}")
            import traceback
            traceback.print_exc()
            print("[game] Restarting game in 3 seconds...")
            import time
            time.sleep(3)

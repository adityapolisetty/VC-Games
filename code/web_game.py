# card_game.py
import numpy as np
import argparse
import pandas as pd
import json
import os

from web_wrangler_fixed import run_ui, start_persistent_server, reset_game_state  # UI server only - using fixed singleton version
from sim_res import _deal_cards_global_deck  # DEAD CODE REMOVED: round_seed import (unused)
from database import init_db, create_session, log_stage_action, log_game_results, close_session, mark_session_completed, delete_session, get_leaderboard_by_signal_type
from simulate_policy import run_policy_simulation

# ------- game parameters -------
MIN_INV = {1: 1.0, 2: 5.0}  # Only 2 stages now
WALLET0 = 100.0
ACE_PAYOUT = 20.0  # Default ace payout multiplier
ACE_RANK = 14
# Flags: restrict to a single signal type and cost
SIGNAL_MODE = "median"  # or "top2"
SIGNAL_COST = 3.0
CARD_VALUES = np.arange(2, 15)  # 2..10, J=11, Q=12, K=13, A=14


# ===============================
# Frontier Data Loading
# ===============================
def load_all_alpha_frontiers(signal_type: str, use_v2: bool = False) -> dict:
    """Load all pre-computed frontier NPZ files for all alpha values (stage allocations).

    Args:
        signal_type: "median" or "top2"
        use_v2: If True, load from frontier_output_v2 (enhanced with Stage 2 concentration)

    Returns:
        Dictionary mapping alpha_pct (int, 0-100 in increments of 5) to frontier data:
        {
            0: {data for alpha=0.00},
            5: {data for alpha=0.05},
            ...
            100: {data for alpha=1.00}
        }

        Each frontier data dict contains:
        {
            "points_by_n": [  # List of 10 elements (n=0 to n=9 signals)
                [  # List of frontier points for this n
                    {
                        "sd": float,           # Standard deviation (%)
                        "mean_gross": float,   # Mean gross return multiplier
                        "max_gross": float,    # Max gross return from simulation
                        "sharpe": float,       # Sharpe ratio
                        "weights": [9 floats], # Portfolio weights (sum to 1.0)
                        "concentration": float, # Sum of squared weights
                        "ace_hit_rate": float, # Fraction of rounds hitting ace
                        "king_hit_rate": float,
                        "queen_hit_rate": float
                    },
                    ...
                ],
                ...
            ],
            "meta": {...}  # Metadata from NPZ file
        }
    """
    # Frontier files are in code directory (for Railway deployment)
    dir_name = 'frontier_output_v2' if use_v2 else 'frontier_output'
    frontier_dir = os.path.join(os.path.dirname(__file__), dir_name)
    frontier_dir = os.path.abspath(frontier_dir)  # Resolve to absolute path

    print(f"[frontier] Looking for frontier files in: {frontier_dir} (v2={use_v2})")
    print(f"[frontier] Directory exists: {os.path.exists(frontier_dir)}")
    if os.path.exists(frontier_dir):
        print(f"[frontier] Files in directory: {len(os.listdir(frontier_dir))}")

    # Alpha values: 0, 5, 10, ..., 95, 100 (21 files)
    alpha_values = list(range(0, 105, 5))

    result = {}

    for alpha_pct in alpha_values:
        # Convert alpha percentage to 3-digit string (e.g., 10 -> "010")
        alpha_str = f"a{alpha_pct:03d}"

        # File pattern: sc3p0000_sp1_s0p2500_ap20p0000_{signal_type}_a{alpha}.npz
        # sp1 = scale_pay=1 (graduated payoffs), s0p2500 = scale_param=0.25
        filename = f"sc3p0000_sp1_s0p2500_ap20p0000_{signal_type}_{alpha_str}.npz"
        filepath = os.path.join(frontier_dir, filename)

        if not os.path.exists(filepath):
            print(f"[frontier] Warning: Missing file {filename}")
            continue

        try:
            with np.load(filepath, allow_pickle=True) as z:
                sd_step = float(z.get('sd_step', 0.1))
                sd_levels_by_n = z['sd_levels_by_n']        # List of 10 arrays (n=0 to 9)
                best_means_by_n = z['best_means_by_n']      # Net return percentages
                best_weights_by_n = z['best_weights_by_n']  # Portfolio weights

                # Optional hit rates
                ace_hits_by_n = z.get('best_ace_hits_by_n')
                king_hits_by_n = z.get('best_king_hits_by_n')
                queen_hits_by_n = z.get('best_queen_hits_by_n')

                # Metadata
                meta = json.loads(str(z['meta']))

                # Extract metadata values (matching vis_sim_res.py)
                total_rounds = meta.get("total_rounds", 200000)  # Default to 200k
                scale_pay = meta.get("params", {}).get("scale_pay", 0)

                # Parse frontier points for each n
                points_by_n = []

                for n in range(10):  # n=0 to n=9 signals
                    sd_vals = sd_levels_by_n[n]        # 1D array of SD values
                    mean_vals = best_means_by_n[n]    # 1D array of mean net return %
                    weight_vecs = best_weights_by_n[n] # 2D array of weight vectors

                    # Handle empty arrays (some n values might have no strategies)
                    if len(sd_vals) == 0:
                        points_by_n.append([])
                        continue

                    # Get hit rates for this n (before coarsening)
                    ace_hits = ace_hits_by_n[n] if ace_hits_by_n is not None and n < len(ace_hits_by_n) else np.array([])
                    king_hits = king_hits_by_n[n] if king_hits_by_n is not None and n < len(king_hits_by_n) else np.array([])
                    queen_hits = queen_hits_by_n[n] if queen_hits_by_n is not None and n < len(queen_hits_by_n) else np.array([])

                    # Coarsen frontier: bin by SD (5pp steps) and keep highest mean in each bin
                    sd_vals, mean_vals, weight_vecs, ace_hits, king_hits, queen_hits = _coarsen_frontier(
                        sd_vals, mean_vals, weight_vecs, ace_hits, king_hits, queen_hits, sd_step=5.0
                    )

                    # Build frontier points for this n
                    n_points = []
                    for i in range(len(sd_vals)):
                        sd_pct = float(sd_vals[i])
                        mean_net_pct = float(mean_vals[i])

                        # Convert net return % to gross return multiplier
                        # net_return = 22.2% -> gross_mult = 1.222
                        mean_gross = (mean_net_pct / 100.0) + 1.0

                        # Convert SD from net % to gross multiplier
                        # SD_net = 15% -> SD_gross = 0.15
                        sd_gross = sd_pct / 100.0

                        # Get weight vector
                        w_vec = np.asarray(weight_vecs[i], dtype=float)

                        # Calculate concentration (sum of squared weights)
                        concentration = float(np.sum(w_vec ** 2))

                        # Sharpe ratio: (mean_gross - 1) / sd_gross = net return / sd
                        sharpe = ((mean_gross - 1.0) / sd_gross) if sd_gross > 0 else 0.0

                        # Hit rates (now from coarsened arrays)
                        ace_rate = float(ace_hits[i]) if len(ace_hits) > 0 and i < len(ace_hits) else 0.0
                        king_rate = float(king_hits[i]) if len(king_hits) > 0 and i < len(king_hits) else 0.0
                        queen_rate = float(queen_hits[i]) if len(queen_hits) > 0 and i < len(queen_hits) else 0.0

                        # Note: max_gross is not in NPZ, so we'll compute it on frontend from individual simulations
                        # For now, use mean_gross as placeholder
                        max_gross = mean_gross * 1.5  # Rough estimate (actual max could be much higher)

                        point = {
                            "sd": sd_pct,                # SD of net returns (%)
                            "sd_gross": sd_gross,        # SD of gross returns (multiplier)
                            "mean_gross": mean_gross,
                            "max_gross": max_gross,  # Placeholder
                            "sharpe": sharpe,
                            "weights": w_vec.tolist(),
                            "concentration": concentration,
                            "ace_hit_rate": ace_rate,
                            "king_hit_rate": king_rate,
                            "queen_hit_rate": queen_rate,
                            "total_rounds": total_rounds,  # For hit rate calculation
                            "scale_pay": scale_pay,        # For conditional king/queen display
                        }
                        n_points.append(point)

                    points_by_n.append(n_points)

                result[alpha_pct] = {
                    "points_by_n": points_by_n,
                    "meta": meta
                }

                print(f"[frontier] Loaded {filename}: {sum(len(pts) for pts in points_by_n)} total points across n=0..9")

        except Exception as e:
            print(f"[frontier] Error loading {filename}: {e}")
            continue

    return result


def _coarsen_frontier(sd_vals, mean_vals, weight_vecs, ace_hits, king_hits, queen_hits, sd_step=5.0):
    """Coarsen frontier by binning SD values (5pp steps) and keeping highest mean in each bin.

    Args:
        sd_vals: 1D array of SD values (percentage)
        mean_vals: 1D array of mean values
        weight_vecs: 2D array of weight vectors
        ace_hits: 1D array of ace hit counts
        king_hits: 1D array of king hit counts
        queen_hits: 1D array of queen hit counts
        sd_step: Binning granularity in percentage points (default 5.0 for 5pp)

    Returns:
        Tuple of (coarsened_sd, coarsened_mean, coarsened_weights, coarsened_ace, coarsened_king, coarsened_queen)
    """
    if len(sd_vals) == 0:
        return sd_vals, mean_vals, weight_vecs, ace_hits, king_hits, queen_hits

    # Bin by SD with 5pp steps (0-5%, 5-10%, 10-15%, etc.)
    sd_bins = {}
    for i in range(len(sd_vals)):
        bin_idx = int(sd_vals[i] / 5.0)  # 5 percentage point bins
        if bin_idx not in sd_bins:
            sd_bins[bin_idx] = []
        sd_bins[bin_idx].append(i)

    # Keep point with highest mean in each bin
    coarsened_sd = []
    coarsened_mean = []
    coarsened_weights = []
    coarsened_ace = []
    coarsened_king = []
    coarsened_queen = []

    for bin_idx in sorted(sd_bins.keys()):
        indices = sd_bins[bin_idx]
        # Find index with highest mean
        best_idx = max(indices, key=lambda idx: mean_vals[idx])
        coarsened_sd.append(sd_vals[best_idx])
        coarsened_mean.append(mean_vals[best_idx])
        coarsened_weights.append(weight_vecs[best_idx])

        # Coarsen hit rates too
        if len(ace_hits) > 0:
            coarsened_ace.append(ace_hits[best_idx])
        if len(king_hits) > 0:
            coarsened_king.append(king_hits[best_idx])
        if len(queen_hits) > 0:
            coarsened_queen.append(queen_hits[best_idx])

    return (
        np.array(coarsened_sd),
        np.array(coarsened_mean),
        np.array(coarsened_weights),
        np.array(coarsened_ace) if len(coarsened_ace) > 0 else np.array([]),
        np.array(coarsened_king) if len(coarsened_king) > 0 else np.array([]),
        np.array(coarsened_queen) if len(coarsened_queen) > 0 else np.array([])
    )


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

    Graduated payoff model:
    - Ace (rank 14): 20x investment
    - King (rank 13): 5x investment (20 * 0.25)
    - Queen (rank 12): 1.25x investment (20 * 0.0625)
    - Others: 0x investment

    Stage 1 investments pay full multiplier.
    Stage 2 investments pay 0.5x multiplier.
    """
    out = df[(df["alive"]) & (df["round"] == 2)].copy()

    for c in ("inv1", "inv2"):
        if c not in out.columns:
            out[c] = 0.0

    # Calculate payout multiplier based on card rank (graduated payoffs)
    rank = out["N"]
    multiplier = np.where(
        rank == 14,  # Ace
        ace_payout,
        np.where(
            rank == 13,  # King
            ace_payout * 0.25,
            np.where(
                rank == 12,  # Queen
                ace_payout * 0.0625,
                0.0  # All other ranks
            )
        )
    )

    # Stage 1 payout: full multiplier
    out["payout1"] = out["inv1"] * multiplier

    # Stage 2 payout: 0.5x multiplier
    out["payout2"] = out["inv2"] * 0.5 * multiplier

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

    # Start persistent HTTP server (once at startup)
    start_persistent_server(port=8765, open_browser=open_first)

    # Game loop - restart after each game ends
    print("[game] Starting game server. Press Ctrl+C to stop.")
    while True:
        session_id = None  # Initialize outside try block for exception handler
        try:
            # CRITICAL: Reset all game state before starting new game
            reset_game_state()
            print("[game] Game state reset - ready for landing page")

            # ---- Stage 0: Landing Page ----
            # Show landing page and wait for player to enter name and choose game type
            # Don't deal board yet - wait until Enter is clicked
            print("[game] Stage 0: Showing landing page, waiting for Enter...")
            stage0_act = run_ui(0, pd.DataFrame(), 0.0, signal_mode=mode, signal_cost=cost)
            if stage0_act is None:
                # Player closed browser or timeout - restart
                print("[game] Stage 0 returned None - restarting")
                continue

            # Extract player info from Stage 0 submission
            team_name = stage0_act.get("player_name", "Team Alpha")
            game_type = stage0_act.get("game_type", "g1")
            # Update signal mode based on game type
            mode = "top2" if game_type == "g2" else "median"
            print(f"[game] Player '{team_name}' chose game type '{game_type}' (signal mode: {mode})")

            # NOW deal the board (after player has entered)
            game_seed = np.random.randint(0, 1_000_000)
            df = draw_deck(n_cards=9, seed=game_seed)
            print(f"[game] Board dealt with seed: {game_seed}")
            print(f"[game] 9 piles created:")
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
            # session_id already initialized outside try block

            # ---- Stage 1 ----
            print(f"[game] Starting Stage 1 with signal_mode={mode}")
            # Create database session NOW (after Stage 0 enter, before Stage 1 actions)
            session_id = create_session(
                team_name=team_name,
                seed=game_seed,
                signal_mode=mode,
                signal_cost=cost
            )

            act = run_ui(1, df, wallet, signal_mode=mode, signal_cost=cost)
            if act is None:
                # Player closed browser or restarted - delete abandoned session
                print("[game] Stage 1 returned None - game abandoned before submission")
                if session_id:
                    delete_session(session_id)
                continue  # Start fresh game

            df, s_spent, _ = stage_buy_signals(
                df, {int(k): v for k, v in act.get("purchases", {}).items()},
                budget=wallet,
                per_signal_cost=cost
            )
            total_signal_cost_stage1 = float(s_spent)
            wallet = max(0.0, wallet - float(s_spent))

            inv_need = {int(k): max(float(v), MIN_INV[1]) for k, v in act.get("invest", {}).items() if float(v) > 0}
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

            # Handle missing purchases (Stage 2 may not have signal purchases)
            purchases = act.get("purchases", {})
            df, s_spent, _ = stage_buy_signals(
                df, {int(k): v for k, v in purchases.items()},
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

            # Gross return multipliers for formula (payoff / budget - kept for net_return_pct)
            g1_formula = stage1_payoff / WALLET0 if WALLET0 > 0 else 0.0
            g2_formula = stage2_payoff / WALLET0 if WALLET0 > 0 else 0.0

            # Gross return multipliers for display (stage-specific denominators)
            # Stage 1: payoff / (invested + signal_cost)
            stage1_resources = investable_stage1 + total_signals_spend
            g1 = stage1_payoff / stage1_resources if stage1_resources > 0 else 0.0

            # DEBUG: Print Stage 1 gross return calculation
            print(f"[DEBUG] Stage 1 Gross Return:")
            print(f"  stage1_payoff = {stage1_payoff}")
            print(f"  investable_stage1 = {investable_stage1}")
            print(f"  total_signals_spend = {total_signals_spend}")
            print(f"  total_signal_cost_stage1 = {total_signal_cost_stage1}")
            print(f"  total_signal_cost_stage2 = {total_signal_cost_stage2}")
            print(f"  stage1_resources = {stage1_resources}")
            print(f"  g1 = {g1}")

            # Stage 2: payoff / invested
            g2 = stage2_payoff / investable_stage2 if investable_stage2 > 0 else 0.0

            # Total: payoff / budget (unchanged)
            gross_return_mult = total_payoff / WALLET0 if WALLET0 > 0 else 0.0

            # Net return using frontier formula (uses g1_formula/g2_formula, not display values)
            net_return_pct = 100.0 * (c1 * (g1_formula - 1.0) + c2 * (g2_formula - 1.0) - signal_cost_fraction)
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

            # Count ace/king/queen hits and track invested amounts and payoffs
            # Track both stage-specific and total values
            ace_hits = 0
            king_hits = 0
            queen_hits = 0

            # Total invested/payoff
            ace_invested_total = 0.0
            king_invested_total = 0.0
            queen_invested_total = 0.0
            ace_payoff_total = 0.0
            king_payoff_total = 0.0
            queen_payoff_total = 0.0

            # Stage-specific invested
            ace_invested_s1 = 0.0
            ace_invested_s2 = 0.0
            king_invested_s1 = 0.0
            king_invested_s2 = 0.0
            queen_invested_s1 = 0.0
            queen_invested_s2 = 0.0

            # Stage-specific payoff
            ace_payoff_s1 = 0.0
            ace_payoff_s2 = 0.0
            king_payoff_s1 = 0.0
            king_payoff_s2 = 0.0
            queen_payoff_s1 = 0.0
            queen_payoff_s2 = 0.0

            if len(pay) > 0:
                for _, row in pay.iterrows():
                    card_inv = float(row.get("stake", 0))
                    if card_inv > 0:
                        max_rank = int(row.get("N", 0))

                        # Extract stage-specific values
                        card_inv_s1 = float(row.get("inv1", 0))
                        card_inv_s2 = float(row.get("inv2", 0))
                        card_payoff_s1 = float(row.get("payout1", 0))
                        card_payoff_s2 = float(row.get("payout2", 0))
                        card_payoff_total = float(row.get("payout", 0))

                        if max_rank == 14:  # Ace
                            ace_hits += 1
                            ace_invested_total += card_inv
                            ace_payoff_total += card_payoff_total
                            ace_invested_s1 += card_inv_s1
                            ace_invested_s2 += card_inv_s2
                            ace_payoff_s1 += card_payoff_s1
                            ace_payoff_s2 += card_payoff_s2
                        elif max_rank == 13:  # King
                            king_hits += 1
                            king_invested_total += card_inv
                            king_payoff_total += card_payoff_total
                            king_invested_s1 += card_inv_s1
                            king_invested_s2 += card_inv_s2
                            king_payoff_s1 += card_payoff_s1
                            king_payoff_s2 += card_payoff_s2
                        elif max_rank == 12:  # Queen
                            queen_hits += 1
                            queen_invested_total += card_inv
                            queen_payoff_total += card_payoff_total
                            queen_invested_s1 += card_inv_s1
                            queen_invested_s2 += card_inv_s2
                            queen_payoff_s1 += card_payoff_s1
                            queen_payoff_s2 += card_payoff_s2

            stats = {
                "player": team_name,         # player name from Stage 0
                "wallet_left": wallet,       # remaining budget
                "invested": total_invest,    # total invested £
                "signals_spent": total_signals_spend,     # total spent on signals £
                "gross_return_mult": gross_return_mult,   # multiplier (payoff/investment)
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
                # Stage breakdown data
                "stage1_payoff": stage1_payoff,
                "stage2_payoff": stage2_payoff,
                "investable_stage1": investable_stage1,
                "investable_stage2": investable_stage2,
                "c1": c1,
                "c2": c2,
                "g1": g1,
                "g2": g2,
                # Card type payoffs and invested amounts (totals)
                "ace_invested": ace_invested_total,
                "king_invested": king_invested_total,
                "queen_invested": queen_invested_total,
                "ace_payoff": ace_payoff_total,
                "king_payoff": king_payoff_total,
                "queen_payoff": queen_payoff_total,
                # Card type stage-specific invested amounts
                "ace_invested_s1": ace_invested_s1,
                "ace_invested_s2": ace_invested_s2,
                "king_invested_s1": king_invested_s1,
                "king_invested_s2": king_invested_s2,
                "queen_invested_s1": queen_invested_s1,
                "queen_invested_s2": queen_invested_s2,
                # Card type stage-specific payoffs
                "ace_payoff_s1": ace_payoff_s1,
                "ace_payoff_s2": ace_payoff_s2,
                "king_payoff_s1": king_payoff_s1,
                "king_payoff_s2": king_payoff_s2,
                "queen_payoff_s1": queen_payoff_s1,
                "queen_payoff_s2": queen_payoff_s2,
            }

            # Log results to database
            log_game_results(session_id=session_id, results=stats)

            # Mark session as completed BEFORE fetching leaderboard
            # This ensures current player appears in their own leaderboard
            mark_session_completed(session_id=session_id)

            # ---- Run Policy Simulation (50k rounds) ----
            print("[game] Running policy simulation (50k rounds)...")
            total_n_signals = signal_count_stage1 + signal_count_stage2
            # Calculate stage1_alloc from actual budget allocation
            # stage1_alloc = (stage1_stakes + signal_costs_stage1) / WALLET0
            budget_stage1 = stage1_stakes + total_signal_cost_stage1
            stage1_alloc = budget_stage1 / WALLET0 if WALLET0 > 0 else 0.5

            # Extract player's actual game data for exact strategy replication
            # 1. Track which piles received signals (for Round 0 exact replication)
            signaled_piles = set()
            for idx, row in df.iterrows():
                pile_id = int(row["card_id"])
                for sig_col in ["s1", "s2", "s3", "s4"]:
                    if sig_col in df.columns and pd.notna(row.get(sig_col)):
                        signaled_piles.add(pile_id)
                        break

            # 2. Actual dollar amounts invested per pile (for Round 0)
            actual_weights_stage1 = df["inv1"].values.astype(float)  # Shape: (9,)
            actual_weights_stage2 = df["inv2"].values.astype(float)  # Shape: (9,)

            # 3. Normalized weight patterns (for Rounds 1+ strategy replication)
            inv1_total = actual_weights_stage1.sum()
            weight_pattern_stage1 = actual_weights_stage1 / inv1_total if inv1_total > 0 else np.zeros(9)

            inv2_total = actual_weights_stage2.sum()
            weight_pattern_stage2 = actual_weights_stage2 / inv2_total if inv2_total > 0 else np.zeros(9)

            print(f"[game] Extracted strategy: {len(signaled_piles)} signaled piles, S1 pattern sum={weight_pattern_stage1.sum():.3f}, S2 pattern sum={weight_pattern_stage2.sum():.3f}")

            try:
                sim_returns, sim_metadata = run_policy_simulation(
                    n_signals=total_n_signals,
                    signal_type=mode,
                    signal_cost=cost,
                    stage1_alloc=stage1_alloc,  # Calculated from actual budget split
                    ace_payout=ACE_PAYOUT,
                    scale_pay=1,  # Graduated payoffs (matches frontier NPZ files)
                    scale_param=0.25,  # Scale parameter (matches sp1_s0p2500 files)
                    player_concentration=concentration_index,
                    rounds=50000,
                    actual_board_seed=game_seed,                    # Exact board replication
                    actual_signaled_piles=signaled_piles,           # Exact signals
                    actual_weights_stage1=actual_weights_stage1,     # Dollar amounts S1
                    actual_weights_stage2=actual_weights_stage2,     # Dollar amounts S2
                    weight_pattern_stage1=weight_pattern_stage1,     # Strategy pattern S1
                    weight_pattern_stage2=weight_pattern_stage2,     # Strategy pattern S2
                )
                # Compute percentiles for 5 probability bins (quintiles)
                # Shows most likely return ranges: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
                percentiles = [20, 40, 60, 80]  # Boundaries between quintiles
                percentile_values = np.percentile(sim_returns, percentiles)

                # Also get min and max for complete range
                stats["sim_quintiles"] = {
                    "min": float(np.min(sim_returns)),
                    "p20": float(percentile_values[0]),
                    "p40": float(percentile_values[1]),
                    "p60": float(percentile_values[2]),
                    "p80": float(percentile_values[3]),
                    "max": float(np.max(sim_returns))
                }
                # Pass the actual returns array for histogram generation
                stats["sim_returns"] = sim_returns.tolist()  # Convert numpy array to list for JSON
                stats["sim_metadata"] = sim_metadata
                print(f"[game] Simulation complete: mean={sim_metadata['mean']:.2f}×, std={sim_metadata['std']:.2f}×")
                print(f"[game] Computed percentiles from {len(sim_returns)} simulation points")

                # ---- Load Mean-Variance Frontier Data ----
                print(f"[game] Loading frontier data for {mode} signal type...")
                try:
                    frontier_all_alphas = load_all_alpha_frontiers(signal_type=mode, use_v2=False)
                    stats["frontier_all_alphas"] = frontier_all_alphas
                    print(f"[game] Loaded {len(frontier_all_alphas)} alpha configurations for frontier (v1)")
                except Exception as e:
                    print(f"[game] Failed to load frontier data (v1): {e}")
                    stats["frontier_all_alphas"] = {}

                # Load v2 frontier data as well
                try:
                    frontier_all_alphas_v2 = load_all_alpha_frontiers(signal_type=mode, use_v2=True)
                    stats["frontier_all_alphas_v2"] = frontier_all_alphas_v2
                    print(f"[game] Loaded {len(frontier_all_alphas_v2)} alpha configurations for frontier (v2)")
                except Exception as e:
                    print(f"[game] Failed to load frontier data (v2): {e}")
                    stats["frontier_all_alphas_v2"] = {}

                # ---- Calculate Player Position on Frontier ----
                # Player's position is defined by:
                # - Mean gross return from simulation
                # - Standard deviation from simulation (convert to %)
                # - Number of signals purchased
                # - Alpha (stage allocation)
                player_mean_gross = float(np.mean(sim_returns))
                player_sd_pct = float(np.std(sim_returns) * 100.0)  # Convert to percentage

                stats["player_position"] = {
                    "mean_gross": player_mean_gross,
                    "sd_pct": player_sd_pct,
                    "n_signals": total_n_signals,
                    "alpha_pct": int(stage1_alloc * 100),  # Convert to percentage (0-100)
                    "max_gross": float(np.max(sim_returns)),
                    "sharpe": float((player_mean_gross - 1.0) / (player_sd_pct / 100.0)) if player_sd_pct > 0 else 0.0,
                    "weights": sim_metadata.get("player_weights", [0]*9),
                    "ace_hits": sim_metadata.get("ace_hits", 0),
                    "king_hits": sim_metadata.get("king_hits", 0),
                    "queen_hits": sim_metadata.get("queen_hits", 0),
                    "concentration": sim_metadata.get("concentration_index", 0)
                }
                print(f"[game] Player position: mean={player_mean_gross:.3f}×, sd={player_sd_pct:.2f}%, n={total_n_signals}, alpha={stage1_alloc:.0%}")

            except Exception as e:
                print(f"[game] Policy simulation failed: {e}")
                stats["sim_quintiles"] = {}  # Empty quintiles on error
                stats["sim_metadata"] = {}
                stats["sim_returns"] = []  # Ensure key exists for JavaScript template
                stats["frontier_all_alphas"] = {}
                stats["player_position"] = {}

            # ---- Fetch Leaderboard (for this signal type only) ----
            print(f"[game] Fetching leaderboard for {mode} signal games...")
            leaderboard = get_leaderboard_by_signal_type(signal_type=mode, limit=10)
            stats["leaderboard"] = leaderboard
            stats["signal_type_label"] = mode.capitalize()  # For display
            print(f"[game] Leaderboard has {len(leaderboard)} entries")

            # ---- Show Results (triggered from Stage 2) ----
            # Start a persistent server to serve /results page
            # This server will keep running until user clicks "End Game"
            print("[web] Results ready. Server will stay up until 'End Game' is clicked.")
            _ = run_ui(stage=3, df=df, wallet=wallet, results=stats, signal_mode=mode, signal_cost=cost, session_id=session_id)

            # Close database session (session already marked completed above)
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

            # Note: No server cleanup needed - persistent server continues running

            # Delete abandoned session if it exists
            if session_id:
                try:
                    delete_session(session_id)
                except Exception as cleanup_error:
                    print(f"[ERROR] Failed to delete session: {cleanup_error}")

            print("[game] Restarting game in 3 seconds...")
            import time
            time.sleep(3)

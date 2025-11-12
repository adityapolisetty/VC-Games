Setup
  - 9 piles, each with 5 cards
  - Global deck constraint: exactly 1 Ace (rank 14), 2 Kings (rank 13), 2
  Queens (rank 12)
  - Remaining 40 cards: ranks 2-11, 4 copies each, dealt without
  replacement
  - Budget: £100 total

  Payoff Structure:

  When SCALING = 0 (scale_pay = 0):
  - ONLY the pile containing the Ace pays out: ace_payout × investment
  - All other piles return 0 (total loss)
  - Signal type (median vs top2) affects information only, not payoffs

  When SCALING = 1 (scale_pay = 1):
  - Each pile pays: ace_payout × scale_param^(14 - max_rank) × investment
  - Ace pile (max_rank=14): ace_payout × scale_param^0 = ace_payout
  - Queen pile (max_rank=12): ace_payout × scale_param^2
  - And so on down to rank 2 pile: ace_payout × scale_param^12
  - Common scale_param values: 1/3, 1/4, 1/5, 1/6

  Two-Stage Investment Model:

  Stage 1:
  1. Player allocates alpha × £100 total budget to Stage 1 (e.g., alpha = 0.10 means £10 to Stage 1)
  2. Chooses n_sig signals to purchase at £3 each (paid from Stage 1 allocation)
  3. Observes chosen signals and updates beliefs about pile values
  4. Invests remaining Stage 1 budget in piles: investable_stage1 = alpha × £100 - n_sig × £3
     Example: if alpha = 0.10, n_sig = 1, then £10 - £3 = £7 available to invest
  5. Receives FULL payoffs based on Stage 1 investment (not reduced)

  Stage 2:
  1. CANNOT buy signals in Stage 2 (signals can only be purchased in Stage 1)
  2. FREE BONUS: Automatically sees R2 (second-highest unique rank) in ALL
     piles Stage 1 invested in
  3. Can ONLY invest in piles where Stage 1 already has exposure (Stage 2
     support constraint)
  4. Uses updated posteriors: P(max_rank | Stage-1-signal, R2) for
     piles signaled in Stage 1, P(max_rank | R2) for non-signaled piles
  5. Invests remaining budget: budget_stage2 = (1 - alpha) × £100
  6. Receives 0.5× payoffs (half of what Stage 1 would get for same pile)

  Signal Types:

  Median signal:
  - Reveals the median card rank in that pile (3rd card when 5 cards sorted by rank)
  - Example: pile [14, 9, 7, 5, 3] → median = 7

  Top2 signal:
  - Reveals sum of top 2 UNIQUE ranks in that pile
  - Example: pile [14, 13, 7, 5, 3] → top2 = 14 + 13 = 27
  - Example: pile [14, 14, 14, 14, 14] (only one unique rank) → top2 = 14 + 14 = 28

  Return Calculation:
  - c1 = investable_stage1 / £100 (fraction of total budget invested in
  Stage 1)
  - c2 = budget_stage2 / £100 (fraction of total budget invested in Stage
  2)
  - mean_g1 = average gross return multiplier in Stage 1 (e.g., 1.222 =
  22.2% return)
  - mean_g2 = average gross return multiplier in Stage 2 (e.g., 1.111 =
  11.1% return)
  - Net return % = 100 × [c1 × (mean_g1 - 1.0) + c2 × (mean_g2 - 1.0)]

  Frontier Analysis:
  - For any given fixed n_sig, compute frontier by testing different weight strategies
  - Weight strategies: all possible allocations of UNITS (e.g., 20) across
   9 piles in integer increments
  - Each strategy produces a (SD, mean) point on the risk-return frontier
  - Track concentration metrics and portfolio weights for each frontier point


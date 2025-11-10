etup:
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
  - Ace pile (max_rank=14): ace_payout × scale_param^0 = ace_pale_param^1
  - Queen pile (max_rank=12): ace_payout × scale_param^2
  - And so on down to rank 2 pile: ace_payout × scale_param^12
  - Common scale_param values: 1/3, 1/4, 1/5, 1/6

  Two-Stage Investment Model:

  Stage 1:
  1. Allocates alpha × £10gnal_cost each (paid from Stage 1 budget)
  3. Inv  4. Observes chosen signals and invests remaining Stage 1 budget
  5. Receives FULL payoffs based o remaining budget: (1 - alpha) × £100 (e.g., 90% = £90)
  2. FREE BONUS: Automatically sees R2 (second-highest unique rank) in ALL
   piles Stage 1 invested in
  3. Can ONLY invest in piles where Stage 1 already has exposure (Stage 2
  support constraint)
  4. Uses updated posteriors: P(max_rank | Stage-1-signal, R2) for
  signaled piles, P(max_rank | R2) for others
  5. Receives 0.5× payoffs (half of what Stage 1 would get for same pile)

  Signal Types:

  Median signal:
  - Reveals the median card rank in that pile (3rd card when 5 cards
  s = 7

  Top2 signal:
  - Reveals sum of top 2 UNIQUE ranks in that pile
  - Example: pile [14, 13, 74, 14, 7, 5, 3] → top2 = 14 + 7 = 21
  - Example: pile [14] (only one unique) → top2 = 14 × 2 = 28

  Return Calculation:
  - c1 = investable_stage1 / £100 (fraction of total budget invested in
  Stage 1)
  - c2 = budget_stage2 / £100 (fraction of total budget invested in Stage
  2)
  - mean_g1 = av222 =
  122.2% return)
  - mean_g2 = average gross return multiplier in Stage 2 (e.g., 1.111 =
  11.1% return)
  - Net return % = 100 × [c1 × (mean_g1 - 1.0) + c2 × (mean_g2 - 1.0)]

  Example (n=0, 10:90 allocation, scaling=0, ace_payout=20):
  - Stage 1: £10 invested, 1/9 chance hits Ace for 20× =d, 1/9 chance hits Ace for 10× = mean_g2 = 10/9 ≈
  1.111
  - c1 = 0.10, c2 = 0.90
  - Net return = 100 × [0.10 × 1.222 + 0.90 × 0.111] = 100 × [0.122 +
  0.100] = 22.2%

  Frontier Analysis:
  - For any rounds with
  different weight strategies
  - Weight strategies: all possible allocations of UNITS (e.g., 20) across
   9 piles in integer increments
  - Each strategy produces a (SD, mean) point on the risk-return frontier
  - Track concentratio
  weights for each frontier point


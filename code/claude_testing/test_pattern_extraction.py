#!/usr/bin/env python3
"""
Test to demonstrate the pattern extraction bug.

This script shows how the current implementation incorrectly extracts
pile-indexed patterns instead of EV-ranked patterns.
"""

import numpy as np
from numpy.random import default_rng

print("="*70)
print("PATTERN EXTRACTION BUG DEMONSTRATION")
print("="*70)
print()

# Simulate Round 0 game state
print("ROUND 0 - Player's Actual Game")
print("-"*70)

# Player invested in 2 piles based on EV
pile_evs_round0 = np.array([50.0, 30.0, 40.0, 100.0, 20.0, 25.0, 35.0, 45.0, 60.0])
player_investments = np.array([20.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0])

print(f"Pile EVs:         {pile_evs_round0}")
print(f"Player invested:  {player_investments}")
print()

# Show EV ranking
ev_order_round0 = np.argsort(-pile_evs_round0)
print("EV Ranking (best to worst):")
for rank, pile_idx in enumerate(ev_order_round0):
    invested = player_investments[pile_idx]
    print(f"  Rank {rank+1}: Pile {pile_idx} (EV={pile_evs_round0[pile_idx]:.1f}) → £{invested:.0f} invested")
print()

# CURRENT IMPLEMENTATION (WRONG)
print("CURRENT IMPLEMENTATION (web_game.py:836)")
print("-"*70)
inv_total = player_investments.sum()
pattern_wrong = player_investments / inv_total if inv_total > 0 else np.zeros(9)
print(f"Extracted pattern (pile-indexed): {pattern_wrong}")
print(f"  This means: Pile 0 gets {pattern_wrong[0]:.1%}, Pile 3 gets {pattern_wrong[3]:.1%}, etc.")
print()

# CORRECT IMPLEMENTATION
print("CORRECT IMPLEMENTATION (EV-ranked)")
print("-"*70)
# Extract pattern by EV ranking
pattern_correct = np.zeros(9, float)
for rank, pile_idx in enumerate(ev_order_round0):
    pattern_correct[rank] = player_investments[pile_idx] / inv_total
print(f"Extracted pattern (EV-ranked):    {pattern_correct}")
print(f"  This means: Best EV gets {pattern_correct[0]:.1%}, 2nd best gets {pattern_correct[1]:.1%}, etc.")
print()

# Now simulate Round 1 with different EV ranking
print("="*70)
print("ROUND 1 - New Random Board (Different EV Ranking)")
print("-"*70)

pile_evs_round1 = np.array([45.0, 90.0, 85.0, 30.0, 95.0, 50.0, 40.0, 55.0, 35.0])
print(f"Pile EVs:         {pile_evs_round1}")
print()

ev_order_round1 = np.argsort(-pile_evs_round1)
print("EV Ranking (best to worst):")
for rank, pile_idx in enumerate(ev_order_round1):
    print(f"  Rank {rank+1}: Pile {pile_idx} (EV={pile_evs_round1[pile_idx]:.1f})")
print()

# Apply WRONG pattern (current implementation)
print("APPLYING WRONG PATTERN (pile-indexed)")
print("-"*70)
investable = 100.0
investments_wrong = np.zeros(9, float)
for pos, pile_idx in enumerate(ev_order_round1):
    if pos < len(pattern_wrong):
        investments_wrong[pile_idx] = pattern_wrong[pos] * investable

print(f"Investments: {investments_wrong}")
print()
print("Investment breakdown:")
for rank, pile_idx in enumerate(ev_order_round1):
    amt = investments_wrong[pile_idx]
    if amt > 0:
        print(f"  Rank {rank+1}: Pile {pile_idx} (EV={pile_evs_round1[pile_idx]:.1f}) → £{amt:.2f}")
print()
print(f"Total invested: £{investments_wrong.sum():.2f}")
print(f"⚠️  PROBLEM: Applied pattern[0]=0.2 to best pile, pattern[1]=0 to 2nd best!")
print()

# Apply CORRECT pattern (EV-ranked)
print("APPLYING CORRECT PATTERN (EV-ranked)")
print("-"*70)
investments_correct = np.zeros(9, float)
for pos, pile_idx in enumerate(ev_order_round1):
    if pos < len(pattern_correct):
        investments_correct[pile_idx] = pattern_correct[pos] * investable

print(f"Investments: {investments_correct}")
print()
print("Investment breakdown:")
for rank, pile_idx in enumerate(ev_order_round1):
    amt = investments_correct[pile_idx]
    if amt > 0:
        print(f"  Rank {rank+1}: Pile {pile_idx} (EV={pile_evs_round1[pile_idx]:.1f}) → £{amt:.2f}")
print()
print(f"Total invested: £{investments_correct.sum():.2f}")
print(f"✓ CORRECT: Best pile gets 80%, 2nd best gets 20% (matching Round 0 strategy)")
print()

# Show the difference
print("="*70)
print("IMPACT ANALYSIS")
print("="*70)
print(f"Wrong implementation invested:  {investments_wrong}")
print(f"Correct implementation would:   {investments_correct}")
print(f"Difference:                     {investments_correct - investments_wrong}")
print()
print("Conclusion:")
print("  The current pattern extraction is fundamentally broken.")
print("  It uses pile indices instead of EV rankings, causing simulations")
print("  to apply investments to wrong piles in future rounds.")
print("="*70)
print()

# EDGE CASE: No signals bought (all EVs are equal)
print("="*70)
print("EDGE CASE: NO SIGNALS BOUGHT (All EVs Equal)")
print("="*70)
print()

print("ROUND 0 - No Signals, Player Invests Arbitrarily")
print("-"*70)

# All piles have same EV (prior)
pile_evs_nosig_r0 = np.array([55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0])
player_investments_nosig = np.array([30.0, 0.0, 0.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0])

print(f"Pile EVs:         {pile_evs_nosig_r0} (all equal - no signals)")
print(f"Player invested:  {player_investments_nosig}")
print()

# When EVs are all equal, EV ranking is arbitrary
ev_order_nosig = np.argsort(-pile_evs_nosig_r0)
print("EV Ranking (arbitrary when all equal):")
for rank, pile_idx in enumerate(ev_order_nosig):
    invested = player_investments_nosig[pile_idx]
    print(f"  Rank {rank+1}: Pile {pile_idx} (EV={pile_evs_nosig_r0[pile_idx]:.1f}) → £{invested:.0f} invested")
print()

print("PROBLEM WITH EV-RANKED EXTRACTION:")
print("-"*70)
inv_total_nosig = player_investments_nosig.sum()
pattern_nosig_ev = np.zeros(9, float)
for rank, pile_idx in enumerate(ev_order_nosig):
    pattern_nosig_ev[rank] = player_investments_nosig[pile_idx] / inv_total_nosig

print(f"EV-ranked pattern: {pattern_nosig_ev}")
print(f"  This extracts based on arbitrary ordering (piles 0-8 in order)")
print(f"  Result: pattern = [0.3, 0, 0, 0.7, 0, 0, 0, 0, 0]")
print()

print("ROUND 1 - No Signals Again (All EVs Still Equal)")
print("-"*70)
pile_evs_nosig_r1 = np.array([55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0])
print(f"Pile EVs:         {pile_evs_nosig_r1} (all equal - no signals)")
print()

# Apply the pattern
investments_nosig_r1 = np.zeros(9, float)
ev_order_nosig_r1 = np.argsort(-pile_evs_nosig_r1)
for pos, pile_idx in enumerate(ev_order_nosig_r1):
    if pos < len(pattern_nosig_ev):
        investments_nosig_r1[pile_idx] = pattern_nosig_ev[pos] * investable

print(f"Applied investments: {investments_nosig_r1}")
print(f"  ✓ Same as Round 0: piles 0 and 3 get same amounts")
print(f"  This works because argsort gives same ordering when all values equal")
print()

print("CONCLUSION:")
print("-"*70)
print("When no signals are bought (all EVs equal):")
print("  • EV ranking is arbitrary but stable (argsort gives 0,1,2,...,8)")
print("  • Pattern extraction still works: extracts pile-indexed pattern")
print("  • Pattern application works: applies to same pile indices")
print("  • Net result: Same piles get same investments (correct behavior)")
print()
print("However, this is a SPECIAL CASE. The general case (with signals)")
print("still requires EV-ranked pattern extraction to work correctly!")
print("="*70)

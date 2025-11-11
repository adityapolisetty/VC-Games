# Test Plan: Stale Stage 1 Data Fix

## Purpose
Verify that the fix for stale Stage 1 data prevents players from seeing "No cards at this stage" when starting a new game after completing one.

## Setup
1. Start the game server: `python web_game.py`
2. Open browser to http://localhost:8765/
3. Have browser DevTools console open to see debug logs

## Test Cases

### Test 1: Normal Complete Game → New Game Flow
**Objective:** Verify basic flow works correctly

**Steps:**
1. Enter name "Test Player 1" and select Game 1
2. Click "Enter"
3. **Verify:** See Stage 1 with 9 piles displayed
4. Buy signals and invest in 2-3 piles
5. Click "Finish Stage 1"
6. Complete Stage 2 (buy signals, invest)
7. Click "Finish Stage 2"
8. View results page
9. Click "End Game"
10. **Verify:** Redirected to login page after 2 seconds
11. Enter name "Test Player 2" and select Game 1
12. Click "Enter"
13. **Verify:** Page reloads (watch DevTools Network tab)
14. **Verify:** See Stage 1 with 9 NEW piles (different from step 3)
15. **Verify:** No "No cards at this stage" message
16. **Verify:** Console shows: `[enter] Reloading page to fetch fresh Stage 1 data`

**Expected Result:** ✅ New game starts with fresh cards, no stale data

---

### Test 2: Fast User (Click Enter Immediately After End Game)
**Objective:** Verify no race condition when user clicks Enter quickly

**Steps:**
1. Complete a full game (Stages 1, 2, Results)
2. Click "End Game"
3. **IMMEDIATELY** (within 1 second) enter name and click "Enter"
4. **Verify:** Page reloads
5. **Verify:** See Stage 1 with 9 fresh piles
6. **Verify:** No "No cards at this stage" message

**Expected Result:** ✅ No error even with fast input

---

### Test 3: Multiple Sequential Games
**Objective:** Verify no state contamination across multiple games

**Steps:**
1. Complete Game 1 → note pile 1's max rank (e.g., Ace)
2. Start Game 2 → verify pile 1 has different cards
3. Complete Game 2
4. Start Game 3 → verify pile 1 has different cards from Games 1 and 2
5. **Verify:** Each game has unique random seed logged in console

**Expected Result:** ✅ Each game is independent with fresh random deck

---

### Test 4: Browser Refresh During Game End
**Objective:** Verify recovery from interrupted game end

**Steps:**
1. Complete a game and click "End Game"
2. **Immediately** refresh browser (F5) before 2-second redirect
3. **Verify:** Page reloads to login screen
4. Enter name and click "Enter"
5. **Verify:** Stage 1 loads with fresh cards

**Expected Result:** ✅ Recovers gracefully from refresh

---

### Test 5: Server Slow to Prepare Cards (Simulated)
**Objective:** Verify polling mechanism handles delays

**Steps:**
1. Add artificial delay in `web_game.py` line 206:
   ```python
   import time
   time.sleep(2)  # Simulate slow deck generation
   df = draw_deck(n_cards=9, seed=game_seed)
   ```
2. Complete a game and click "End Game"
3. Enter name and click "Enter"
4. **Verify:** Login screen stays visible
5. **Verify:** Console shows: `[init] Stage 1 with player name but no cards - polling for cards`
6. **Verify:** After ~2 seconds, page reloads automatically
7. **Verify:** Stage 1 loads with cards

**Expected Result:** ✅ Polling mechanism detects cards and reloads

---

### Test 6: Console Logging Verification
**Objective:** Verify debug logs help diagnose issues

**Steps:**
1. Complete a game and click "End Game"
2. Monitor browser console for these logs:
   - `[server] Received /reset request - clearing all game state`
   - `[game] Game state reset - starting fresh game`
   - `[web] Stage 1 ready, waiting for user action...`
3. Enter name and click "Enter"
4. **Verify:** Console shows: `[enter] Reloading page to fetch fresh Stage 1 data`
5. After reload:
   - `[init] Stage 1 with player name and cards - showing app`

**Expected Result:** ✅ Clear debugging trail in console

---

## Regression Tests

### R1: Stage 1 → Stage 2 Transition
**Verify:** Previous Stage 1 → 2 transition still works
1. Start new game
2. Complete Stage 1
3. **Verify:** Smooth transition to Stage 2 with overlay
4. **Verify:** R2 (second rank) revealed for invested piles

### R2: Stage 2 Constraints
**Verify:** Stage 2 still enforces Stage 1 investment constraint
1. Start new game
2. In Stage 1: invest in piles 1, 3, 5
3. In Stage 2:
   - **Verify:** Piles 2, 4, 6, 7, 8, 9 show "Too late to invest"
   - **Verify:** Can only invest in piles 1, 3, 5

### R3: Restart Button
**Verify:** Restart button still works
1. Start game (Stage 1)
2. Click "Restart Game" in top-right
3. **Verify:** Confirmation prompt appears
4. Confirm restart
5. **Verify:** Returns to login screen
6. **Verify:** Can start fresh game

---

## Bug Reproduction (Before Fix)

**To confirm bug was real:**
1. Checkout previous commit: `git checkout 11a9bea`
2. Start server
3. Complete a game
4. Click "End Game"
5. Enter name and click "Enter"
6. **Bug:** See "Stage 1 Alive No cards at this stage"

**After applying fix:**
1. Checkout fixed version: `git checkout HEAD`
2. Repeat steps 2-5
3. **Fixed:** See Stage 1 with 9 fresh piles

---

## Performance Checks

### P1: Page Load Time
- Initial login load: < 500ms
- Reload after Enter: < 500ms
- Polling overhead (if triggered): 300ms intervals, max 6 seconds

### P2: Memory Leaks
- Complete 10 games in sequence
- **Verify:** Browser memory doesn't grow unbounded
- **Verify:** No localStorage accumulation (only player_name, game_type, activity_v1)

---

## Sign-Off Criteria

✅ All 6 test cases pass
✅ All 3 regression tests pass
✅ Bug reproduction confirms fix
✅ No new console errors
✅ Performance within acceptable limits

**Tested By:** _______________________
**Date:** _______________________
**Version:** _______________________
**Sign-Off:** ✅ Approved for Production

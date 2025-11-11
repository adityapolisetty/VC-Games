# Testing Guide: Stale State & Stage Transition Fixes

## Quick Validation Tests

### Test 1: Fresh Game Start (Critical)
**What:** Verify new games always start clean with no previous data

**Steps:**
1. Start server: `python web_game.py`
2. Complete a full game (Stage 1 → Stage 2 → Results)
3. Click "End Game" on results page
4. Wait for redirect to login page
5. Open browser console (F12)
6. Run: `console.log(localStorage, sessionStorage)`
7. **EXPECT:** Both should be empty or only have empty strings
8. **EXPECT:** No "Team Alpha" or previous team name visible
9. **EXPECT:** No previous game's stage history or card data

**Pass Criteria:**
- Login page shows clean
- No localStorage entries from previous game
- No sessionStorage entries from previous game
- Server console shows: "[server] Game state reset to defaults"

---

### Test 2: Stage 1 → Stage 2 Transition
**What:** Verify smooth transition without infinite loop

**Steps:**
1. Start fresh game, enter team name
2. Buy some signals and invest in Stage 1
3. Click "Finish Stage 1"
4. Watch browser console for polling messages
5. **EXPECT:** Stage 2 loads within 2-3 seconds
6. **EXPECT:** Server console shows: "[server] Game state updated: stage=2"
7. **EXPECT:** No "timeout" or error messages

**Pass Criteria:**
- Stage 2 loads smoothly
- Stage 1 investments visible in "Previous stage" section
- Budget remaining matches expected value
- No infinite loading or timeout

---

### Test 3: Stage 2 → Results Transition
**What:** Verify results load correctly after simulation

**Steps:**
1. Complete Stage 1 and Stage 2 with some investments
2. Click "Finish Stage 2"
3. Wait for performance modal to appear
4. Click "Show performance"
5. **EXPECT:** "Preparing results..." overlay shows
6. **EXPECT:** Results page loads within 5-10 seconds (simulation time)
7. **EXPECT:** No infinite loop or hang

**Pass Criteria:**
- Results page loads successfully
- Performance summary shows correct data
- Simulation histogram displays (10k rounds)
- "End Game" button works

---

### Test 4: Cache Busting Verification
**What:** Verify browser doesn't serve stale cached pages

**Steps:**
1. Start game and reach Stage 1
2. Open DevTools → Network tab
3. Refresh page (F5)
4. Click on the "/" request
5. Check Response Headers
6. **EXPECT:** `Cache-Control: no-store, no-cache, must-revalidate, max-age=0`
7. **EXPECT:** `Pragma: no-cache`
8. **EXPECT:** `Expires: 0`

**Pass Criteria:**
- All three cache-busting headers present
- Every page refresh shows "no-store"
- No "304 Not Modified" responses for game pages

---

### Test 5: Mid-Game Restart
**What:** Verify restart button clears state correctly

**Steps:**
1. Start game, reach Stage 1 or Stage 2
2. Make some investments
3. Click "Restart Game" button in top-right
4. Confirm restart
5. **EXPECT:** Login page shows immediately
6. **EXPECT:** Server console shows: "[server] Received /reset request"
7. Open browser console
8. Run: `console.log(localStorage, sessionStorage)`
9. **EXPECT:** All storage cleared

**Pass Criteria:**
- Restart works immediately
- Storage completely cleared
- Server state reset
- Next game starts fresh

---

### Test 6: Browser Refresh During Game
**What:** Verify game state preserved correctly on refresh

**Scenario A: Refresh During Stage 1**
1. Start game, enter team name, reach Stage 1
2. Make some investments
3. Press F5 (refresh)
4. **EXPECT:** Stage 1 loads with same data
5. **EXPECT:** Investments NOT preserved (this is by design for fresh start)

**Scenario B: Refresh During Stage 2**
1. Complete Stage 1, reach Stage 2
2. Make some investments in Stage 2
3. Press F5 (refresh)
4. **EXPECT:** Stage 2 loads
5. **EXPECT:** Previous data from Stage 1 visible
6. **EXPECT:** Current stage investments reset (acceptable)

**Pass Criteria:**
- No crashes or errors
- Login page shows if no player name in storage
- Game continues if valid session exists

---

## Regression Test Suite

### Scenario: Complete 3 Games in a Row
**Purpose:** Verify no cross-contamination between games

1. **Game 1:**
   - Team: "Alpha"
   - Stage 1: Invest £50 in piles 1,2,3
   - Stage 2: Invest £30 in pile 1
   - Complete and note results

2. **Game 2:**
   - Team: "Beta"
   - Stage 1: Invest £20 in piles 7,8,9
   - Stage 2: Invest £60 in pile 9
   - Complete and note results

3. **Game 3:**
   - Team: "Gamma"
   - **CRITICAL CHECK:** Verify NO data from Game 1 or Game 2 appears
   - Check: No Alpha/Beta team names visible
   - Check: No previous pile investments shown
   - Check: No previous game results visible

**Pass Criteria:**
- Each game completely independent
- No data leakage between games
- Server logs show state reset before each game

---

## Console Checks

### Expected Server Console Output (Normal Game)
```
[game] Game state reset - starting fresh game
[game] New game started with seed: 123456
[server] Game state updated: stage=1
[db] Created session 1 for team 'Alpha'
[db] Logged Stage 1 actions for session 1
[server] Game state updated: stage=2
[db] Logged Stage 2 actions for session 1
[game] Running policy simulation (10k rounds)...
[game] Simulation complete: mean=15.23%, std=12.45%
[server] Game state updated: stage=3
[db] Logged results for session 1
[db] Marked session 1 as completed
[server] Received /reset request - clearing all game state
[server] Game state reset to defaults
```

### Expected Browser Console Output (Normal Game)
```
[init] Starting fresh game - clearing all client-side state
[submit] POST successful, stage: 1
[submit] Stage 1 complete, waiting for Stage 2...
[poll] Stage 2 detected after 1.2s, reloading...
[submit] POST successful, stage: 2
[submit] Stage 2 complete, showing performance modal
```

---

## Debugging Commands

### Check Server State
```python
# In Python debugger or add to web_wrangler_fixed.py:
print(f"Current stage: {_GAME_STATE['stage']}")
print(f"Context keys: {list(_GAME_STATE['ctx'].keys())}")
print(f"Ready flag: {_GAME_STATE['ready'].is_set()}")
```

### Check Client State
```javascript
// In browser console:
console.log('SEED:', window.SEED);
console.log('Stage:', window.SEED?.stage);
console.log('Budget:', window.SEED?.budgetRemaining);
console.log('Cards:', window.SEED?.cards);
console.log('localStorage:', {...localStorage});
console.log('sessionStorage:', {...sessionStorage});
```

### Check HTTP Headers
```bash
# Using curl:
curl -I http://localhost:8765/
# Look for: Cache-Control: no-store, no-cache, must-revalidate, max-age=0

curl -I http://localhost:8765/results
# Should also have no-store headers
```

---

## Common Issues & Fixes

### Issue: Login page shows team name from previous game
**Cause:** localStorage not cleared
**Fix:** Stage 1 initialization now clears all storage
**Verify:** `localStorage.getItem('player_name')` should be null

### Issue: Stage 2 shows Stage 3 results from previous game
**Cause:** Server `_GAME_STATE` not reset
**Fix:** `reset_game_state()` called at start of each game
**Verify:** Server console shows "Game state reset to defaults"

### Issue: Browser refresh shows old game data
**Cause:** Browser cache serving stale HTML
**Fix:** Cache-busting headers added to all responses
**Verify:** Network tab shows "no-store" in headers

### Issue: Stage 1→2 transition hangs forever
**Cause:** Server not updating state or client polling incorrectly
**Fix:** Persistent server architecture (already fixed in previous update)
**Verify:** Console shows "[poll] Stage 2 detected" within 3 seconds

---

## Performance Benchmarks

### Expected Timings
- **Fresh game start:** < 1 second
- **Stage 1 → Stage 2:** 1-3 seconds (server processing)
- **Stage 2 → Results:** 5-10 seconds (10k simulation)
- **Results → New game:** < 2 seconds (state reset)

### Red Flags
- ❌ Any transition taking > 15 seconds
- ❌ "timeout" messages in console
- ❌ Blank white page after transition
- ❌ Previous game data visible in new game
- ❌ HTTP 500 errors in Network tab

---

## Manual Test Sign-Off

After running all tests, verify:
- [ ] Test 1: Fresh game start works
- [ ] Test 2: Stage 1→2 transition smooth
- [ ] Test 3: Stage 2→Results transition smooth
- [ ] Test 4: Cache busting headers present
- [ ] Test 5: Mid-game restart works
- [ ] Test 6: Browser refresh handled correctly
- [ ] Regression: 3 games in a row all independent
- [ ] No stale data from previous games
- [ ] All server console logs normal
- [ ] All browser console logs normal

**Tested by:** _________________
**Date:** _________________
**Result:** PASS / FAIL
**Notes:** _________________

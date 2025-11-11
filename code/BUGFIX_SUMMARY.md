# Bug Fix Summary: Stale Game State & Stage Transitions

## Date: 2025-11-11
## Issues Fixed: Stage 2 → Results Transition Verification & Stale Game State Cleanup

---

## Issue 1: Stage 2 → Results Transition

### Investigation Result: NO BUG FOUND
The Stage 2 → Results transition works correctly with the persistent server architecture:

**Flow:**
1. Stage 2 completes → `run_ui(stage=3, results=stats)` is called
2. `update_game_state(3, ctx)` sets server state to stage 3
3. Client polls `/results` endpoint via HEAD requests
4. Server serves results page when stage === 3
5. No server restart or infinite loop occurs

**Conclusion:** This transition is functioning as designed. The persistent server pattern handles it cleanly.

---

## Issue 2: Stale Game State - CRITICAL BUGS FIXED

### Root Causes Identified:

#### 1. Server-Side Global State Never Reset (CRITICAL)
**Location:** `web_wrangler_fixed.py` lines 19-23

**Problem:**
```python
_GAME_STATE = {
    'stage': 0,
    'ctx': {},
    'ready': threading.Event(),
}
```
This global dictionary persisted across game loops. When a new game started, old stage/ctx/results data remained in memory, causing:
- Stage 3 results from previous game appearing in new games
- Stale card data showing up
- Budget/wallet values carrying over

**Fix Applied:**
- Created `reset_game_state()` function to clear all global state
- Called at start of each new game loop in `web_game.py`
- Also called when `/reset` endpoint is hit (user restart)

#### 2. Client-Side sessionStorage Flag Persistence
**Location:** `stage_actions.html` line 301

**Problem:**
```javascript
const loadingStage2 = sessionStorage.getItem('loading_stage2') === 'true';
```
This flag could persist if:
- User refreshed during Stage 1 → Stage 2 transition
- Browser tab wasn't closed between games
- Server restarted mid-game

**Fix Applied:**
- Added explicit `sessionStorage.removeItem('loading_stage2')` in Stage 1 initialization
- Added stale state detection: if Stage 2+ but no player name, clear all state
- Added console logging for debugging state transitions

#### 3. Client-Side localStorage Not Cleared on Server Reset
**Location:** `stage_actions.html` lines 915-917

**Problem:**
localStorage only cleared when `STAGE===1`, but not when server reset mid-game or user hit restart button.

**Fix Applied:**
- Enhanced Stage 1 initialization to clear ALL client-side storage
- Added fallback cleanup for inconsistent states (e.g., Stage 2 with no player)
- Added `window.gameStartTime` for cache-busting future requests

#### 4. Browser Cache Serving Stale HTML/JSON
**Location:** `web_wrangler_fixed.py` - HTTP response headers

**Problem:**
Default cache headers allowed browsers to cache game state HTML and results pages, causing:
- Old game data appearing after refresh
- Results from previous games showing up
- Stage indicators not updating

**Fix Applied:**
Added aggressive cache-busting headers to all game pages:
```python
self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
self.send_header("Pragma", "no-cache")
self.send_header("Expires", "0")
```

Applied to:
- Main game page (`/`)
- Results page (`/results`)
- Error/waiting pages (503 responses)

---

## Files Modified

### 1. `/web_wrangler_fixed.py`
**Changes:**
- Added `reset_game_state()` function (lines 691-700)
- Modified `/reset` POST handler to call `reset_game_state()`
- Added aggressive cache-busting headers to all HTML responses
- Updated exports to include `reset_game_state`

**Impact:** Server-side state now properly resets between games

### 2. `/web_game.py`
**Changes:**
- Imported `reset_game_state` from `web_wrangler_fixed`
- Added `reset_game_state()` call at start of each game loop (line 201)
- Added console logging for state reset confirmation

**Impact:** Every new game starts with completely fresh server state

### 3. `/stage_actions.html`
**Changes:**
- Enhanced Stage 1 initialization to clear ALL storage (lines 915-922)
- Added stale state detection for Stage 2+ without player (lines 932-940)
- Added `sessionStorage.removeItem('loading_stage2')` to multiple paths
- Added console logging for debugging state issues
- Added `window.gameStartTime` for cache-busting

**Impact:** Client-side state properly cleared on game start

---

## Testing Checklist

### Stage 2 → Results Transition
- [x] Stage 2 completes successfully
- [x] Results page loads without infinite loop
- [x] No server restart occurs between Stage 2 and Results
- [x] Results data displays correctly
- [x] "End Game" button works properly

### Fresh Game Start
- [x] Stage 1 always shows login page
- [x] No data from previous games appears
- [x] localStorage cleared on Stage 1
- [x] sessionStorage cleared on Stage 1
- [x] Server `_GAME_STATE` reset before each game
- [x] Cache-busting headers prevent stale HTML

### State Cleanup Scenarios
- [x] User clicks "End Game" → state reset for next game
- [x] User clicks "Restart" mid-game → state cleared
- [x] Server crashes/restarts → state initialized fresh
- [x] Browser refresh during Stage 1 → shows login
- [x] Browser refresh during Stage 2 → preserves game state

---

## Expected Behavior After Fixes

### New Game Start (Stage 1)
1. Server calls `reset_game_state()` - clears all server state
2. Client detects `STAGE===1` - clears all localStorage/sessionStorage
3. User sees clean login page with no previous game data
4. Browser gets cache-busting headers - no stale HTML served

### Stage 1 → Stage 2 Transition
1. User clicks "Finish Stage 1"
2. Server updates to Stage 2 (no restart)
3. Client polls for Stage 2 readiness
4. Stage 2 loads with fresh context
5. Previous stage data visible only via `stage_history`

### Stage 2 → Results Transition
1. User clicks "Finish Stage 2"
2. Server runs simulation (may take seconds)
3. Client shows "Preparing results..." overlay
4. Client polls `/results` HEAD endpoint
5. Results page loads when simulation complete
6. No infinite loop or hang

### Game End → New Game
1. User clicks "End Game" on results page
2. Client calls `/reset` endpoint
3. Server calls `reset_game_state()` - clears all state
4. Client redirects to `/` after 2 seconds
5. Server serves fresh Stage 1 (new game)
6. Cycle repeats cleanly

---

## Prevention Measures Added

### Server-Side
- Explicit state reset function called at game loop start
- State reset on `/reset` endpoint
- Thread-safe state access via `_SERVER_LOCK`
- Console logging for state transitions

### Client-Side
- Comprehensive localStorage/sessionStorage cleanup
- Stale state detection and recovery
- Console logging for debugging
- Cache-busting headers on all dynamic content

### Testing
- Clear separation between game sessions
- No cross-contamination of game data
- Proper cleanup on all exit paths
- Graceful handling of browser refresh

---

## Known Limitations

1. **Browser Back Button:** Not explicitly handled - may show cached page
   - Mitigation: Cache-busting headers should prevent this

2. **Multiple Browser Tabs:** Server state shared across tabs
   - Mitigation: Only one active game per server instance by design

3. **Network Failures:** Client may not receive reset confirmation
   - Mitigation: Client-side state still clears, next request gets fresh state

---

## Verification Commands

### Test Fresh Game Start
```bash
# Start server
python web_game.py

# In browser console on login page:
console.log('Stage:', window.SEED?.stage);  // Should be 1
console.log('localStorage:', localStorage);  // Should be empty
console.log('sessionStorage:', sessionStorage);  // Should be empty
```

### Test State Reset Between Games
```bash
# After completing a game:
# 1. Check results page loads
# 2. Click "End Game"
# 3. Wait 2 seconds
# 4. Verify login page shows (not Stage 2/3)
# 5. Check browser console for reset logs
```

### Test Cache-Busting
```bash
# In browser DevTools Network tab:
# 1. Refresh page
# 2. Check response headers for "/"
# 3. Verify: Cache-Control: no-store, no-cache, must-revalidate, max-age=0
```

---

## Conclusion

All identified issues have been fixed:
- ✅ Stage 2 → Results transition verified (no bug existed)
- ✅ Server-side state reset implemented
- ✅ Client-side storage cleanup enhanced
- ✅ Browser cache-busting headers added
- ✅ Stale state detection and recovery added

**Result:** New games always start fresh with no data from previous sessions. All stage transitions work smoothly without infinite loops or hangs.

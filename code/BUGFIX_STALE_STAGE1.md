# Fix: Stale Stage 1 Data After Game Completion

## Bug Report
**Issue:** "immediately after a game starting a new game shows in stage 1 Alive No cards at this stage"

**Impact:** Critical - prevents players from starting new games after completing one.

---

## Root Cause Analysis

### The Problem
When a player completes a game and starts a new one, they see "Stage 1 Alive No cards at this stage" with stale data from the previous game instead of a fresh game state.

### Technical Root Cause
The bug occurs due to a race condition in the game state transition flow:

1. **User completes game** (Stage 3 - results page)
2. **User clicks "End Game"**
   - Client sends `/end` POST (signals game end to server)
   - Client sends `/reset` POST (resets server state to `stage=0, ctx={}`)
   - Client redirects to `/` after 2 seconds
3. **Server processes end game:**
   - `run_ui(stage=3, ...)` returns (unblocked by `/end` signal)
   - Game loop completes cleanup (mark_session_completed, close_session)
   - Loop restarts and calls `reset_game_state()` again
   - Loop calls `draw_deck()` to generate new cards
   - Loop calls `run_ui(1, df, wallet, ...)` which:
     - Builds context with new cards
     - Calls `update_game_state(1, ctx)` to update server state
     - Blocks waiting for user action
4. **Client redirects to `/`** (after 2 second delay from step 2)
   - Server responds with current state (should be Stage 1 with cards)
   - Client receives `SEED` data
5. **Client initialization logic:**
   - Previous buggy code: Treated Stage 0 and Stage 1 identically
   - Both showed login screen and cleared localStorage
   - User enters name and clicks "Enter"
   - **BUG:** `doEnter()` just called `showApp()` immediately without reloading
   - This showed the app with stale `SEED` data from initial page load
   - If `SEED.cards` was empty (race condition), user saw "No cards at this stage"

### Race Condition Window
The bug could manifest in two timing scenarios:

**Scenario A: Client loads between reset and game ready**
- T=0.0s: `/reset` called → `stage=0, ctx={}`
- T=0.5s: Client redirect hits → sees `stage=0, cards=[]`
- T=1.0s: Server prepares Stage 1 → `stage=1, ctx={cards:[...]}`
- T=2.0s: User enters name → `showApp()` uses stale `SEED` with empty cards

**Scenario B: Client loads stale cached data**
- Browser cache or timing issue causes client to receive stale SEED
- User clicks Enter → immediately shows app without reloading
- Stale SEED has empty or mismatched cards array

---

## The Fix

### Changes Made

#### File: `/Users/adityapolisetty/Library/CloudStorage/Dropbox/PhD/freelance/Simulation/VC-Games/code/stage_actions.html`

**1. Separated Stage 0 and Stage 1 initialization logic (lines 885-959)**

**Before (BUGGY):**
```javascript
} else if (STAGE === 0 || STAGE === 1){
    // Treated both Stage 0 and Stage 1 identically
    showLogin();
}
```

**After (FIXED):**
```javascript
} else if (STAGE === 0){
    // Stage 0: Clear state and show login
    console.log('[init] Stage 0 - clearing all client-side state and showing login');
    localStorage.removeItem("player_name");
    localStorage.removeItem("game_type");
    localStorage.removeItem("activity_v1");
    sessionStorage.removeItem("loading_stage2");
    showLogin();
} else if (STAGE === 1){
    // Stage 1: Check if player has already entered (page reload after Enter)
    const n=localStorage.getItem("player_name");
    if(n && SEED.cards && SEED.cards.length > 0) {
        // Player entered and server has cards ready - show game
        showApp();
    } else if (n && (!SEED.cards || SEED.cards.length === 0)) {
        // Player entered but server not ready - poll for cards
        pollForCards();
    } else {
        // No player name - show login
        showLogin();
    }
}
```

**2. Fixed `doEnter()` to always reload page (lines 961-973)**

**Before (BUGGY):**
```javascript
function doEnter(){
    if (enterBtn) enterBtn.disabled = true;
    const nm=(nameInput.value||"Team Alpha").trim();
    const gt = (document.querySelector('input[name="gameType"]:checked')||{value:'g1'}).value;
    localStorage.setItem("player_name", nm);
    localStorage.setItem("game_type", gt);

    // BUG: Just shows app with stale SEED data!
    showApp();
}
```

**After (FIXED):**
```javascript
function doEnter(){
    if (enterBtn) enterBtn.disabled = true;
    const nm=(nameInput.value||"Team Alpha").trim();
    const gt = (document.querySelector('input[name="gameType"]:checked')||{value:'g1'}).value;
    localStorage.setItem("player_name", nm);
    localStorage.setItem("game_type", gt);

    // CRITICAL FIX: Always reload page to get fresh Stage 1 data
    // This prevents showing stale SEED data from previous game
    console.log('[enter] Reloading page to fetch fresh Stage 1 data');
    location.href = "/";
}
```

**3. Added polling mechanism for Stage 1 cards readiness (lines 906-931)**

Added `pollForCards()` function to handle edge case where:
- User enters name (localStorage set)
- Page reloads
- Server hasn't finished preparing Stage 1 cards yet
- Client polls server every 300ms until cards are ready
- Timeout after 20 attempts (6 seconds) with error handling

---

## How It Works Now

### Flow: Complete Game → Start New Game

1. **User clicks "End Game" on results page**
   - Client sends `/end` → `/reset` → redirects to `/` after 2 seconds
   - Server completes game cleanup and starts new game loop

2. **Client loads `/` (Stage 0 or transitioning to Stage 1)**
   - **If Stage 0:** Shows login, clears all state
   - **If Stage 1 without player_name:** Shows login, clears all state
   - **If Stage 1 with player_name but no cards:** Shows login, polls for cards
   - **If Stage 1 with player_name and cards:** Shows game immediately

3. **User enters name and clicks "Enter"**
   - `doEnter()` saves player_name and game_type to localStorage
   - **CRITICAL:** Reloads page (`location.href = "/"`) to fetch fresh data
   - Does NOT use stale SEED data

4. **After page reload:**
   - Server has prepared Stage 1 with cards
   - Client loads with `SEED = {stage: 1, cards: [9 piles], ...}`
   - Initialization logic detects: `STAGE===1 && localStorage.player_name && SEED.cards.length > 0`
   - Calls `showApp()` → builds grid with fresh cards
   - User can play normally

5. **Edge case: Cards not ready yet**
   - If server still preparing cards after page reload
   - `pollForCards()` polls every 300ms
   - Once cards detected in HTML, reloads page
   - User sees game with cards

---

## Verification & Testing

### Test Scenario 1: Normal Flow
1. Complete a game (all stages)
2. Click "End Game"
3. Wait for redirect to login
4. Enter name and click "Enter"
5. **Expected:** See Stage 1 with 9 fresh piles (new game)
6. **Bug would show:** "No cards at this stage" or stale old cards

### Test Scenario 2: Fast User
1. Complete a game
2. Click "End Game"
3. Immediately enter name and click "Enter" (before 2 sec redirect)
4. **Expected:** Page reloads, shows Stage 1 with fresh cards
5. **Bug would show:** Empty cards or crash

### Test Scenario 3: Server Slow to Prepare
1. Complete a game
2. Click "End Game"
3. Enter name and click "Enter"
4. Server takes 1-2 seconds to prepare cards
5. **Expected:** Polling mechanism detects cards, page reloads, game starts
6. **Bug would show:** "No cards at this stage"

### Test Scenario 4: Multiple Games in Sequence
1. Complete Game 1 → click "End Game"
2. Start Game 2 → verify fresh piles
3. Complete Game 2 → click "End Game"
4. Start Game 3 → verify fresh piles (different from Game 1 and 2)
5. **Expected:** Each game has unique random seed and fresh piles
6. **Bug would show:** Game 3 might show Game 1 or Game 2 piles

---

## Prevention Measures

### Client-Side
1. **Explicit Stage 0 vs Stage 1 handling** - different initialization flows
2. **Always reload after Enter** - never use stale SEED data
3. **Polling mechanism** - handles server preparation delays gracefully
4. **Console logging** - helps debug state transitions

### Server-Side (Already Implemented)
1. **`reset_game_state()`** - called at start of each game loop
2. **Thread-safe state access** - prevents race conditions
3. **Cache-busting headers** - prevents browser from caching stale HTML
4. **Stage-based routing** - serves appropriate content for each stage

---

## Related Files

- **`stage_actions.html`** - Client-side game logic and UI (MODIFIED)
- **`web_wrangler_fixed.py`** - Server HTTP handler and state management (NO CHANGES)
- **`web_game.py`** - Game loop orchestration (NO CHANGES)

---

## Commit Message

```
Fix stale Stage 1 data bug after game completion

User reported: "immediately after a game starting a new game shows in stage 1 Alive No cards at this stage"

Root cause: Race condition where doEnter() showed app with stale SEED data instead of reloading page to get fresh Stage 1 state.

Changes:
- Separated Stage 0 and Stage 1 initialization logic
- doEnter() now always reloads page to fetch fresh data
- Added polling mechanism for Stage 1 cards readiness
- Prevents showing stale data from previous games

Fixes critical bug preventing players from starting new games after completion.
```

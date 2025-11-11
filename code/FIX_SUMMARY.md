# Stage Transition Bug Fix - Summary

## What Was Fixed

The game was getting stuck in an infinite loop showing "your investments are hard at work" when transitioning from Stage 1 to Stage 2.

## Root Cause

The original `web_wrangler.py` created a new HTTP server for each stage, which caused:
- Port binding race conditions
- Socket release delays (TIME_WAIT)
- Unreliable stage transitions
- 502 Bad Gateway errors

## Solution

Switched to persistent singleton server architecture in `web_wrangler_fixed.py`:
- **One server** runs for entire game lifecycle
- **State updates** instead of server restarts
- **Instant transitions** with no port conflicts
- **Always responsive** to client polling

## Changes Made

### 1. web_game.py
```python
# OLD
from web_wrangler import run_ui

# NEW
from web_wrangler_fixed import run_ui, start_persistent_server

# Added at startup (once)
start_persistent_server(port=8765, open_browser=open_first)
```

### 2. web_wrangler_fixed.py
- Fixed syntax error (duplicate global declaration)
- Implements singleton server pattern
- Thread-safe state management

## How It Works Now

```
[Startup]
  └─> start_persistent_server() ← Creates server ONCE
       └─> Server listens on port 8765

[Stage 1]
  └─> run_ui(stage=1, ...) ← Updates _GAME_STATE
       └─> User submits
            └─> POST /submit received

[Transition]
  └─> Game logic processes Stage 1 (~1-2 seconds)
       └─> run_ui(stage=2, ...) ← Updates _GAME_STATE (instant!)
            └─> Client polls "/" every 200ms
                 └─> Sees "stage":2 in response
                      └─> Redirects to Stage 2 ✓

[Stage 2]
  └─> User submits
       └─> run_ui(stage=3, ...) ← Results page
```

## Expected Behavior

- **Before**: Infinite loop, stuck at "hard at work"
- **After**: Smooth 1-3 second transition to Stage 2

## Files Changed

1. `/code/web_game.py` - Use persistent server
2. `/code/web_wrangler_fixed.py` - Fix syntax error
3. `/code/BUGFIX_STAGE_TRANSITION.md` - Detailed documentation

## Testing

Run the game and verify:
1. Stage 1 completes normally
2. "Your investments are hard at work" appears briefly (1-3 sec)
3. Stage 2 loads automatically
4. No more infinite loops or 502 errors

## Rollback

If needed, revert web_game.py import:
```python
from web_wrangler import run_ui  # Old multi-server architecture
```

# Bug Fix: Infinite Loop During Stage 1 to Stage 2 Transition

## Problem Summary
After completing Stage 1, the game sometimes enters an infinite loop showing "your investments are hard at work" on the landing page instead of transitioning to Stage 2.

## Root Cause Analysis

The issue was caused by **server lifecycle management problems** in the multi-server architecture (`web_wrangler.py`):

1. **Server Creation Per Stage**: Each `run_ui()` call created a NEW HTTP server instance
2. **Port Reuse Race Condition**: When Stage 1 completed, the server shut down but didn't release the socket fast enough for Stage 2 to bind
3. **Incomplete Cleanup**: Multiple server threads created conflicting session states
4. **Client Polling Failure**: The client polls for `"stage":2` in the HTML, but if the server hangs during transition, it never sees Stage 2

### Technical Details

**Old Architecture (web_wrangler.py):**
```python
def run_ui(stage, ...):
    # Create NEW server for each stage
    srv = HTTPServer((host, port), _H)

    # Start daemon thread
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    # Wait for submission
    posted_event.wait(timeout=300)

    # Shutdown (but port may not release immediately)
    srv.shutdown()
    srv.server_close()
```

**Problems:**
- Creating/destroying servers creates race conditions
- Socket release isn't immediate (TIME_WAIT state)
- Session state gets confused across server instances
- Client can't reliably detect stage transitions

## Solution

Switched to **persistent singleton server architecture** (`web_wrangler_fixed.py`):

### Key Changes

1. **Single Persistent Server**: Created once at startup, runs for entire game lifecycle
2. **Dynamic State Management**: Server responds differently based on current game state
3. **Clean Stage Transitions**: No server restarts between stages - just state updates
4. **Reliable Client Polling**: Client can always reach server and detect state changes

### Architecture Changes

**New Architecture (web_wrangler_fixed.py):**
```python
# Global persistent server instance
_SERVER_INSTANCE = None
_GAME_STATE = {
    'stage': 0,
    'ctx': {},
    'ready': threading.Event()
}

def start_persistent_server(port, open_browser):
    """Start server once at application startup"""
    global _SERVER_INSTANCE
    srv = HTTPServer((host, port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _SERVER_INSTANCE = srv

def update_game_state(stage, ctx):
    """Update server state (no restart needed)"""
    _GAME_STATE['stage'] = stage
    _GAME_STATE['ctx'] = ctx
    _GAME_STATE['ready'].set()

def run_ui(stage, df, wallet, ...):
    """Update state and wait for user action (no server creation)"""
    if _SERVER_INSTANCE is None:
        start_persistent_server(...)

    update_game_state(stage, ctx)
    _SESSION_EVENT.wait(timeout=300)
    return _SESSION_DATA
```

### Files Modified

1. **web_game.py**:
   - Changed import from `web_wrangler` to `web_wrangler_fixed`
   - Added `start_persistent_server()` call at startup
   - Removed cleanup code for multiple servers
   - Removed `open_browser` parameter from stage transitions (only used on first start)

2. **web_wrangler_fixed.py**:
   - Fixed syntax error (duplicate global declaration in do_POST method)
   - Implements singleton server pattern
   - Dynamic routing based on `_GAME_STATE`
   - Thread-safe state updates with `_SERVER_LOCK`

### Benefits

1. **No Port Conflicts**: Server never releases port during game lifecycle
2. **Instant Transitions**: State updates are immediate (no server restart delay)
3. **Reliable Polling**: Client can always reach server to check stage status
4. **Clean Session Management**: Single source of truth for session state
5. **Better Error Recovery**: Server survives game restarts and errors

## Testing the Fix

### Manual Test Procedure

1. Start the game server:
   ```bash
   python3 web_game.py
   ```

2. Complete Stage 1:
   - Enter team name
   - Select game type
   - Make investments
   - Click "Finish Stage 1"

3. Verify Stage 2 transition:
   - Should see "Your investments are hard at work..." (brief, 1-2 seconds)
   - Should automatically redirect to Stage 2
   - Stage 2 should show:
     - "Stage 2" in header
     - Previous investments visible
     - Only Stage 1 invested piles are interactive
     - Free R2 (second rank) reveals for Stage 1 piles

4. Complete Stage 2:
   - Make additional investments
   - Click "Finish Stage 2"
   - Should see performance modal
   - Click "Show performance"
   - Should see results page

### Expected Behavior

**Stage 1 → Stage 2 Transition:**
- Client submits Stage 1 data via POST /submit
- Server processes Stage 1 (1-2 seconds)
- Server updates state to Stage 2
- Client polls root "/" endpoint every 200ms
- When client sees `"stage":2` in response, redirects to "/"
- Stage 2 loads with new context

**No More Infinite Loop:**
- Server responds immediately to client polls
- No 502 Bad Gateway errors
- No stuck "hard at work" messages
- Smooth transition within 2-3 seconds max

### Debug Logging

The fix includes detailed console logging:

```
[server] Persistent server started at http://127.0.0.1:8765/
[game] New game started with seed: 123456
[web] Stage 1 ready, waiting for user action...
[submit] Stage 1 complete, waiting for Stage 2...
[poll] Still waiting for Stage 2... (0.7s)
[poll] Stage 2 detected after 1.2s, reloading...
[server] Game state updated: stage=2
[web] Stage 2 ready, waiting for user action...
```

## Rollback Plan

If issues arise, revert to original architecture by changing web_game.py:

```python
# Rollback: use old multi-server architecture
from web_wrangler import run_ui  # instead of web_wrangler_fixed
```

And remove the `start_persistent_server()` call.

## Future Improvements

1. **Health Checks**: Add `/health` endpoint checks during polling
2. **Timeout Handling**: Better error messages if Stage 2 takes too long
3. **State Validation**: Verify server state before each transition
4. **Metrics**: Track transition times and success rates
5. **WebSocket Upgrade**: Consider WebSockets for real-time state updates

## Related Issues

This fix also resolves:
- "502 Bad Gateway" errors during stage transitions
- "Stale game state" bugs after errors
- Port exhaustion when running multiple games sequentially

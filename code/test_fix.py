#!/usr/bin/env python3
"""
Test script to verify the stage transition fix works correctly.
This simulates the Stage 1 -> Stage 2 transition without running the full game.
"""

import time
import threading
from web_wrangler_fixed import start_persistent_server, update_game_state, _GAME_STATE, _SERVER_LOCK

def test_server_lifecycle():
    """Test that the persistent server handles stage transitions correctly"""

    print("\n=== Testing Persistent Server Stage Transitions ===\n")

    # 1. Start persistent server (once at startup)
    print("1. Starting persistent server...")
    start_persistent_server(port=8765, open_browser=False)
    time.sleep(0.5)
    print("   ✓ Server started\n")

    # 2. Set up Stage 1 context
    print("2. Setting up Stage 1...")
    stage1_ctx = {
        'stage': 1,
        'wallet': 100.0,
        'cards': [
            {'card_id': 0, 'N': 14, 'med': 7, 'sum2': 20},
            {'card_id': 1, 'N': 13, 'med': 8, 'sum2': 18},
        ],
        'prev_signals': {},
        'prev_invest': {},
        'stage_history': [],
    }
    update_game_state(stage=1, ctx=stage1_ctx)

    # Verify state update
    with _SERVER_LOCK:
        assert _GAME_STATE['stage'] == 1, "Stage should be 1"
        assert _GAME_STATE['ready'].is_set(), "Ready flag should be set"
    print("   ✓ Stage 1 context updated\n")

    # 3. Simulate Stage 1 processing time
    print("3. Simulating Stage 1 processing (1 second)...")
    time.sleep(1.0)
    print("   ✓ Stage 1 processing complete\n")

    # 4. Transition to Stage 2 (should be instant!)
    print("4. Transitioning to Stage 2...")
    start_time = time.time()

    stage2_ctx = {
        'stage': 2,
        'wallet': 90.0,
        'cards': stage1_ctx['cards'],
        'prev_signals': {0: [1]},  # Stage 1 signal
        'prev_invest': {0: 10.0},   # Stage 1 investment
        'stage1_invested': [0],     # Piles invested in Stage 1
        'stage_history': [{'signals': 5.0, 'stakes': 10.0}],
    }
    update_game_state(stage=2, ctx=stage2_ctx)

    transition_time = time.time() - start_time

    # Verify state update
    with _SERVER_LOCK:
        assert _GAME_STATE['stage'] == 2, "Stage should be 2"
        assert _GAME_STATE['ctx']['wallet'] == 90.0, "Wallet should be updated"
        assert _GAME_STATE['ready'].is_set(), "Ready flag should be set"

    print(f"   ✓ Stage 2 transition complete in {transition_time*1000:.2f}ms\n")

    # 5. Verify server is still responsive
    print("5. Verifying server responsiveness...")
    import requests
    try:
        # Note: Server binds to 0.0.0.0 but we can test via 127.0.0.1
        response = requests.get('http://127.0.0.1:8765/health', timeout=2)
        assert response.status_code == 200, "Server should respond to health check"
        print("   ✓ Server is responsive\n")
    except Exception as e:
        print(f"   ✗ Server not responsive: {e}\n")
        return False

    # 6. Verify Stage 2 content in HTML
    print("6. Verifying Stage 2 is served correctly...")
    try:
        response = requests.get('http://127.0.0.1:8765/', timeout=2)
        html = response.text

        # Check for Stage 2 context
        assert '"stage":2' in html or '"stage": 2' in html, "HTML should contain stage 2"
        assert '90.0' in html or '90' in html, "HTML should show updated wallet"

        print("   ✓ Stage 2 HTML served correctly\n")
    except Exception as e:
        print(f"   ✗ Failed to verify Stage 2 HTML: {e}\n")
        return False

    print("=== All Tests Passed! ===\n")
    print("Summary:")
    print(f"  - Server lifecycle: ✓")
    print(f"  - Stage 1 setup: ✓")
    print(f"  - Stage 2 transition: ✓ ({transition_time*1000:.2f}ms)")
    print(f"  - Server responsiveness: ✓")
    print(f"  - HTML correctness: ✓")
    print("\nThe fix is working correctly!")

    return True

if __name__ == '__main__':
    try:
        success = test_server_lifecycle()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

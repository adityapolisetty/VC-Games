# Quick Start Guide - Playwright Testing

Get started with testing in 5 minutes.

---

## Installation (One-Time Setup)

### 1. Install Node.js Dependencies

```bash
cd tests
npm install
npx playwright install chromium
```

**Expected output**:
```
✔ Success! Created a Playwright Test project
...
Downloading browsers...
✔ chromium vXXX downloaded
```

---

## Running Tests - Local

### Quick Test Run (Automatic Server)

The easiest way - Playwright starts and stops the server automatically:

```bash
cd tests
npx playwright test
```

**What happens**:
1. Playwright starts `web_game.py` on port 8765
2. Runs all 41 tests
3. Shows results in terminal
4. Stops the server

**Expected output**:
```
Running 41 tests using 1 worker

  ✓ full_game_flow.spec.ts:10:3 › should complete full game with median signal (5s)
  ✓ full_game_flow.spec.ts:45:3 › should complete full game with top2 signal (4s)
  ...

41 passed (3m)
```

### Manual Server Control (Alternative)

If you want to see server logs:

**Terminal 1 - Start Server**:
```bash
cd code
python web_game.py
```

**Terminal 2 - Run Tests**:
```bash
cd tests
TEST_URL=http://localhost:8765 npx playwright test
```

---

## Running Tests - Railway

After deploying to Railway:

```bash
cd tests
TEST_URL=https://your-app.railway.app npx playwright test
```

Replace `your-app.railway.app` with your actual Railway domain.

---

## Useful Commands

### Run Specific Test File

```bash
# Only full game flow tests
npx playwright test full_game_flow.spec.ts

# Only restart tests
npx playwright test restart_navigation.spec.ts

# Only edge case tests
npx playwright test edge_cases.spec.ts
```

### Run Single Test

```bash
npx playwright test -g "should complete full game with median"
```

### Watch Mode (Auto-rerun on changes)

```bash
npx playwright test --ui
```

### Debug Mode (Step Through)

```bash
npx playwright test --debug
```

### See Browser (Headed Mode)

```bash
npx playwright test --headed
```

### View Test Report

After running tests:

```bash
npx playwright show-report
```

---

## Understanding Test Results

### Success

```
✓ full_game_flow.spec.ts:10:3 › should complete full game (5s)

41 passed (3m)
```

All green checkmarks = all tests passed.

### Failure

```
✗ restart_navigation.spec.ts:15:3 › should restart from Stage 1 (2s)

  Error: Timed out waiting for selector "h1:has-text('VC Card Game')"
```

Red X = test failed. Look at error message for details.

### View Failed Test Details

```bash
# Open HTML report
npx playwright show-report

# Click on failed test to see:
# - Error message
# - Screenshot at failure
# - Video recording
# - Trace (step-by-step)
```

---

## Common Issues & Quick Fixes

### Port 8765 Already in Use

**Error**: `Address already in use`

**Fix**:
```bash
# Find process using port
lsof -i :8765

# Kill it
kill -9 <PID>
```

### Tests Timeout

**Error**: `Timeout 30000ms exceeded`

**Fix**: Increase timeout in `playwright.config.ts`:
```typescript
timeout: 120 * 1000,  // 2 minutes
```

### Database Locked

**Error**: `database is locked`

**Fix**:
```bash
# Ensure no other Python processes
pkill -f web_game.py

# Delete lock file
rm game_data.db-journal

# Re-run tests
```

---

## Test Structure

```
tests/
├── e2e/
│   ├── full_game_flow.spec.ts      # 9 tests - Complete game scenarios
│   ├── restart_navigation.spec.ts  # 10 tests - Restart functionality
│   └── edge_cases.spec.ts          # 22 tests - Boundary conditions
├── helpers/
│   └── page_objects.ts             # Reusable page components
├── test-results/                    # Generated after test run
├── screenshots/                     # Failure screenshots
└── playwright.config.ts            # Configuration
```

---

## Load Testing with Locust

### Install Locust

```bash
pip install locust
```

### Run Load Test

```bash
cd tests

# Web UI (recommended for first time)
locust -f locustfile.py --host=http://localhost:8765

# Open browser to http://localhost:8089
# Enter: 10 users, spawn rate 2
# Click "Start swarming"
```

### Headless Load Test

```bash
# 10 users, 2/second spawn rate, run for 60 seconds
locust -f locustfile.py --host=http://localhost:8765 \
  --users 10 --spawn-rate 2 --run-time 60s --headless
```

### Load Test Railway

```bash
locust -f locustfile.py --host=https://your-app.railway.app \
  --users 20 --spawn-rate 5 --run-time 120s --headless
```

**Monitor**:
- Requests per second
- Response times (P50, P95, P99)
- Failure rate
- Railway metrics (CPU, memory)

---

## Development Workflow

### Before Committing Code

```bash
# 1. Run tests locally
cd tests
npx playwright test

# 2. If all pass, commit
cd ..
git add .
git commit -m "Your commit message"
git push
```

### After Deploying to Railway

```bash
# 1. Run smoke test
cd tests
TEST_URL=https://your-app.railway.app \
  npx playwright test --grep "should complete full game with median"

# 2. If passes, run full suite (optional)
TEST_URL=https://your-app.railway.app npx playwright test
```

---

## Next Steps

1. **Read full documentation**: `tests/README.md`
2. **Review deployment checklist**: `tests/DEPLOYMENT_CHECKLIST.md`
3. **Customize tests** for your specific needs
4. **Set up CI/CD** (GitHub Actions, Railway webhooks)
5. **Add monitoring** (UptimeRobot, Sentry)

---

## Help & Resources

- **Playwright Docs**: https://playwright.dev
- **This Project README**: `tests/README.md`
- **Deployment Guide**: `tests/DEPLOYMENT_CHECKLIST.md`

---

## Cheat Sheet

```bash
# Install
npm install && npx playwright install chromium

# Test local (auto server)
npx playwright test

# Test local (manual server)
TEST_URL=http://localhost:8765 npx playwright test

# Test Railway
TEST_URL=https://your-app.railway.app npx playwright test

# Single test
npx playwright test -g "test name"

# Debug
npx playwright test --debug

# Report
npx playwright show-report

# Load test
locust -f locustfile.py --host=http://localhost:8765
```

---

**That's it! You're ready to test.**

Run `npx playwright test` and watch the magic happen.

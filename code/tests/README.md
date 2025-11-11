# VC Card Game - Playwright Test Suite

Comprehensive end-to-end testing for the VC Card Game application using Playwright.

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running Tests](#running-tests)
5. [Test Structure](#test-structure)
6. [Configuration](#configuration)
7. [CI/CD Integration](#cicd-integration)
8. [Troubleshooting](#troubleshooting)

---

## Testing Strategy

### Phase 1: Local Testing (Recommended First)

Run tests against `http://localhost:8765` before deploying to Railway.

**Benefits**:
- Faster feedback cycle
- Easier debugging with Python server logs
- No deployment overhead
- Free unlimited test runs

**When to use**: During development and before every deployment

### Phase 2: Railway Deployment Testing

Run smoke tests against production after deployment.

**Benefits**:
- Validates production environment
- Tests real network conditions
- Catches environment-specific issues

**When to use**: After deploying to Railway, as post-deployment verification

### Phase 3: Continuous Testing

- **Local**: Full test suite on every code change
- **Railway**: Smoke tests post-deployment via CI/CD

---

## Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- VC Card Game application files

## Installation

### 1. Install Node.js Dependencies

```bash
cd tests
npm init -y
npm install --save-dev @playwright/test@latest
npx playwright install chromium
```

### 2. Install Python Dependencies

```bash
cd ..
pip install -r requirements.txt  # If you have one
# Or manually install:
# pip install numpy pandas sqlite3
```

### 3. Verify Installation

```bash
npx playwright --version
```

---

## Running Tests

### Local Testing (Recommended)

#### Option A: Automatic Server Startup (Recommended)

Playwright will automatically start and stop the server:

```bash
cd tests
npx playwright test
```

#### Option B: Manual Server Startup

If you prefer to control the server manually:

**Terminal 1: Start Server**
```bash
cd code
python web_game.py
```

**Terminal 2: Run Tests**
```bash
cd tests
TEST_URL=http://localhost:8765 npx playwright test
```

### Railway Testing

After deploying to Railway:

```bash
cd tests
TEST_URL=https://your-railway-app.up.railway.app npx playwright test
```

### Run Specific Test Files

```bash
# Full game flow only
npx playwright test full_game_flow.spec.ts

# Restart navigation only
npx playwright test restart_navigation.spec.ts

# Edge cases only
npx playwright test edge_cases.spec.ts
```

### Run Single Test

```bash
npx playwright test -g "should complete full game with median signal"
```

### Debug Mode

```bash
npx playwright test --debug
```

### Headed Mode (See Browser)

```bash
npx playwright test --headed
```

### Generate HTML Report

```bash
npx playwright show-report
```

---

## Test Structure

```
tests/
├── playwright.config.ts          # Main configuration
├── e2e/                           # Test files
│   ├── full_game_flow.spec.ts    # Complete game scenarios
│   ├── restart_navigation.spec.ts # Restart button tests
│   └── edge_cases.spec.ts        # Boundary conditions
├── helpers/
│   └── page_objects.ts           # Reusable page objects
├── screenshots/                   # Screenshots on failure
├── test-results/                  # Test artifacts
└── README.md                      # This file
```

### Test Coverage

#### 1. Full Game Flow Tests (`full_game_flow.spec.ts`)

- Complete game with median signals
- Complete game with top2 signals
- No signal purchases
- Stage 1 only investment
- Maximum signal purchases
- Return calculation verification
- Leaderboard display
- Table formatting (14px font)

**Total**: 9 tests

#### 2. Restart Navigation Tests (`restart_navigation.spec.ts`)

- Restart from Stage 1 (immediate)
- Restart from Stage 1 (after investments)
- Restart from Stage 2
- Restart from Results page
- Multiple restarts with same team name
- Game state cleanup after restart
- Race condition handling (rapid clicking)
- Separate game states for different teams
- Restart during submission
- Restart after viewing full results

**Total**: 10 tests

#### 3. Edge Cases (`edge_cases.spec.ts`)

- Empty team name validation
- Very long team names
- Special characters in team name
- Double-click submission prevention
- Browser refresh (Stage 1, Stage 2, Results)
- Zero investments
- Negative investment values
- Investment exceeding budget
- Fractional investment values
- Stage 2 without Stage 1 base
- Rapid sequential games
- Concurrent requests
- Histogram zero-count exclusion
- Leaderboard highlighting
- Browser back button
- Large investment values
- Table formatting consistency

**Total**: 22 tests

**Grand Total**: 41 comprehensive tests

---

## Configuration

### Environment Variables

- `TEST_URL`: Override base URL (default: `http://localhost:8765`)
- `CI`: Set to `true` in CI environment for stricter settings

### Browser Configuration

Edit `playwright.config.ts` to enable additional browsers:

```typescript
projects: [
  { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
  { name: 'webkit', use: { ...devices['Desktop Safari'] } },
]
```

### Timeouts

Adjust in `playwright.config.ts`:

```typescript
timeout: 60 * 1000,           // 60s per test
navigationTimeout: 30 * 1000, // 30s for navigation
actionTimeout: 10 * 1000,     // 10s for actions
```

---

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/test.yml`:

```yaml
name: Playwright Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        cd tests
        npm ci
        npx playwright install --with-deps chromium

    - name: Run Playwright tests
      run: |
        cd tests
        npx playwright test

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: playwright-report
        path: tests/test-results/
        retention-days: 30
```

### Railway Post-Deployment Hook

Add to Railway project settings:

```bash
# After deployment
curl -X POST https://your-ci-server.com/trigger-tests?env=production
```

---

## Troubleshooting

### Server Won't Start

**Problem**: `Error: Failed to start server on port 8765`

**Solutions**:
1. Check if port 8765 is already in use:
   ```bash
   lsof -i :8765
   kill -9 <PID>
   ```

2. Start server manually and disable auto-start:
   ```bash
   # In playwright.config.ts, set:
   reuseExistingServer: true
   ```

### Tests Timing Out

**Problem**: Tests fail with timeout errors

**Solutions**:
1. Increase timeout in `playwright.config.ts`
2. Check server logs for errors
3. Verify database is accessible
4. Run in headed mode to observe: `npx playwright test --headed`

### Database Lock Errors

**Problem**: SQLite database locked

**Solutions**:
1. Ensure `workers: 1` in config (sequential execution)
2. Close any Python processes accessing the database
3. Delete `game_data.db` and restart

### Screenshot/Video Not Captured

**Problem**: Failures don't generate artifacts

**Solutions**:
1. Verify settings in config:
   ```typescript
   screenshot: 'only-on-failure',
   video: 'retain-on-failure',
   trace: 'retain-on-failure',
   ```

2. Check `test-results/` directory exists
3. Run with explicit screenshot:
   ```typescript
   await page.screenshot({ path: 'debug.png' });
   ```

### Railway Tests Fail but Local Pass

**Problem**: Tests pass locally but fail on Railway

**Possible causes**:
1. Environment variables missing
2. Database not persisted (add Railway volume)
3. Network latency (increase timeouts)
4. Railway service not fully started (add health check)

**Solutions**:
1. Add health check endpoint to `web_wrangler_fixed.py`
2. Increase Railway memory/CPU allocation
3. Check Railway logs for errors

### Flaky Tests

**Problem**: Tests pass sometimes, fail others

**Solutions**:
1. Add explicit waits:
   ```typescript
   await page.waitForLoadState('networkidle');
   ```

2. Use `waitForSelector` instead of immediate assertions:
   ```typescript
   await page.waitForSelector('h2:has-text("Stage 1")');
   ```

3. Increase retries in CI:
   ```typescript
   retries: process.env.CI ? 2 : 0
   ```

---

## Performance Testing with Locust

For load testing multiple concurrent users, see separate Locust guide.

**Quick start**:

```bash
# Install Locust
pip install locust

# Create locustfile.py (see example below)
# Run load test
locust -f locustfile.py --host=http://localhost:8765
```

**Basic Locust Example**:

```python
from locust import HttpUser, task, between

class VCGameUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def play_game(self):
        # Start game
        self.client.post("/", data={
            "team_name": f"LoadTest_{self.user_id}",
            "signal_type": "median"
        })

        # Stage 1
        self.client.post("/stage1", data={
            "invest_0": "50",
            "invest_1": "50"
        })

        # Stage 2
        self.client.post("/stage2", data={
            "invest2_0": "25"
        })
```

---

## Railway Deployment Checklist

### Pre-Deployment

- [ ] All local tests pass
- [ ] Database migrations applied (if any)
- [ ] Environment variables configured in Railway
- [ ] Health check endpoint added
- [ ] Resource limits set (memory, CPU)

### Deployment

- [ ] Deploy to Railway
- [ ] Wait for deployment to complete
- [ ] Verify service is running (check logs)

### Post-Deployment Verification

- [ ] Run smoke tests:
  ```bash
  TEST_URL=https://your-app.railway.app npx playwright test --grep "should complete full game with median"
  ```

- [ ] Manually test one complete game flow
- [ ] Check database is persisting data
- [ ] Verify leaderboard shows historical games
- [ ] Test restart functionality from production

### Performance Validation

- [ ] Run Locust with 10 concurrent users
- [ ] Monitor Railway metrics (CPU, memory)
- [ ] Check response times < 2s for all endpoints
- [ ] Verify no 500 errors under load

---

## Test Maintenance

### Adding New Tests

1. Create test in appropriate spec file
2. Use page objects for reusability
3. Add descriptive test name
4. Include assertions for all critical behaviors
5. Take screenshot on failure for debugging

### Updating Tests After Code Changes

1. Run full test suite locally
2. Update page objects if selectors changed
3. Update assertions if behavior changed
4. Re-run to verify all pass
5. Commit test changes with code changes

### Best Practices

- **Use data-testid**: Add `data-testid` attributes to HTML for stable selectors
- **Avoid sleep()**: Use `waitFor` methods instead
- **Unique team names**: Use `generateTeamName()` to avoid conflicts
- **Sequential execution**: Keep `workers: 1` to prevent database conflicts
- **Cleanup**: Restart game state between tests
- **Screenshots**: Take screenshots on failure for easier debugging

---

## Additional Resources

- [Playwright Documentation](https://playwright.dev)
- [Playwright Best Practices](https://playwright.dev/docs/best-practices)
- [Locust Documentation](https://docs.locust.io)
- [Railway Documentation](https://docs.railway.app)

---

## Support

For issues or questions:

1. Check this README troubleshooting section
2. Review Playwright documentation
3. Check Railway logs for server errors
4. Run tests in debug mode: `npx playwright test --debug`

---

## License

Same as VC Card Game project

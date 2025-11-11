# Installation & First Run Guide

Step-by-step installation guide for the Playwright test suite.

---

## Prerequisites Check

Before starting, verify you have:

```bash
# Node.js 18+ (required)
node --version
# Should show: v18.x.x or higher

# npm (comes with Node.js)
npm --version
# Should show: 9.x.x or higher

# Python 3.8+ (required)
python3 --version
# Should show: Python 3.8.x or higher

# Git (required)
git --version
# Should show: git version 2.x.x
```

If any are missing:
- **Node.js**: Download from https://nodejs.org (LTS version)
- **Python**: Download from https://python.org
- **Git**: Download from https://git-scm.com

---

## Step 1: Navigate to Tests Directory

```bash
cd /Users/adityapolisetty/Library/CloudStorage/Dropbox/PhD/freelance/Simulation/VC-Games/code/tests
```

---

## Step 2: Install Node.js Dependencies

```bash
npm install
```

**Expected output**:
```
added 12 packages, and audited 13 packages in 2s
found 0 vulnerabilities
```

**If errors occur**:
```bash
# Remove package-lock.json and try again
rm package-lock.json
npm install
```

---

## Step 3: Install Playwright Browsers

```bash
npx playwright install chromium
```

**Expected output**:
```
Downloading browsers...
  - chromium v1095 (100 MB) ✓
Done in 60s
```

**If you want all browsers** (optional):
```bash
npx playwright install  # Installs Chromium, Firefox, WebKit
```

---

## Step 4: Verify Installation

```bash
npx playwright --version
```

**Expected output**:
```
Version 1.40.0
```

---

## Step 5: Run Your First Test

```bash
npx playwright test --grep "should complete full game with median"
```

**What happens**:
1. Playwright starts the Python server automatically
2. Runs 1 smoke test (complete game flow)
3. Shows results in terminal
4. Stops the server

**Expected output**:
```
Running 1 test using 1 worker

  ✓ full_game_flow.spec.ts:10:3 › should complete full game with median signal (5s)

1 passed (7s)
```

---

## Step 6: Run Full Test Suite

```bash
npx playwright test
```

**What happens**:
1. Starts server
2. Runs all 41 tests
3. Shows progress in terminal
4. Stops server

**Expected output**:
```
Running 41 tests using 1 worker

  ✓ full_game_flow.spec.ts:10:3 › should complete full game with median signal (5s)
  ✓ full_game_flow.spec.ts:45:3 › should complete full game with top2 signal (4s)
  ...
  ✓ edge_cases.spec.ts:463:3 › should maintain table formatting (3s)

41 passed (3m 24s)
```

---

## Step 7: View Test Report

```bash
npx playwright show-report
```

**What happens**:
- Opens browser with interactive HTML report
- Shows all test results
- Includes screenshots/videos if any failures

---

## Troubleshooting Installation

### Port 8765 Already in Use

**Error**:
```
Error: Port 8765 is already in use
```

**Solution**:
```bash
# Find process using port
lsof -i :8765

# Kill it
kill -9 <PID>

# Or change port in playwright.config.ts
```

### Playwright Install Fails

**Error**:
```
Failed to install browsers
```

**Solution**:
```bash
# Install with system dependencies
npx playwright install --with-deps chromium

# On Mac with Apple Silicon, may need Rosetta
softwareupdate --install-rosetta
```

### Python Server Won't Start

**Error**:
```
Error: spawn python ENOENT
```

**Solution**:
```bash
# Verify Python is in PATH
which python3

# Update playwright.config.ts to use python3 explicitly:
# command: 'python3 web_game.py'
```

### Tests Fail with "Cannot find module"

**Error**:
```
Cannot find module '@playwright/test'
```

**Solution**:
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

---

## Next Steps

After successful installation:

1. **Read Quick Start**: `QUICKSTART.md`
2. **Review Testing Strategy**: `TESTING_STRATEGY.md`
3. **Run specific test suites**:
   ```bash
   npx playwright test full_game_flow.spec.ts
   npx playwright test restart_navigation.spec.ts
   npx playwright test edge_cases.spec.ts
   ```
4. **Set up for Railway testing**: See `DEPLOYMENT_CHECKLIST.md`

---

## Verification Checklist

- [ ] Node.js 18+ installed
- [ ] npm installed
- [ ] Python 3.8+ installed
- [ ] `npm install` completed successfully
- [ ] `npx playwright install chromium` completed
- [ ] First test passes: `npx playwright test --grep "should complete full game"`
- [ ] Full test suite runs: `npx playwright test`
- [ ] HTML report opens: `npx playwright show-report`

---

## Installation Complete!

You're ready to test. Run:

```bash
npx playwright test
```

And watch 41 comprehensive tests validate your application.

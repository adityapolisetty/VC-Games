# Railway Deployment Checklist

Complete checklist for deploying and testing the VC Card Game on Railway.

---

## Pre-Deployment Checklist

### Code Readiness

- [ ] All local Playwright tests pass (41/41)
  ```bash
  cd tests
  npx playwright test
  ```

- [ ] Manual testing completed
  - [ ] Full game flow with median signals
  - [ ] Full game flow with top2 signals
  - [ ] Restart from each stage
  - [ ] Multiple sequential games

- [ ] Code review completed
  - [ ] Recent fixes verified (restart navigation, table formatting)
  - [ ] No console errors in browser
  - [ ] No Python errors in server logs

### Database

- [ ] Database schema is correct
  ```bash
  sqlite3 game_data.db ".schema"
  ```

- [ ] Database file size is reasonable (< 100MB for initial deployment)
  ```bash
  ls -lh game_data.db
  ```

- [ ] Test data cleaned up (optional - remove test teams)
  ```sql
  DELETE FROM games WHERE team_name LIKE 'Test%';
  ```

### Configuration Files

- [ ] `web_game.py` has correct port configuration
  ```python
  # Should bind to 0.0.0.0 for Railway
  PORT = int(os.environ.get('PORT', 8765))
  ```

- [ ] Dependencies documented
  - [ ] Create `requirements.txt`:
    ```bash
    pip freeze > requirements.txt
    ```

- [ ] `.gitignore` excludes test artifacts
  ```
  tests/test-results/
  tests/screenshots/
  tests/playwright-report/
  node_modules/
  __pycache__/
  *.pyc
  ```

### Railway Setup

- [ ] Railway account created
- [ ] Railway CLI installed (optional)
  ```bash
  npm install -g @railway/cli
  railway login
  ```

- [ ] Project created in Railway dashboard

---

## Deployment Steps

### 1. Connect Repository

- [ ] Link GitHub repository to Railway project
  - Or use Railway CLI: `railway link`

### 2. Configure Environment Variables

In Railway dashboard, set:

- [ ] `PORT` = `8765` (Railway will override this automatically)
- [ ] `DATABASE_PATH` = `/app/data/game_data.db` (if using persistent volume)
- [ ] `PYTHON_VERSION` = `3.10` or `3.11`

### 3. Configure Build Settings

- [ ] **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Start Command**:
  ```bash
  python web_game.py
  ```

- [ ] **Root Directory**: `/code` (if repository has subdirectories)

### 4. Add Persistent Volume (Recommended)

For database persistence across deployments:

- [ ] Create volume in Railway dashboard
- [ ] Mount point: `/app/data`
- [ ] Update database path in code to use `/app/data/game_data.db`

### 5. Configure Health Checks (Optional but Recommended)

Add to `web_wrangler_fixed.py`:

```python
def do_GET(self):
    if self.path == '/health':
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')
        return
    # ... rest of existing code
```

- [ ] Health check endpoint added
- [ ] Configure in Railway: `/health` endpoint

### 6. Deploy

- [ ] Commit and push to GitHub
  ```bash
  git add .
  git commit -m "Deploy to Railway"
  git push origin main
  ```

- [ ] Railway automatically builds and deploys
- [ ] Monitor build logs in Railway dashboard

### 7. Verify Deployment

- [ ] Check Railway logs for startup messages
- [ ] Verify no error messages
- [ ] Note the deployment URL (e.g., `https://vc-card-game-production.up.railway.app`)

---

## Post-Deployment Verification

### Immediate Checks

- [ ] Service is running
  ```bash
  curl https://your-app.railway.app/health
  # Should return: OK
  ```

- [ ] Landing page loads
  ```bash
  curl https://your-app.railway.app/
  # Should return HTML with "VC Card Game"
  ```

- [ ] No 500 errors in Railway logs

### Manual Testing

- [ ] Play one complete game manually
  1. [ ] Open URL in browser
  2. [ ] Enter team name and select signal type
  3. [ ] Purchase signals and make Stage 1 investments
  4. [ ] Complete Stage 2
  5. [ ] View results page
  6. [ ] Verify all tables display correctly
  7. [ ] Click Restart and verify returns to landing page

- [ ] Test restart functionality from each stage
  - [ ] Stage 1 → Restart → Landing page
  - [ ] Stage 2 → Restart → Landing page
  - [ ] Results → Restart → Landing page

### Automated Testing

- [ ] Run Playwright smoke tests
  ```bash
  cd tests
  TEST_URL=https://your-app.railway.app npx playwright test --grep "should complete full game with median"
  ```

- [ ] Run full Playwright test suite (optional)
  ```bash
  TEST_URL=https://your-app.railway.app npx playwright test
  ```

- [ ] All critical tests pass (at minimum: full game flow tests)

### Database Verification

- [ ] Database persists data across restarts
  1. [ ] Play game and note team name
  2. [ ] Trigger Railway redeploy
  3. [ ] Play another game
  4. [ ] Check leaderboard shows both games

- [ ] Leaderboard displays historical games

### Performance Testing

- [ ] Run basic load test
  ```bash
  cd tests
  pip install locust
  locust -f locustfile.py --host=https://your-app.railway.app --users 10 --spawn-rate 2 --run-time 60s --headless
  ```

- [ ] Monitor Railway metrics during load test
  - [ ] CPU usage < 80%
  - [ ] Memory usage < 80%
  - [ ] No crashes or restarts

- [ ] Key performance metrics
  - [ ] P95 response time < 2000ms
  - [ ] Error rate < 1%
  - [ ] Successful request rate > 99%

### User Experience Validation

- [ ] Test on different devices
  - [ ] Desktop browser (Chrome, Firefox)
  - [ ] Mobile browser (iOS Safari, Android Chrome)
  - [ ] Tablet

- [ ] Verify responsive design works
- [ ] No JavaScript console errors
- [ ] All tables formatted correctly (14px font, consistent padding)

---

## Rollback Plan

If deployment fails or critical issues found:

### Option 1: Rollback in Railway

- [ ] Go to Railway dashboard
- [ ] Select "Deployments" tab
- [ ] Click "Rollback" on previous working deployment

### Option 2: Hotfix

- [ ] Fix issue locally
- [ ] Test fix thoroughly
- [ ] Commit and push
  ```bash
  git add .
  git commit -m "Hotfix: [description]"
  git push origin main
  ```
- [ ] Monitor Railway auto-deploy

### Option 3: Emergency Shutdown

- [ ] Pause service in Railway dashboard
- [ ] Display maintenance page (optional)
- [ ] Fix issues offline
- [ ] Resume service after verification

---

## Monitoring & Maintenance

### Daily Checks

- [ ] Check Railway logs for errors
- [ ] Monitor resource usage (CPU, memory)
- [ ] Verify service uptime

### Weekly Checks

- [ ] Review leaderboard for data integrity
- [ ] Check database size growth
- [ ] Run performance benchmark

### Monthly Checks

- [ ] Database cleanup (archive old games if needed)
- [ ] Review and optimize slow queries
- [ ] Update dependencies
- [ ] Re-run full test suite

---

## Troubleshooting Guide

### Service Won't Start

**Symptoms**: Deployment succeeds but service crashes

**Check**:
1. Railway logs for Python errors
2. Database path is correct
3. All dependencies installed
4. Port binding to `0.0.0.0` not `localhost`

**Fix**:
```python
# In web_game.py
server_address = ('0.0.0.0', PORT)  # Not '127.0.0.1'
```

### Database Not Persisting

**Symptoms**: Games disappear after redeploy

**Check**:
1. Volume is mounted correctly
2. Database path points to volume
3. Write permissions on volume

**Fix**:
- Add persistent volume in Railway
- Update `DATABASE_PATH` to use volume mount point

### Slow Response Times

**Symptoms**: Pages take > 5 seconds to load

**Check**:
1. Railway resource limits
2. Database query performance
3. Monte Carlo simulation parameters

**Fix**:
- Upgrade Railway plan
- Add database indexes
- Optimize simulation (reduce iterations)

### 500 Internal Server Errors

**Symptoms**: Random 500 errors

**Check**:
1. Railway logs for Python exceptions
2. Database connection errors
3. Memory limits exceeded

**Fix**:
- Add error handling to `web_wrangler_fixed.py`
- Increase Railway memory allocation
- Add database connection pooling

### CSS/JS Not Loading

**Symptoms**: Page loads but unstyled

**Check**:
1. Inline styles in HTML template
2. No external CSS/JS dependencies missing

**Fix**:
- Ensure all styles are inline in `stage_actions.html`

---

## Success Criteria

Deployment is considered successful when:

- [x] Service is accessible at Railway URL
- [x] Landing page loads within 2 seconds
- [x] Complete game flow works end-to-end
- [x] Restart functionality works from all stages
- [x] Database persists data across redeployments
- [x] Leaderboard displays correctly
- [x] No errors in Railway logs for 24 hours
- [x] Performance under load: 10 concurrent users, < 1% error rate
- [x] All critical Playwright tests pass

---

## Next Steps After Successful Deployment

1. **Share URL** with beta testers
2. **Monitor** first week closely
3. **Collect feedback** on UX and performance
4. **Set up alerts** for downtime (use Railway webhooks or UptimeRobot)
5. **Document** any production-specific configuration
6. **Plan** for scaling if user base grows

---

## Contact & Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Playwright Issues: https://github.com/microsoft/playwright/issues

---

**Deployment Date**: _____________

**Deployed By**: _____________

**Railway URL**: _____________

**Deployment Notes**:

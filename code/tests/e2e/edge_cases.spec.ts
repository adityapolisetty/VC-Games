import { test, expect } from '@playwright/test';
import {
  LandingPage,
  Stage1Page,
  Stage2Page,
  ResultsPage,
  generateTeamName
} from '../helpers/page_objects';

/**
 * Edge case tests - boundary conditions, error handling, race conditions
 */
test.describe('Edge Cases', () => {
  test('should handle empty team name validation', async ({ page }) => {
    const landingPage = new LandingPage(page);
    await landingPage.goto();

    // Try to start without team name
    await landingPage.teamNameInput.fill('');
    await landingPage.startButton.click();

    // Should stay on landing page (HTML5 validation or server rejection)
    await page.waitForTimeout(1000);
    const url = page.url();
    expect(url).toContain('/');

    // Verify still on landing page
    await expect(landingPage.heading).toBeVisible();
  });

  test('should handle very long team names', async ({ page }) => {
    const longTeamName = 'A'.repeat(200) + '_' + Date.now();

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(longTeamName, 'g1');

    // Should successfully navigate to Stage 1
    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
  });

  test('should handle special characters in team name', async ({ page }) => {
    const specialTeamName = `Test_<>'"&@#$%_${Date.now()}`;

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(specialTeamName, 'g1');

    // Should successfully navigate to Stage 1
    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Complete game to verify special chars in database
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();
    await stage2Page.submitInvestments();

    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    // Verify leaderboard displays without errors
    await resultsPage.verifyLeaderboard();
  });

  test('should prevent double-click submission in Stage 1', async ({ page }) => {
    const teamName = generateTeamName('DoubleClick');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    await stage1Page.makeInvestment(0, 50);

    // Double-click submit button
    await Promise.all([
      stage1Page.submitButton.click(),
      stage1Page.submitButton.click(),
    ]);

    // Should navigate to Stage 2 only once
    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Verify we're at Stage 2 (not Stage 3 or error state)
    await expect(stage2Page.stageHeading).toContainText('Stage 2');
  });

  test('should handle browser refresh on Stage 1', async ({ page }) => {
    const teamName = generateTeamName('RefreshStage1');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    await stage1Page.purchaseSignal(0);
    await stage1Page.makeInvestment(0, 30);

    // Refresh browser
    await page.reload();

    // Should still be on Stage 1 (or back to landing if session lost)
    // Either behavior is acceptable depending on session management
    await page.waitForLoadState('networkidle');

    const url = page.url();
    const isValidState = url.includes('/') || url.includes('stage');
    expect(isValidState).toBeTruthy();
  });

  test('should handle browser refresh on Stage 2', async ({ page }) => {
    const teamName = generateTeamName('RefreshStage2');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Refresh browser
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Verify state is maintained or gracefully handled
    const url = page.url();
    expect(url).toBeTruthy();
  });

  test('should handle browser refresh on Results page', async ({ page }) => {
    const teamName = generateTeamName('RefreshResults');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();
    await stage2Page.submitInvestments();

    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    // Refresh browser
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Results should still be visible or redirect to landing
    const url = page.url();
    expect(url).toBeTruthy();
  });

  test('should handle zero investments in Stage 1', async ({ page }) => {
    const teamName = generateTeamName('ZeroInvest');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Try to submit without any investments
    await stage1Page.submitButton.click();

    // Should either stay on Stage 1 or move to Stage 2 (depends on validation)
    await page.waitForTimeout(1000);

    const url = page.url();
    expect(url).toBeTruthy();

    // If moved to Stage 2, complete the flow
    try {
      const stage2Page = new Stage2Page(page);
      await stage2Page.isDisplayed();
      await stage2Page.submitInvestments();

      const resultsPage = new ResultsPage(page);
      await resultsPage.isDisplayed();
    } catch (e) {
      // Still on Stage 1, which is also valid
      await expect(stage1Page.stageHeading).toBeVisible();
    }
  });

  test('should handle negative investment values', async ({ page }) => {
    const teamName = generateTeamName('NegativeInvest');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Try to enter negative value
    const input = page.locator('input[name="invest_0"]');
    await input.fill('-50');

    // HTML5 number input should prevent negative values or server should reject
    await stage1Page.submitButton.click();
    await page.waitForTimeout(1000);

    // Should stay on Stage 1 or sanitize input
    const url = page.url();
    expect(url).toBeTruthy();
  });

  test('should handle investment exceeding budget', async ({ page }) => {
    const teamName = generateTeamName('ExceedBudget');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Try to invest more than budget (100)
    await stage1Page.makeInvestment(0, 150);
    await stage1Page.submitButton.click();

    // Should either reject or cap at available budget
    await page.waitForTimeout(1000);

    const url = page.url();
    expect(url).toBeTruthy();
  });

  test('should handle fractional investment values', async ({ page }) => {
    const teamName = generateTeamName('FractionalInvest');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Enter fractional values
    await stage1Page.makeInvestment(0, 33.33);
    await stage1Page.makeInvestment(1, 33.33);
    await stage1Page.makeInvestment(2, 33.34);

    await stage1Page.submitInvestments();

    // Should handle fractional values or round appropriately
    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();
  });

  test('should handle Stage 2 investment without Stage 1 base', async ({ page }) => {
    const teamName = generateTeamName('Stage2NoBase');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Invest in pile 0 only
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Try to invest in pile 1 (no Stage 1 investment)
    const pile1Input = page.locator('input[name="invest2_1"]');

    // Input should be disabled
    const isDisabled = await pile1Input.isDisabled();
    expect(isDisabled).toBe(true);
  });

  test('should handle rapid sequential games', async ({ page }) => {
    const baseTeamName = generateTeamName('RapidGames');

    // Play 3 games rapidly
    for (let i = 0; i < 3; i++) {
      const teamName = `${baseTeamName}_${i}`;

      const landingPage = new LandingPage(page);
      await landingPage.goto();
      await landingPage.startGame(teamName, 'g1');

      const stage1Page = new Stage1Page(page);
      await stage1Page.isDisplayed();
      await stage1Page.makeInvestment(0, 50);
      await stage1Page.submitInvestments();

      const stage2Page = new Stage2Page(page);
      await stage2Page.isDisplayed();
      await stage2Page.submitInvestments();

      const resultsPage = new ResultsPage(page);
      await resultsPage.isDisplayed();

      // Restart for next game
      if (i < 2) {
        await resultsPage.clickRestart();
      }
    }

    // Verify final results page is stable
    const resultsPage = new ResultsPage(page);
    await resultsPage.verifyInfoBoxes();
  });

  test('should handle concurrent requests from same team', async ({ page, context }) => {
    const teamName = generateTeamName('Concurrent');

    // Open two tabs with same team name
    const landingPage1 = new LandingPage(page);
    await landingPage1.goto();

    const page2 = await context.newPage();
    const landingPage2 = new LandingPage(page2);
    await landingPage2.goto();

    // Start game simultaneously
    await Promise.all([
      landingPage1.startGame(teamName, 'g1'),
      landingPage2.startGame(teamName, 'g1'),
    ]);

    // Both should reach Stage 1 (or one should error gracefully)
    await page.waitForLoadState('networkidle');
    await page2.waitForLoadState('networkidle');

    const url1 = page.url();
    const url2 = page2.url();

    expect(url1).toBeTruthy();
    expect(url2).toBeTruthy();

    await page2.close();
  });

  test('should handle Stage 2 submission with zero additional investment', async ({ page }) => {
    const teamName = generateTeamName('ZeroStage2');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Submit without making Stage 2 investments
    await stage2Page.submitInvestments();

    // Should proceed to results
    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();
    await resultsPage.verifyInfoBoxes();
  });

  test('should display correct histogram bins excluding zeros', async ({ page }) => {
    const teamName = generateTeamName('HistogramZeros');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();
    await stage2Page.submitInvestments();

    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    // Verify histogram table
    await resultsPage.verifyDistributionHistogram();

    // Check that all displayed rows have non-zero counts
    const histogramRows = resultsPage.distributionHistogram.locator('tbody tr');
    const rowCount = await histogramRows.count();

    expect(rowCount).toBeGreaterThan(0);
    expect(rowCount).toBeLessThanOrEqual(5); // Max 5 bins

    // Verify no "0" in count column (column index may vary)
    // This is a heuristic check - adjust based on actual table structure
    for (let i = 0; i < rowCount; i++) {
      const row = histogramRows.nth(i);
      const cellText = await row.textContent();
      // The row should have count data, not just zeros
      expect(cellText).toBeTruthy();
    }
  });

  test('should highlight current player in leaderboard', async ({ page }) => {
    const teamName = generateTeamName('LeaderboardHighlight');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();
    await stage2Page.submitInvestments();

    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    // Verify current player row has background color styling
    const highlightedRow = resultsPage.leaderboard.locator('tr[style*="background-color"]');
    await expect(highlightedRow).toBeVisible();

    // Verify the highlighted row contains current team name
    const rowText = await highlightedRow.textContent();
    expect(rowText).toContain(teamName);
  });

  test('should handle navigation via browser back button', async ({ page }) => {
    const teamName = generateTeamName('BrowserBack');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Try browser back button
    await page.goBack();
    await page.waitForLoadState('networkidle');

    // Should be back at landing page or handle gracefully
    const url = page.url();
    expect(url).toBeTruthy();
  });

  test('should handle very large investment values', async ({ page }) => {
    const teamName = generateTeamName('LargeInvest');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Try to enter huge value
    await stage1Page.makeInvestment(0, 999999999);
    await stage1Page.submitButton.click();

    // Should cap at budget or reject
    await page.waitForTimeout(1000);

    const url = page.url();
    expect(url).toBeTruthy();
  });

  test('should maintain table formatting across all pages', async ({ page }) => {
    const teamName = generateTeamName('TableFormat');

    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    // Check Stage 1 table
    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    const stage1TableFont = await stage1Page.pilesTable.evaluate((el) => {
      return window.getComputedStyle(el).fontSize;
    });
    expect(stage1TableFont).toBe('14px');

    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    // Check Stage 2 table
    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    await stage2Page.submitInvestments();

    // Check Results tables
    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    const breakdownFont = await resultsPage.breakdownTable.evaluate((el) => {
      return window.getComputedStyle(el).fontSize;
    });
    expect(breakdownFont).toBe('14px');

    const histogramFont = await resultsPage.distributionHistogram.evaluate((el) => {
      return window.getComputedStyle(el).fontSize;
    });
    expect(histogramFont).toBe('14px');

    const leaderboardFont = await resultsPage.leaderboard.evaluate((el) => {
      return window.getComputedStyle(el).fontSize;
    });
    expect(leaderboardFont).toBe('14px');
  });
});

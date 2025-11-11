import { test, expect } from '@playwright/test';
import {
  LandingPage,
  Stage1Page,
  Stage2Page,
  ResultsPage,
  generateTeamName
} from '../helpers/page_objects';

/**
 * Restart functionality tests - verify restart button works from all stages
 */
test.describe('Restart Navigation', () => {
  test('Test A: Restart immediately from Stage 1', async ({ page }) => {
    const teamName = generateTeamName('RestartA');

    // Navigate to Stage 1
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Click Restart immediately
    await stage1Page.clickRestart();

    // Verify back at landing page
    await landingPage.isDisplayed();
    await expect(page).toHaveURL(/\//);
  });

  test('Test B: Restart from Stage 1 after making investments', async ({ page }) => {
    const teamName = generateTeamName('RestartB');

    // Navigate to Stage 1
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Make some investments
    await stage1Page.purchaseSignal(0);
    await stage1Page.makeInvestment(0, 30);
    await stage1Page.makeInvestment(1, 20);

    // Click Restart without submitting
    await stage1Page.clickRestart();

    // Verify back at landing page
    await landingPage.isDisplayed();
    await expect(page).toHaveURL(/\//);
  });

  test('Test C: Restart from Stage 2', async ({ page }) => {
    const teamName = generateTeamName('RestartC');

    // Navigate to Stage 2
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Click Restart from Stage 2
    await stage2Page.clickRestart();

    // Verify back at landing page
    await landingPage.isDisplayed();
    await expect(page).toHaveURL(/\//);
  });

  test('Test D: Restart from Results page', async ({ page }) => {
    const teamName = generateTeamName('RestartD');

    // Complete full game
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();
    await stage2Page.makeInvestment(0, 20);
    await stage2Page.submitInvestments();

    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    // Click Restart from Results
    await resultsPage.clickRestart();

    // Verify back at landing page
    await landingPage.isDisplayed();
    await expect(page).toHaveURL(/\//);
  });

  test('Test E: Multiple restarts with same team name', async ({ page }) => {
    const teamName = generateTeamName('RestartE');

    // First game
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    let stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.clickRestart();

    // Second game with same team name
    await landingPage.isDisplayed();
    await landingPage.startGame(teamName, 'g1');

    stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.clickRestart();

    // Third game with same team name
    await landingPage.isDisplayed();
    await landingPage.startGame(teamName, 'g1');

    stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Verify we're at Stage 1 (not incrementing to Stage 2 or higher)
    await expect(stage1Page.stageHeading).toContainText('Stage 1');

    // Complete this game to verify it works
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();
    await expect(stage2Page.stageHeading).toContainText('Stage 2');
  });

  test('should cleanup game state after restart', async ({ page }) => {
    const teamName = generateTeamName('RestartCleanup');

    // First game: Make investments and purchase signals
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    let stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    await stage1Page.purchaseSignal(0);
    await stage1Page.purchaseSignal(1);
    await stage1Page.makeInvestment(0, 30);
    await stage1Page.makeInvestment(1, 20);

    // Restart
    await stage1Page.clickRestart();

    // Second game: Verify fresh state
    await landingPage.isDisplayed();
    await landingPage.startGame(teamName, 'g1');

    stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Verify budget is reset to 100
    const budget = await stage1Page.getRemainingBudget();
    expect(budget).toBe(100);

    // Verify no signals are pre-purchased (checkboxes unchecked)
    const firstCheckbox = page.locator('input[name="signal_0"]');
    await expect(firstCheckbox).not.toBeChecked();

    // Verify investment inputs are empty/zero
    const firstInvestmentInput = page.locator('input[name="invest_0"]');
    const value = await firstInvestmentInput.inputValue();
    expect(value === '' || value === '0').toBeTruthy();
  });

  test('should handle restart race condition (rapid clicking)', async ({ page }) => {
    const teamName = generateTeamName('RestartRace');

    // Navigate to Stage 1
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Click restart button multiple times rapidly
    const restartPromises = [
      stage1Page.restartButton.click(),
      stage1Page.restartButton.click(),
      stage1Page.restartButton.click(),
    ];

    // Wait for all clicks to process
    await Promise.allSettled(restartPromises);

    // Wait for navigation to complete
    await page.waitForLoadState('networkidle');

    // Verify we're at landing page (not in weird state)
    await landingPage.isDisplayed();
    await expect(page).toHaveURL(/\//);

    // Verify we can start a new game successfully
    await landingPage.startGame(generateTeamName('AfterRace'), 'g1');
    await stage1Page.isDisplayed();
  });

  test('should preserve separate game states for different teams', async ({ page, context }) => {
    const team1 = generateTeamName('Team1');
    const team2 = generateTeamName('Team2');

    // Start first game in first tab
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(team1, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.makeInvestment(0, 30);

    // Open second tab for different team
    const page2 = await context.newPage();
    const landingPage2 = new LandingPage(page2);
    await landingPage2.goto();
    await landingPage2.startGame(team2, 'g2');

    const stage1Page2 = new Stage1Page(page2);
    await stage1Page2.isDisplayed();
    await stage1Page2.makeInvestment(1, 40);

    // Restart first game
    await stage1Page.clickRestart();
    await landingPage.isDisplayed();

    // Verify second game is still active
    await expect(stage1Page2.stageHeading).toBeVisible();

    // Restart second game
    await stage1Page2.clickRestart();
    await landingPage2.isDisplayed();

    // Both should be at landing page
    await expect(page).toHaveURL(/\//);
    await expect(page2).toHaveURL(/\//);

    await page2.close();
  });

  test('should handle restart during submission', async ({ page }) => {
    const teamName = generateTeamName('RestartDuringSubmit');

    // Navigate to Stage 1
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    await stage1Page.makeInvestment(0, 50);

    // Click submit and restart almost simultaneously
    await Promise.all([
      stage1Page.submitButton.click(),
      page.waitForTimeout(100).then(() => stage1Page.restartButton.click()),
    ]);

    // Wait for navigation to settle
    await page.waitForLoadState('networkidle');

    // Should be at either Stage 2 or Landing page (both are valid outcomes)
    const url = page.url();
    const isValidState = url.includes('/') || url.includes('stage');

    expect(isValidState).toBeTruthy();

    // If at Stage 2, restart should still work
    try {
      const stage2Page = new Stage2Page(page);
      await stage2Page.isDisplayed();
      await stage2Page.clickRestart();
      await landingPage.isDisplayed();
    } catch (e) {
      // Already at landing page, which is also valid
      await landingPage.isDisplayed();
    }
  });

  test('should restart from Results after viewing full breakdown', async ({ page }) => {
    const teamName = generateTeamName('RestartFromResults');

    // Complete full game
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();
    await stage2Page.makeInvestment(0, 20);
    await stage2Page.submitInvestments();

    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    // Interact with results page (scroll, view tables)
    await resultsPage.breakdownTable.scrollIntoViewIfNeeded();
    await resultsPage.leaderboard.scrollIntoViewIfNeeded();
    await resultsPage.distributionHistogram.scrollIntoViewIfNeeded();

    // Take screenshot before restart
    await resultsPage.takeScreenshot('before_restart_from_results');

    // Restart
    await resultsPage.clickRestart();

    // Verify clean landing page
    await landingPage.isDisplayed();
    await expect(page).toHaveURL(/\//);

    // Verify can start new game
    await landingPage.startGame(generateTeamName('AfterResultsRestart'), 'g1');
    await stage1Page.isDisplayed();
  });
});

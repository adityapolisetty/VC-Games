import { test, expect } from '@playwright/test';
import {
  LandingPage,
  Stage1Page,
  Stage2Page,
  ResultsPage,
  generateTeamName
} from '../helpers/page_objects';

/**
 * Complete game flow test - plays through all stages
 */
test.describe('Full Game Flow', () => {
  test('should complete full game with median signal type', async ({ page }) => {
    const teamName = generateTeamName('FullFlow');

    // Stage 0: Landing Page
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.isDisplayed();
    await landingPage.startGame(teamName, 'g1');

    // Stage 1: Investment with signals
    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Purchase signals for first 3 piles
    await stage1Page.purchaseSignal(0);
    await stage1Page.purchaseSignal(1);
    await stage1Page.purchaseSignal(2);

    // Verify signals are revealed
    await stage1Page.verifySignalRevealed(0);
    await stage1Page.verifySignalRevealed(1);
    await stage1Page.verifySignalRevealed(2);

    // Make investments (invest in piles 0, 1, 2)
    await stage1Page.makeInvestment(0, 30);
    await stage1Page.makeInvestment(1, 20);
    await stage1Page.makeInvestment(2, 10);

    // Submit Stage 1
    await stage1Page.submitInvestments();

    // Stage 2: Additional investment with R2 reveals
    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Verify R2 is revealed for Stage 1 invested piles
    await stage2Page.verifyR2Revealed(0);
    await stage2Page.verifyR2Revealed(1);
    await stage2Page.verifyR2Revealed(2);

    // Make Stage 2 investments
    await stage2Page.makeInvestment(0, 15);
    await stage2Page.makeInvestment(1, 10);

    // Submit Stage 2
    await stage2Page.submitInvestments();

    // Stage 3: Results
    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    // Verify all result components
    await resultsPage.verifyInfoBoxes();
    await resultsPage.verifyBreakdownTable();
    await resultsPage.verifyDistributionHistogram();
    await resultsPage.verifyLeaderboard();
    await resultsPage.verifyMeanSDStats();

    // Take screenshot of results
    await resultsPage.takeScreenshot('results_median_full_flow');
  });

  test('should complete full game with top2 signal type', async ({ page }) => {
    const teamName = generateTeamName('FullFlowTop2');

    // Stage 0: Landing Page
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.isDisplayed();
    await landingPage.startGame(teamName, 'g2');

    // Stage 1: Investment with top2 signals
    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Purchase top2 signals
    await stage1Page.purchaseSignal(3);
    await stage1Page.purchaseSignal(4);

    // Verify signals are revealed
    await stage1Page.verifySignalRevealed(3);
    await stage1Page.verifySignalRevealed(4);

    // Make investments
    await stage1Page.makeInvestment(3, 40);
    await stage1Page.makeInvestment(4, 30);

    // Submit Stage 1
    await stage1Page.submitInvestments();

    // Stage 2: Additional investment
    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Verify R2 reveals
    await stage2Page.verifyR2Revealed(3);
    await stage2Page.verifyR2Revealed(4);

    // Make Stage 2 investments
    await stage2Page.makeInvestment(3, 10);

    // Submit Stage 2
    await stage2Page.submitInvestments();

    // Stage 3: Results
    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    // Verify results components
    await resultsPage.verifyInfoBoxes();
    await resultsPage.verifyBreakdownTable();
    await resultsPage.verifyDistributionHistogram();
    await resultsPage.verifyLeaderboard();

    // Take screenshot
    await resultsPage.takeScreenshot('results_top2_full_flow');
  });

  test('should handle no signal purchases', async ({ page }) => {
    const teamName = generateTeamName('NoSignals');

    // Start game
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    // Stage 1: No signals, just investments
    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Make investments without purchasing signals
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.makeInvestment(1, 50);

    // Submit
    await stage1Page.submitInvestments();

    // Stage 2
    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Make Stage 2 investment
    await stage2Page.makeInvestment(0, 20);

    // Submit
    await stage2Page.submitInvestments();

    // Verify results
    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();
    await resultsPage.verifyInfoBoxes();
  });

  test('should handle Stage 1 only investment (skip Stage 2)', async ({ page }) => {
    const teamName = generateTeamName('Stage1Only');

    // Start game
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    // Stage 1: Invest all budget
    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    await stage1Page.makeInvestment(0, 100);

    // Submit
    await stage1Page.submitInvestments();

    // Stage 2: No investments (all budget used in Stage 1)
    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Submit without making investments
    await stage2Page.submitInvestments();

    // Verify results
    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();
    await resultsPage.verifyInfoBoxes();

    // Verify budget info shows full Stage 1 investment
    const budgetText = await resultsPage.budgetInfo.textContent();
    expect(budgetText).toContain('100');
  });

  test('should handle maximum signal purchases', async ({ page }) => {
    const teamName = generateTeamName('MaxSignals');

    // Start game
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    // Stage 1: Purchase signals for all 9 piles
    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    for (let i = 0; i < 9; i++) {
      await stage1Page.purchaseSignal(i);
    }

    // Make minimal investments to proceed
    await stage1Page.makeInvestment(0, 10);

    // Submit
    await stage1Page.submitInvestments();

    // Stage 2
    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    // Submit
    await stage2Page.submitInvestments();

    // Verify results show all signals spent
    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    const signalsText = await resultsPage.signalsSpentInfo.textContent();
    expect(signalsText).toContain('9');
  });

  test('should calculate correct net return', async ({ page }) => {
    const teamName = generateTeamName('ReturnCalc');

    // Start game
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    // Stage 1: Simple investment
    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    // Stage 2: Additional investment
    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();

    await stage2Page.makeInvestment(0, 25);
    await stage2Page.submitInvestments();

    // Verify results calculate return
    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    const netReturn = await resultsPage.getNetReturn();
    // Net return should be a valid number (could be positive or negative)
    expect(typeof netReturn).toBe('number');
    expect(netReturn).toBeGreaterThan(-100); // Loss can't exceed 100%
  });

  test('should display player in leaderboard', async ({ page }) => {
    const teamName = generateTeamName('Leaderboard');

    // Play through complete game
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

    // Verify leaderboard
    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();
    await resultsPage.verifyLeaderboard();

    // Verify current player has a rank
    const rank = await resultsPage.getCurrentPlayerRank();
    expect(rank).toBeGreaterThan(0);
  });

  test('should verify table formatting (14px font)', async ({ page }) => {
    const teamName = generateTeamName('TableFormat');

    // Navigate to Stage 1
    const landingPage = new LandingPage(page);
    await landingPage.goto();
    await landingPage.startGame(teamName, 'g1');

    const stage1Page = new Stage1Page(page);
    await stage1Page.isDisplayed();

    // Check table font size
    const table = stage1Page.pilesTable;
    const fontSize = await table.evaluate((el) => {
      return window.getComputedStyle(el).fontSize;
    });

    expect(fontSize).toBe('14px');

    // Complete game to check Results page formatting
    await stage1Page.makeInvestment(0, 50);
    await stage1Page.submitInvestments();

    const stage2Page = new Stage2Page(page);
    await stage2Page.isDisplayed();
    await stage2Page.submitInvestments();

    const resultsPage = new ResultsPage(page);
    await resultsPage.isDisplayed();

    // Verify Results page tables also have 14px font
    const breakdownTableFont = await resultsPage.breakdownTable.evaluate((el) => {
      return window.getComputedStyle(el).fontSize;
    });
    expect(breakdownTableFont).toBe('14px');
  });
});

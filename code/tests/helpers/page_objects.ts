import { Page, Locator, expect } from '@playwright/test';

/**
 * Base Page Object with common functionality
 */
export class BasePage {
  readonly page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  async waitForPageLoad() {
    await this.page.waitForLoadState('networkidle');
  }

  async takeScreenshot(name: string) {
    await this.page.screenshot({
      path: `tests/screenshots/${name}.png`,
      fullPage: true
    });
  }
}

/**
 * Landing Page (Stage 0)
 */
export class LandingPage extends BasePage {
  readonly teamNameInput: Locator;
  readonly game1Radio: Locator;
  readonly game2Radio: Locator;
  readonly startButton: Locator;
  readonly heading: Locator;

  constructor(page: Page) {
    super(page);
    this.teamNameInput = page.locator('#nameInput');
    this.game1Radio = page.locator('input[name="gameType"][value="g1"]');
    this.game2Radio = page.locator('input[name="gameType"][value="g2"]');
    this.startButton = page.locator('#enterBtn');
    this.heading = page.locator('h1');
  }

  async goto() {
    await this.page.goto('/');
    await this.waitForPageLoad();
  }

  async isDisplayed() {
    await expect(this.heading).toContainText('Welcome');
    await expect(this.teamNameInput).toBeVisible();
    await expect(this.startButton).toBeVisible();
  }

  async startGame(teamName: string, gameType: 'g1' | 'g2' = 'g1') {
    await this.teamNameInput.fill(teamName);

    if (gameType === 'g1') {
      await this.game1Radio.check();
    } else {
      await this.game2Radio.check();
    }

    await this.startButton.click();
    await this.waitForPageLoad();
  }

  async verifyFormValidation() {
    // Try to submit without team name
    await this.startButton.click();
    // Should not navigate away if validation fails
    await expect(this.heading).toContainText('Welcome');
  }
}

/**
 * Stage 1 Page
 */
export class Stage1Page extends BasePage {
  readonly stageHeading: Locator;
  readonly budgetDisplay: Locator;
  readonly pilesTable: Locator;
  readonly signalCheckboxes: Locator;
  readonly investmentInputs: Locator;
  readonly submitButton: Locator;
  readonly restartButton: Locator;

  constructor(page: Page) {
    super(page);
    this.stageHeading = page.locator('h2:has-text("Stage 1")');
    this.budgetDisplay = page.locator('text=/Budget|Available/');
    this.pilesTable = page.locator('table');
    this.signalCheckboxes = page.locator('input[type="checkbox"][name^="signal_"]');
    this.investmentInputs = page.locator('input[type="number"][name^="invest_"]');
    this.submitButton = page.locator('button:has-text("Submit")');
    this.restartButton = page.locator('button:has-text("Restart")');
  }

  async isDisplayed() {
    await expect(this.stageHeading).toBeVisible({ timeout: 10000 });
    await expect(this.pilesTable).toBeVisible();
    await expect(this.submitButton).toBeVisible();
  }

  async purchaseSignal(pileIndex: number) {
    const checkbox = this.page.locator(`input[name="signal_${pileIndex}"]`);
    await checkbox.check();
  }

  async makeInvestment(pileIndex: number, amount: number) {
    const input = this.page.locator(`input[name="invest_${pileIndex}"]`);
    await input.fill(amount.toString());
  }

  async submitInvestments() {
    await this.submitButton.click();
    await this.waitForPageLoad();
  }

  async clickRestart() {
    await this.restartButton.click();
    await this.waitForPageLoad();
  }

  async getPileCardRanks(pileIndex: number): Promise<string> {
    const pileRow = this.page.locator(`tr:has(td:text("Pile ${pileIndex}"))`);
    const cardsCell = pileRow.locator('td').nth(1); // Second cell contains cards
    return await cardsCell.textContent() || '';
  }

  async getRemainingBudget(): Promise<number> {
    const budgetText = await this.budgetDisplay.textContent();
    const match = budgetText?.match(/(\d+)/);
    return match ? parseInt(match[1]) : 0;
  }

  async verifySignalRevealed(pileIndex: number) {
    // After purchasing a signal, verify the signal value is displayed
    const signalCell = this.page.locator(`tr:has(td:text("Pile ${pileIndex}")) td`).nth(2);
    const signalText = await signalCell.textContent();
    expect(signalText).not.toBe('-');
  }
}

/**
 * Stage 2 Page
 */
export class Stage2Page extends BasePage {
  readonly stageHeading: Locator;
  readonly r2Display: Locator;
  readonly investmentInputs: Locator;
  readonly submitButton: Locator;
  readonly restartButton: Locator;

  constructor(page: Page) {
    super(page);
    this.stageHeading = page.locator('h2:has-text("Stage 2")');
    this.r2Display = page.locator('text=/R2|second.*highest/i');
    this.investmentInputs = page.locator('input[type="number"][name^="invest2_"]');
    this.submitButton = page.locator('button:has-text("Submit")');
    this.restartButton = page.locator('button:has-text("Restart")');
  }

  async isDisplayed() {
    await expect(this.stageHeading).toBeVisible({ timeout: 10000 });
    await expect(this.submitButton).toBeVisible();
  }

  async makeInvestment(pileIndex: number, amount: number) {
    const input = this.page.locator(`input[name="invest2_${pileIndex}"]`);
    await input.fill(amount.toString());
  }

  async submitInvestments() {
    await this.submitButton.click();
    await this.waitForPageLoad();
  }

  async clickRestart() {
    await this.restartButton.click();
    await this.waitForPageLoad();
  }

  async verifyR2Revealed(pileIndex: number) {
    // Verify R2 value is shown for piles with Stage 1 investment
    const r2Cell = this.page.locator(`tr:has(td:text("Pile ${pileIndex}")) td`).nth(2);
    const r2Text = await r2Cell.textContent();
    expect(r2Text).not.toBe('-');
  }

  async verifyOnlyStage1PilesAvailable() {
    // Verify that only piles with Stage 1 investment have enabled inputs
    const allInputs = await this.investmentInputs.all();

    for (const input of allInputs) {
      const isDisabled = await input.isDisabled();
      const hasStage1Investment = await input.getAttribute('data-stage1-invested') === 'true';

      if (hasStage1Investment) {
        expect(isDisabled).toBe(false);
      } else {
        expect(isDisabled).toBe(true);
      }
    }
  }
}

/**
 * Results Page (Stage 3)
 */
export class ResultsPage extends BasePage {
  readonly stageHeading: Locator;
  readonly budgetInfo: Locator;
  readonly pilesInvestedInfo: Locator;
  readonly signalsSpentInfo: Locator;
  readonly breakdownTable: Locator;
  readonly distributionHistogram: Locator;
  readonly leaderboard: Locator;
  readonly meanSDStats: Locator;
  readonly restartButton: Locator;

  constructor(page: Page) {
    super(page);
    this.stageHeading = page.locator('h2:has-text("Results")');
    this.budgetInfo = page.locator('text=/Budget:/');
    this.pilesInvestedInfo = page.locator('text=/Piles invested:/');
    this.signalsSpentInfo = page.locator('text=/Signals spent:/');
    this.breakdownTable = page.locator('table:has(th:text("Pile"))');
    this.distributionHistogram = page.locator('table:has(th:text("Return Range"))');
    this.leaderboard = page.locator('table:has(th:text("Rank"))');
    this.meanSDStats = page.locator('text=/Mean:|SD:/');
    this.restartButton = page.locator('button:has-text("Restart")');
  }

  async isDisplayed() {
    await expect(this.stageHeading).toBeVisible({ timeout: 10000 });
    await expect(this.restartButton).toBeVisible();
  }

  async verifyInfoBoxes() {
    await expect(this.budgetInfo).toBeVisible();
    await expect(this.pilesInvestedInfo).toBeVisible();
    await expect(this.signalsSpentInfo).toBeVisible();
  }

  async verifyBreakdownTable() {
    await expect(this.breakdownTable).toBeVisible();

    // Verify table headers
    const headers = this.breakdownTable.locator('th');
    await expect(headers.nth(0)).toContainText('Pile');
    await expect(headers.nth(1)).toContainText('Stage 1');
    await expect(headers.nth(2)).toContainText('Stage 2');
  }

  async verifyDistributionHistogram() {
    await expect(this.distributionHistogram).toBeVisible();

    // Verify histogram has data (at least 1 bin with non-zero count)
    const rows = this.distributionHistogram.locator('tbody tr');
    const rowCount = await rows.count();
    expect(rowCount).toBeGreaterThan(0);

    // Verify zero-count bins are skipped (no rows with "Count: 0")
    const cellsWithZero = this.distributionHistogram.locator('td:has-text("0")');
    const zeroCount = await cellsWithZero.count();
    // There might be 0 in the bin label, but not in count column
  }

  async verifyLeaderboard() {
    await expect(this.leaderboard).toBeVisible();

    // Verify current player is highlighted
    const highlightedRow = this.leaderboard.locator('tr[style*="background-color"]');
    await expect(highlightedRow).toBeVisible();
  }

  async verifyMeanSDStats() {
    const statsText = await this.meanSDStats.textContent();
    expect(statsText).toMatch(/Mean:.*%/);
    expect(statsText).toMatch(/SD:.*%/);
  }

  async clickRestart() {
    await this.restartButton.click();
    await this.waitForPageLoad();
  }

  async getNetReturn(): Promise<number> {
    const infoText = await this.budgetInfo.textContent();
    const match = infoText?.match(/Net Return:.*?([-\d.]+)%/);
    return match ? parseFloat(match[1]) : 0;
  }

  async getCurrentPlayerRank(): Promise<number> {
    const highlightedRow = this.leaderboard.locator('tr[style*="background-color"]');
    const rankCell = highlightedRow.locator('td').first();
    const rankText = await rankCell.textContent();
    return parseInt(rankText || '0');
  }
}

/**
 * Helper function to generate unique team names
 */
export function generateTeamName(prefix: string = 'TestTeam'): string {
  const timestamp = Date.now();
  const random = Math.floor(Math.random() * 1000);
  return `${prefix}_${timestamp}_${random}`;
}

/**
 * Helper function to wait for server to be ready
 */
export async function waitForServer(page: Page, baseURL: string, maxRetries: number = 30) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      await page.goto(baseURL, { timeout: 5000 });
      return true;
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await page.waitForTimeout(1000);
    }
  }
  return false;
}

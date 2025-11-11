import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for VC Card Game testing
 * Supports both local and Railway deployment testing
 */
export default defineConfig({
  // Test directory
  testDir: './e2e',

  // Maximum time one test can run
  timeout: 60 * 1000, // 60 seconds

  // Test execution settings
  fullyParallel: false, // Run tests sequentially to avoid database conflicts
  forbidOnly: !!process.env.CI, // Fail build on .only() in CI
  retries: process.env.CI ? 2 : 0, // Retry failed tests in CI
  workers: 1, // Single worker to prevent database race conditions

  // Output folder for test artifacts (separate from HTML report)
  outputDir: 'test-results/artifacts',

  // Reporter configuration
  reporter: [
    ['html', { outputFolder: 'test-results/html-report' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['list'], // Console output
  ],

  // Shared settings for all projects
  use: {
    // Base URL - override with TEST_URL environment variable
    baseURL: process.env.TEST_URL || 'http://localhost:8765',

    // Collect trace on failure for debugging
    trace: 'retain-on-failure',

    // Screenshot on failure
    screenshot: 'only-on-failure',

    // Video on failure
    video: 'retain-on-failure',

    // Navigation timeout
    navigationTimeout: 30 * 1000,

    // Action timeout
    actionTimeout: 10 * 1000,
  },

  // Configure projects for different browsers
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
      },
    },

    // Uncomment for cross-browser testing
    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },

    // {
    //   name: 'webkit',
    //   use: { ...devices['Desktop Safari'] },
    // },

    // Mobile testing (optional)
    // {
    //   name: 'mobile-chrome',
    //   use: { ...devices['Pixel 5'] },
    // },
  ],

  // Web server configuration for local testing
  webServer: process.env.TEST_URL ? undefined : {
    command: 'python web_game.py',
    url: 'http://localhost:8765',
    timeout: 120 * 1000, // 2 minutes to start
    reuseExistingServer: !process.env.CI,
    stdout: 'pipe',
    stderr: 'pipe',
    cwd: '..',
  },
});

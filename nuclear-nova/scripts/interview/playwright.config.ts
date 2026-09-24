import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: '../../tests/browser', testMatch: '**/*.e2e.ts',
  timeout: 60_000, expect: { timeout: 10_000 }, workers: 1,
  use: { baseURL: process.env.INTERVIEW_URL || 'http://127.0.0.1:4323', viewport: { width: 1440, height: 1000 }, trace: 'retain-on-failure' },
  outputDir: '../../test-results/interview', reporter: 'list',
});

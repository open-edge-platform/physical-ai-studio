import {defineConfig} from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  use: {baseURL: 'http://127.0.0.1:4173', viewport: {width: 1200, height: 675}},
  webServer: {
    command: 'npm run preview -- --port 4173',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});

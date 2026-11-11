import { HttpResponse } from 'msw';

import { getMockedRemoteTrainer, getMockedRemoteTrainerHealth } from '../src/test-utils/mocks/mock-remote-trainer';
import { expect, http, test } from './fixtures';

test('expanded healthy training targets scroll inside settings, not below the footer', async ({ page, network }) => {
    await page.setViewportSize({ width: 1280, height: 600 });
    network.use(
        http.get('/api/remote-trainers', () => HttpResponse.json([getMockedRemoteTrainer()])),
        http.get('/api/remote-trainers/{remote_trainer_id}/health', () =>
            HttpResponse.json(getMockedRemoteTrainerHealth())
        ),
        http.get('/api/remote-servers/feature-status', () => HttpResponse.json({ network_exposed: false })),
        http.get('/api/jobs', () => HttpResponse.json([]))
    );
    await page.goto('/settings/training-targets');
    await expect(page.getByText('NVIDIA A100', { exact: false })).toBeVisible();

    const panel = page.getByRole('tabpanel');
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollHeight - window.innerHeight)).toBe(0);
    expect(await panel.evaluate((element) => element.scrollHeight)).toBeGreaterThan(
        await panel.evaluate((element) => element.clientHeight)
    );

    await panel.evaluate((element) => {
        element.scrollTop = element.scrollHeight;
    });
    await expect(page.getByText('Last checked', { exact: true })).toBeInViewport();
    await expect(page.getByRole('button', { name: 'Logs', exact: true })).toBeInViewport();
    await page.mouse.move(1100, 400);
    await page.mouse.wheel(0, 1000);
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0);
});

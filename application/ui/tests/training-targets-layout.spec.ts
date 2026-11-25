import { HttpResponse } from 'msw';

import { getMockedRemoteTrainer, getMockedRemoteTrainerHealth } from '../src/test-utils/mocks/mock-remote-trainer';
import { expect, http, test } from './fixtures';

test('expanded healthy training targets scroll inside settings, not below the footer', async ({ page, network }) => {
    await page.setViewportSize({ width: 1280, height: 600 });
    const trainer = getMockedRemoteTrainer();
    network.use(
        http.get('/api/remote-trainers', () => HttpResponse.json([trainer])),
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
    const detailId = await panel
        .getByRole('button', { name: `Show details for ${trainer.name}`, exact: true })
        .getAttribute('aria-controls');
    expect(detailId).toBeTruthy();
    const detail = page.locator(`[id="${detailId}"]`);
    const footer = page.getByRole('contentinfo');
    await expect(async () => {
        const detailBottom = await detail.evaluate((element) => element.getBoundingClientRect().bottom);
        const panelBottom = await panel.evaluate((element) => element.getBoundingClientRect().bottom);
        const footerTop = await footer.evaluate((element) => element.getBoundingClientRect().top);

        expect(detailBottom).toBeLessThanOrEqual(panelBottom);
        expect(detailBottom).toBeLessThanOrEqual(footerTop);
    }).toPass();
    await expect(page.getByText('Last checked', { exact: true })).toBeInViewport();
    await expect(footer.getByRole('button', { name: 'Logs', exact: true })).toBeInViewport();
    await page.mouse.move(1100, 400);
    await page.mouse.wheel(0, 1000);
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0);
});

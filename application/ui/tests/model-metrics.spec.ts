// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Locator } from '@playwright/test';
import { HttpResponse } from 'msw';

import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { expect, http, test } from './fixtures';

const projectId = 'project-1';
const model = getMockedModel({ dataset_id: null });
const metrics = [
    { step: 49, train_loss: 0.8, 'lr-AdamWBF16Kahan': 0.00001 },
    {
        step: 65,
        val_loss: 0.9,
        val_action_error: 4,
        'val_action_error_wrist.pos': 20,
        'val_action_error_shoulder.pos': 2,
    },
    { step: 99, train_loss: 0.6, 'lr-AdamWBF16Kahan': 0.00002 },
    {
        step: 131,
        val_loss: 0.7,
        val_action_error: 3.8,
        'val_action_error_wrist.pos': 19,
        'val_action_error_shoulder.pos': 1.8,
    },
    { step: 149, train_loss: 0.4, 'lr-AdamWBF16Kahan': 0.00003 },
    {
        step: 197,
        val_loss: 0.5,
        val_action_error: 3.5,
        'val_action_error_wrist.pos': 18,
        'val_action_error_shoulder.pos': 1.5,
    },
];

const curve = (graph: Locator, color: string) => graph.locator(`path.recharts-line-curve[stroke="${color}"]`);

const pointOnPath = (path: Locator, fraction: number) =>
    path.evaluate((element, ratio) => {
        const svgPath = element as SVGPathElement;
        const point = svgPath.getPointAtLength(svgPath.getTotalLength() * ratio);
        const screenPoint = new DOMPoint(point.x, point.y).matrixTransform(svgPath.getScreenCTM() ?? undefined);
        return { x: screenPoint.x, y: screenPoint.y };
    }, fraction);

test.beforeEach(async ({ page, network }) => {
    await page.setViewportSize({ width: 1440, height: 1400 });
    network.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([model])),
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.get('/api/models/{model_id}', () =>
            HttpResponse.json({ model, exports: [], training_summary: null, hparams: null })
        )
    );
    await page.route('**/api/models/*/metrics', (route) =>
        route.fulfill({
            contentType: 'text/event-stream',
            body: metrics.map((row) => `data: ${JSON.stringify(row)}\n\n`).join('') + 'data: DONE\n\n',
        })
    );
    await page.goto(`/projects/${projectId}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    await page.getByRole('tab', { name: 'Model Metrics', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Validation action error graph' })).toBeVisible();
});

test('emphasizes the mean, highlights joints, and toggles lines with tooltip values', async ({ page }) => {
    const graph = page.getByRole('region', { name: 'Validation action error graph' });
    const mean = curve(graph, 'var(--energy-blue)');
    const wrist = curve(graph, 'var(--spectrum-global-color-orange-500)');
    const legend = graph.getByRole('button', { name: 'wrist', exact: true });

    await expect(mean).toHaveAttribute('stroke-width', '4');
    await expect(wrist).toHaveAttribute('stroke-opacity', '0.25');
    await legend.hover();
    await expect(wrist).toHaveAttribute('stroke-opacity', '1');
    await graph.getByRole('button', { name: 'Mean', exact: true }).hover();
    await expect(wrist).toHaveAttribute('stroke-opacity', '0.25');

    // Hover an actual point along the wide, invisible hit target rather than dispatching a synthetic event.
    const point = await pointOnPath(wrist, 0.2);
    await page.mouse.move(point.x, point.y);
    await expect(wrist).toHaveAttribute('stroke-opacity', '1');

    const tooltip = graph.getByRole('tooltip');
    await expect(tooltip).toBeVisible();
    await tooltip.getByRole('button', { name: 'Highlight shoulder' }).hover();
    await expect(curve(graph, 'var(--spectrum-global-color-magenta-500)')).toHaveAttribute('stroke-opacity', '1');
    await expect(wrist).toHaveAttribute('stroke-opacity', '0.25');

    await legend.click();
    await expect(legend).toHaveAttribute('aria-pressed', 'false');
    await expect(wrist).toHaveCount(0);
    await mean.hover({ position: { x: 10, y: 0 }, force: true });
    await expect(tooltip.getByRole('button', { name: 'Highlight wrist' })).toHaveCount(0);
    await expect(tooltip.getByRole('button', { name: 'Highlight Mean' })).toBeVisible();
    await legend.click();
    await expect(wrist).toHaveCount(1);
});

test('renders non-AdamW learning rates and supports hiding and restoring the only line', async ({ page }) => {
    const graph = page.getByRole('region', { name: 'Learning rate graph' });
    const legend = graph.getByRole('button', { name: 'AdamWBF16Kahan', exact: true });
    const line = curve(graph, 'var(--energy-blue)');

    await expect(line).toHaveCount(1);
    await legend.click();
    await expect(line).toHaveCount(0);
    await legend.click();
    await expect(line).toHaveCount(1);
});

test('expands plots to fill a multi-column row when only one graph has a legend', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1400 });
    const training = page.getByRole('region', { name: 'Training loss graph' });
    const validation = page.getByRole('region', { name: 'Validation loss graph' });
    const action = page.getByRole('region', { name: 'Validation action error graph' });

    await expect(async () => {
        const regions = await Promise.all([training, validation, action].map((graph) => graph.boundingBox()));
        const plots = await Promise.all(
            [training, validation, action].map((graph) => graph.locator('.recharts-responsive-container').boundingBox())
        );
        if (regions.some((box) => !box) || plots.some((box) => !box)) throw new Error('Expected rendered graphs');

        const tops = regions.map((box) => box?.y ?? 0);
        const bottoms = plots.map((box) => (box?.y ?? 0) + (box?.height ?? 0));
        expect(Math.max(...tops) - Math.min(...tops)).toBeLessThan(1);
        expect(Math.max(...bottoms) - Math.min(...bottoms)).toBeLessThan(1);
        expect(plots[0]?.height).toBeGreaterThan(300);
    }).toPass();
});

test('snaps synchronized cursors and tooltip values to each graph’s nearest recorded step', async ({ page }) => {
    const training = page.getByRole('region', { name: 'Training loss graph' });
    const validation = page.getByRole('region', { name: 'Validation loss graph' });
    const action = page.getByRole('region', { name: 'Validation action error graph' });
    const learningRate = page.getByRole('region', { name: 'Learning rate graph' });
    // The training curve's midpoint is the sample at step 99.
    await expect(training.locator('.recharts-area-curve')).toHaveCount(1);
    const point = await pointOnPath(training.locator('.recharts-area-curve'), 0.5);
    await page.mouse.move(point.x, point.y);

    await expect(training.getByRole('tooltip')).toContainText('Step: 99');
    await expect(training.getByRole('tooltip')).toContainText('Loss: 0.6');
    await expect(validation.getByRole('tooltip')).toContainText('Step: 131');
    await expect(validation.getByRole('tooltip')).toContainText('Loss: 0.7');
    await expect(action.getByRole('tooltip')).toContainText('Step: 131');
    await expect(action.getByRole('tooltip')).toContainText('Mean: 3.8');
    await expect(learningRate.getByRole('tooltip')).toContainText('Step: 99');
    await expect(learningRate.getByRole('tooltip')).toContainText('AdamWBF16Kahan: 0.00002');

    for (const graph of [training, validation, action, learningRate]) {
        const cursor = graph.locator('.recharts-tooltip-cursor');
        await expect(cursor).toHaveCount(1);
        // A synchronized cursor must sit on its own selected data point, not at the source's x position.
        const cursorBounds = await cursor.boundingBox();
        const dotBounds = await graph.locator('.recharts-active-dot circle').first().boundingBox();
        expect(cursorBounds).not.toBeNull();
        expect(dotBounds).not.toBeNull();
        expect(Math.abs((cursorBounds?.x ?? 0) - ((dotBounds?.x ?? 0) + (dotBounds?.width ?? 0) / 2))).toBeLessThan(1);
    }

    // Sync also works in reverse, clamping shorter series to their last sample.
    const lastValidationPoint = await pointOnPath(validation.locator('.recharts-area-curve'), 1);
    await page.mouse.move(lastValidationPoint.x - 1, lastValidationPoint.y);
    await expect(validation.getByRole('tooltip')).toContainText('Step: 197');
    await expect(training.getByRole('tooltip')).toContainText('Step: 149');
    await expect(training.getByRole('tooltip')).toContainText('Loss: 0.4');
    await expect(action.getByRole('tooltip')).toContainText('Step: 197');
    await expect(learningRate.getByRole('tooltip')).toContainText('Step: 149');

    await page.mouse.move(0, 0);
    for (const graph of [training, validation, action, learningRate]) {
        await expect(graph.getByRole('tooltip')).toBeHidden();
        await expect(graph.locator('.recharts-tooltip-cursor')).toHaveCount(0);
    }
});

for (const title of ['Learning rate', 'Validation action error']) {
    test(`${title} tooltip follows the selected step across the plot`, async ({ page }) => {
        const graph = page.getByRole('region', { name: `${title} graph` });
        const line = curve(graph, 'var(--energy-blue)');
        const firstPoint = await pointOnPath(line, 0.1);
        await page.mouse.move(firstPoint.x, firstPoint.y);

        const tooltip = graph.getByRole('tooltip');
        await expect(tooltip).toBeVisible();
        const firstPosition = await tooltip.boundingBox();
        expect(firstPosition).not.toBeNull();

        const lastPoint = await pointOnPath(line, 0.9);
        await page.mouse.move(lastPoint.x, lastPoint.y);
        await expect(tooltip).toContainText(title === 'Learning rate' ? 'Step: 149' : 'Step: 197');
        await expect(async () => {
            const lastPosition = await tooltip.boundingBox();
            expect(lastPosition).not.toBeNull();
            expect(Math.abs((lastPosition?.x ?? 0) - (firstPosition?.x ?? 0))).toBeGreaterThan(20);
        }).toPass();
    });
}

test('opens chart-specific header help with the keyboard without shifting the plots', async ({ page }) => {
    await expect(page.getByText('Validation metrics are a sanity check')).toHaveCount(0);
    for (const [title, description] of [
        ['Training loss', 'optimization loss on training demonstrations'],
        ['Validation loss', 'Loss on held-out demonstrations, not task success'],
        ['Validation action error', 'in dataset units'],
        ['Learning rate', 'Separate lines represent optimizers or parameter groups'],
    ]) {
        const plot = page.getByRole('region', { name: `${title} graph` }).locator('.recharts-responsive-container');
        const before = await plot.boundingBox();
        const button = page.getByRole('button', { name: `${title} information` });
        await button.focus();
        await button.press('Enter');

        const dialog = page.getByRole('dialog', { name: title, exact: true });
        await expect(dialog).toBeVisible();
        await expect(dialog).toContainText(description);
        expect(await plot.boundingBox()).toEqual(before);
        await page.keyboard.press('Escape');
        await expect(dialog).toBeHidden();
        await expect(button).toBeFocused();
    }
});

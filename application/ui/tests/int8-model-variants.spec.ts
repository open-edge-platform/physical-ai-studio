import { HttpResponse } from 'msw';

import { getMockedExportVariant } from '../src/test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { getMockedModelDetailResponse } from '../src/test-utils/mocks/mock-model-detail-response';
import { getMockedOptimizeModelJob } from '../src/test-utils/mocks/mock-optimize-model-job';
import { expect, http, test } from './fixtures';

for (const variant of ['int8', 'original'] as const) {
    test(`removes the ${variant} model only after confirmation and refreshes available cards`, async ({
        page,
        network,
    }) => {
        const model = getMockedModel({ available_backends: ['openvino'], dataset_id: null });
        const detail = getMockedModelDetailResponse({ model, available_actions: [] });
        detail.exports[0].variants = [getMockedExportVariant(), getMockedExportVariant({ id: 'int8' })];
        let removed = false;
        await page.routeWebSocket('**/api/jobs/ws', () => {});
        await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
        network.use(
            http.get('/api/projects/{project_id}/models', () => HttpResponse.json([detail.model])),
            http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
            http.get('/api/jobs', () => HttpResponse.json([])),
            http.get('/api/system/devices/inference', () => HttpResponse.json([])),
            http.delete('/api/models/{model_id}/exports/{backend}/variants/{variant}', ({ params }) => {
                expect(params.variant).toBe(variant);
                removed = true;
                detail.exports[0].variants = [getMockedExportVariant({ id: variant === 'int8' ? 'original' : 'int8' })];
                detail.available_actions = variant === 'int8' ? [{ backend: 'openvino', operation: 'int8_sym' }] : [];
                return new HttpResponse(null, { status: 204 });
            })
        );
        await page.goto(`/projects/${model.project_id}/models`);
        await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
        const label = variant === 'int8' ? 'INT8' : 'Original';
        const card = page.getByRole('region', {
            name: variant === 'original' ? 'OpenVINO model format' : `OpenVINO · ${label}`,
        });
        await card.getByRole('button', { name: `Actions for OpenVINO ${label}` }).click();
        await expect(page.getByRole('menuitem', { name: /Remove (export|backend)/ })).toHaveCount(0);
        await page.getByRole('menuitem', { name: 'Remove model' }).click();
        const dialog = page.getByRole('dialog');
        await expect(dialog).toContainText('training checkpoint, training history, and other backends are kept');
        expect(removed).toBe(false);
        await dialog.getByRole('button', { name: 'Cancel' }).click();
        await expect(card).toBeVisible();
        await card.getByRole('button', { name: `Actions for OpenVINO ${label}` }).click();
        await page.getByRole('menuitem', { name: 'Remove model' }).click();
        await page.getByRole('dialog').getByRole('button', { name: 'Remove model', exact: true }).click();
        await expect(card).toHaveCount(0);
        await expect(
            page.getByRole('region', { name: variant === 'int8' ? 'OpenVINO model format' : 'OpenVINO · INT8' })
        ).toBeVisible();
        if (variant === 'int8') await expect(page.getByRole('button', { name: 'Add model format' })).toBeVisible();
        expect(removed).toBe(true);
    });
}

test('creates an INT8 export and selects it for download and inference', async ({ page, network }, testInfo) => {
    const model = getMockedModel({ available_backends: ['openvino'], dataset_id: null });
    const detail = getMockedModelDetailResponse({ model });
    let jobs: ReturnType<typeof getMockedOptimizeModelJob>[] = [];
    let complete = () => {};
    await page.routeWebSocket('**/api/jobs/ws', (socket) => {
        complete = () => {
            detail.exports[0].variants?.push(getMockedExportVariant({ id: 'int8', weight_compression: 'int8_sym' }));
            detail.available_actions = [];
            jobs = [getMockedOptimizeModelJob({ status: 'completed' })];
            socket.send(JSON.stringify({ event: 'JOB_UPDATE', data: jobs[0] }));
        };
    });
    await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
    network.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([detail.model])),
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/policies/backends', () => HttpResponse.json({ act: ['openvino'] })),
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.get('/api/system/devices/inference', () =>
            HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'openvino', device: 'CPU' }])
        ),
        http.post('/api/models/{model_id}/optimize', () => {
            jobs = [getMockedOptimizeModelJob()];
            return HttpResponse.json(jobs[0], { status: 202 });
        })
    );
    await page.goto(`/projects/${model.project_id}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    await expect(page.getByRole('button', { name: 'Add model format' })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('available-model-cards.png'), fullPage: true });
    await page.getByRole('button', { name: 'Add model format' }).click();
    await page.getByRole('button', { name: 'Optimize to INT8' }).click();
    await expect(page.getByRole('dialog')).toHaveCount(0);
    await expect(page.getByRole('region', { name: 'OpenVINO · INT8' }).getByRole('status')).toContainText(
        'Export queued'
    );
    complete();
    const card = page.getByRole('region', { name: 'OpenVINO · INT8' });
    await expect(card).toBeVisible();
    await expect(page.getByRole('region', { name: 'OpenVINO model format' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Add model format' })).toHaveCount(0);
    await card.getByRole('button', { name: 'Actions for OpenVINO INT8' }).click();
    await expect(page.getByRole('menuitem', { name: 'Download', exact: true })).toHaveAttribute('href', /variant=int8/);
    await page.keyboard.press('Escape');
    await page.screenshot({ path: testInfo.outputPath('int8-model-cards.png'), fullPage: true });
    await card.getByRole('button', { name: 'Run', exact: true }).click();
    await expect(page).toHaveURL(/inference\/openvino\?device=CPU&variant=int8$/);
});

test('adds a missing backend from a checkpoint without placeholder cards', async ({ page, network }, testInfo) => {
    const model = getMockedModel({ available_backends: [], dataset_id: null });
    const detail = getMockedModelDetailResponse({
        model,
        exports: [],
        available_actions: [{ backend: 'torch', operation: 'export' }],
    });
    const payload = {
        model_id: model.id!,
        backend: 'torch',
        operation: 'export',
        cancellation_requested: false,
    } as const;
    let jobs: ReturnType<typeof getMockedOptimizeModelJob>[] = [];
    let complete = () => {};
    await page.routeWebSocket('**/api/jobs/ws', (socket) => {
        complete = () => {
            detail.exports = [{ type: 'torch', size_bytes: 4096, file_count: 3, variants: [getMockedExportVariant()] }];
            detail.available_actions = [];
            detail.model = { ...model, available_backends: ['torch'] };
            jobs = [getMockedOptimizeModelJob({ payload, status: 'completed' })];
            socket.send(JSON.stringify({ event: 'JOB_UPDATE', data: jobs[0] }));
        };
    });
    await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
    network.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([detail.model])),
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.get('/api/system/devices/inference', () =>
            HttpResponse.json([{ type: 'cpu', name: 'CPU', backend: 'torch', device: 'cpu' }])
        ),
        http.post('/api/models/{model_id}/exports/{backend}', () => {
            jobs = [getMockedOptimizeModelJob({ payload })];
            return HttpResponse.json(jobs[0], { status: 202 });
        })
    );
    await page.goto(`/projects/${model.project_id}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    await expect(page.getByText('Unavailable', { exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Add model format' }).click();
    await page.getByRole('dialog').getByRole('button', { name: 'Export model', exact: true }).click();
    await expect(page.getByRole('dialog')).toHaveCount(0);
    await expect(page.getByRole('region', { name: 'PyTorch model format' }).getByRole('status')).toContainText(
        'Export queued'
    );
    complete();
    const card = page.getByRole('region', { name: 'PyTorch model format' });
    await expect(card).toBeVisible();
    await expect(page.getByRole('heading', { name: 'OpenVINO', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Add model format' })).toHaveCount(0);
    await page.screenshot({ path: testInfo.outputPath('checkpoint-export-card.png'), fullPage: true });
    await card.getByRole('button', { name: 'Run', exact: true }).click();
    await expect(page).toHaveURL(/inference\/torch\?device=cpu&variant=original$/);
});

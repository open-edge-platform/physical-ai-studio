import { HttpResponse } from 'msw';

import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { getMockedBackendExportDetail, getMockedModelDetail } from '../src/test-utils/mocks/mock-model-detail';
import { getMockedModelExportJob } from '../src/test-utils/mocks/mock-model-export-job';
import { expect, http, test } from './fixtures';

test('selects a backend in a dialog and hands export progress over to its card', async ({
    page,
    network,
}, testInfo) => {
    const model = getMockedModel({ available_backends: ['torch'], dataset_id: null });
    const detail = getMockedModelDetail({ model, exports: [getMockedBackendExportDetail()] });
    const payload = {
        model_id: model.id!,
        backend: 'onnx',
        operation: 'export',
        cancellation_requested: false,
    } as const;
    let jobs: ReturnType<typeof getMockedModelExportJob>[] = [];
    let accept = () => {};
    const accepted = new Promise<void>((resolve) => {
        accept = resolve;
    });
    let updateJob = (_status: 'running' | 'completed') => {};
    await page.routeWebSocket('**/api/jobs/ws', (socket) => {
        updateJob = (status) => {
            if (status === 'completed') {
                detail.exports.push(getMockedBackendExportDetail({ type: 'onnx' }));
                detail.available_actions = [{ backend: 'openvino', operation: 'export' }];
                detail.model = { ...model, available_backends: ['torch', 'onnx'] };
            }
            jobs = [getMockedModelExportJob({ payload, status, progress: status === 'running' ? 50 : 100 })];
            socket.send(JSON.stringify({ event: 'JOB_UPDATE', data: jobs[0] }));
        };
    });
    await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
    network.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([detail.model])),
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.post('/api/models/{model_id}/exports/{backend}', async ({ params }) => {
            expect(params.backend).toBe('onnx');
            await accepted;
            jobs = [getMockedModelExportJob({ payload })];
            return HttpResponse.json(jobs[0], { status: 202 });
        })
    );
    await page.goto(`/projects/${model.project_id}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    const addVariant = page.getByRole('button', { name: 'Add variant' });
    await expect(addVariant).toBeEnabled();
    const icon = await addVariant.locator('svg').boundingBox();
    const title = await addVariant.getByText('Add variant', { exact: true }).boundingBox();
    expect(icon).not.toBeNull();
    expect(title).not.toBeNull();
    expect(Math.abs(icon!.y + icon!.height / 2 - title!.y - title!.height / 2)).toBeLessThan(2);
    expect(icon!.x + icon!.width).toBeLessThan(title!.x);

    await addVariant.click();
    const dialog = page.getByRole('dialog', { name: 'Add variant' });
    await expect(dialog.getByRole('radio', { name: /PyTorch/ })).toHaveCount(0);
    await dialog.getByRole('radio', { name: /ONNX/ }).check();
    expect(jobs).toHaveLength(0);
    await page.screenshot({ path: testInfo.outputPath('add-variant-dialog.png'), fullPage: true });
    await dialog.getByRole('button', { name: 'Export variant', exact: true }).click();
    try {
        await expect(dialog.getByRole('status')).toContainText('Requesting export');
        await expect(dialog.getByRole('button', { name: /Export variant/ })).toBeDisabled();
        await expect(dialog.getByRole('radio', { name: /OpenVINO/ })).toBeDisabled();
        await expect(dialog.getByRole('button', { name: 'Cancel' })).toBeDisabled();
        await page.keyboard.press('Escape');
        await expect(dialog).toBeVisible();
    } finally {
        accept();
    }
    await expect(dialog).toHaveCount(0);
    const card = page.getByRole('region', { name: 'ONNX variant' });
    await expect(card.getByRole('status')).toContainText('Export queued');
    await expect(card.getByRole('progressbar')).toBeVisible();
    updateJob('running');
    await expect(card.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '50');
    updateJob('completed');
    await expect(card.getByRole('button', { name: 'Download ONNX export' })).toBeVisible();
    await expect(card.getByRole('progressbar')).toHaveCount(0);
    await expect(addVariant).toBeEnabled();
});

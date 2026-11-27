import { HttpResponse } from 'msw';

import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { getMockedBackendExportDetail, getMockedModelDetail } from '../src/test-utils/mocks/mock-model-detail';
import { getMockedModelExportJob } from '../src/test-utils/mocks/mock-model-export-job';
import { expect, http, test } from './fixtures';

test('removes a canceled export card and keeps it hidden after a refresh', async ({ page, network }) => {
    const model = getMockedModel({ available_backends: ['torch'], dataset_id: null });
    const detail = getMockedModelDetail({ model, exports: [getMockedBackendExportDetail()] });
    const payload = {
        model_id: model.id!,
        backend: 'onnx',
        operation: 'export',
        cancellation_requested: false,
    } as const;
    let job = getMockedModelExportJob({ payload, status: 'running' });
    await page.routeWebSocket('**/api/jobs/ws', () => {});
    await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
    network.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([model])),
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/jobs', () => HttpResponse.json([job])),
        http.post('/api/jobs/{job_id}:interrupt', () => {
            job = { ...job, status: 'canceled' };
            return HttpResponse.json(job);
        })
    );
    await page.goto(`/projects/${model.project_id}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    const card = page.getByRole('region', { name: 'ONNX model format' });
    await card.getByRole('button', { name: 'Cancel export' }).click();
    await expect(card).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Retry export' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Download PyTorch export' })).toBeVisible();

    await page.reload();
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    await expect(page.getByRole('button', { name: 'Add model format' })).toBeEnabled();
    await expect(card).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Retry export' })).toHaveCount(0);
    await page.getByRole('button', { name: 'Add model format' }).click();
    await expect(page.getByRole('dialog').getByRole('radio', { name: /ONNX/ })).toBeEnabled();
});

test('selects a model format in a dialog and hands export progress over to its card', async ({
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
            jobs = [
                getMockedModelExportJob({
                    payload,
                    status,
                    progress: status === 'running' ? 50 : 100,
                    message: status === 'running' ? 'Exporting ONNX checkpoint' : 'ONNX export created',
                }),
            ];
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
            jobs = [getMockedModelExportJob({ payload, message: 'Waiting to export ONNX' })];
            return HttpResponse.json(jobs[0], { status: 202 });
        })
    );
    await page.goto(`/projects/${model.project_id}/models`);
    await page.getByRole('button', { name: `Show details for ${model.name}` }).click();
    const addModelFormat = page.getByRole('button', { name: 'Add model format' });
    await expect(addModelFormat).toBeEnabled();
    await expect(addModelFormat.getByText('Export to another inference backend')).toBeVisible();
    const icon = await addModelFormat.locator('svg').boundingBox();
    const title = await addModelFormat.getByText('Add model format', { exact: true }).boundingBox();
    expect(icon).not.toBeNull();
    expect(title).not.toBeNull();
    expect(Math.abs(icon!.y + icon!.height / 2 - title!.y - title!.height / 2)).toBeLessThan(2);
    expect(icon!.x + icon!.width).toBeLessThan(title!.x);

    await addModelFormat.click();
    const dialog = page.getByRole('dialog', { name: 'Add model format' });
    await expect(dialog.getByRole('radiogroup', { name: 'Model format' })).toBeVisible();
    await expect(dialog.getByRole('radio', { name: /PyTorch/ })).toHaveCount(0);
    await dialog.getByRole('radio', { name: /ONNX/ }).check();
    expect(jobs).toHaveLength(0);
    await page.screenshot({ path: testInfo.outputPath('add-model-format-dialog.png'), fullPage: true });
    await dialog.getByRole('button', { name: 'Export model', exact: true }).click();
    try {
        await expect(dialog.getByRole('status')).toContainText('Requesting export');
        await expect(dialog.getByRole('button', { name: /Export model/ })).toBeDisabled();
        await expect(dialog.getByRole('radio', { name: /OpenVINO/ })).toBeDisabled();
        await expect(dialog.getByRole('button', { name: 'Cancel' })).toBeDisabled();
        await page.keyboard.press('Escape');
        await expect(dialog).toBeVisible();
    } finally {
        accept();
    }
    await expect(dialog).toHaveCount(0);
    const card = page.getByRole('region', { name: 'ONNX model format' });
    const border = card.locator('svg[aria-hidden="true"] > rect');
    await expect(card.getByRole('status')).toContainText('Export queued');
    await expect(card).toHaveAttribute('aria-busy', 'true');
    await expect(card.getByRole('progressbar')).toHaveCount(0);
    await expect(border).toBeVisible();
    await expect(border).toHaveCSS('stroke-dasharray', '1px, 7px');
    await expect(border).not.toHaveCSS('animation-name', 'none');
    updateJob('running');
    await expect(card.getByRole('status')).toContainText('Exporting model');
    await expect(card.getByRole('button', { name: 'Cancel export' })).toBeEnabled();
    await page.screenshot({ path: testInfo.outputPath('export-in-background.png'), fullPage: true });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await expect(border).toHaveCSS('animation-name', 'none');
    await expect(border).toBeVisible();
    updateJob('completed');
    await expect(card.getByRole('button', { name: 'Download ONNX export' })).toBeVisible();
    await expect(border).toHaveCount(0);
    await expect(card).toHaveAttribute('aria-busy', 'false');
    await expect(addModelFormat).toBeEnabled();
});

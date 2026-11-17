import { HttpResponse } from 'msw';

import { getMockedExportVariant } from '../src/test-utils/mocks/mock-export-variant';
import { getMockedModel } from '../src/test-utils/mocks/mock-model';
import { getMockedModelDetailResponse } from '../src/test-utils/mocks/mock-model-detail-response';
import { getMockedOptimizeModelJob } from '../src/test-utils/mocks/mock-optimize-model-job';
import { expect, http, test } from './fixtures';

test('creates an INT8 export and selects it for download and inference', async ({ page, network }) => {
    const model = getMockedModel({ available_backends: ['openvino'], dataset_id: null });
    const detail = getMockedModelDetailResponse({ model });
    let jobs: ReturnType<typeof getMockedOptimizeModelJob>[] = [];
    let complete = () => {};
    await page.routeWebSocket('**/api/jobs/ws', (socket) => {
        complete = () => {
            detail.exports[0].variants?.push(getMockedExportVariant({ id: 'int8', weight_compression: 'int8_sym' }));
            jobs = [getMockedOptimizeModelJob({ status: 'completed' })];
            socket.send(JSON.stringify({ event: 'JOB_UPDATE', data: jobs[0] }));
        };
    });
    await page.routeWebSocket('**/api/projects/*/runtime/ws', (socket) => socket.close());
    network.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([model])),
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
    await page.getByRole('button', { name: 'Optimize to INT8' }).click();
    await expect(page.getByRole('status')).toContainText('Optimization pending');
    complete();
    await page.getByRole('button', { name: /Variant/ }).click();
    await page.getByRole('option', { name: 'INT8' }).click();
    await expect(page.getByRole('button', { name: 'Download OpenVINO export', exact: true })).toHaveAttribute(
        'href',
        /variant=int8/
    );
    await page.getByRole('button', { name: 'Run model', exact: true }).click();
    const dialog = page.getByRole('dialog');
    await dialog.getByRole('button', { name: /Variant/ }).click();
    await page.getByRole('option', { name: 'INT8' }).click();
    await dialog.getByRole('button', { name: 'Start', exact: true }).click();
    await expect(page).toHaveURL(/inference\/openvino\?device=CPU&variant=int8$/);
});

import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaModelExportJob } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedExportVariant } from '../../../test-utils/mocks/mock-export-variant';
import { getMockedModelDetailResponse } from '../../../test-utils/mocks/mock-model-detail-response';
import { getMockedOptimizeModelJob } from '../../../test-utils/mocks/mock-optimize-model-job';
import { render } from '../../../test-utils/render';
import { ModelFormats } from './model-formats';

const detail = getMockedModelDetailResponse();

beforeEach(() => {
    server.use(
        http.get('/api/models/{model_id}', () => HttpResponse.json(detail)),
        http.get('/api/system/devices/inference', () => HttpResponse.json([])),
        http.get('/api/jobs', () => HttpResponse.json([]))
    );
});

it('queues an INT8 job without replacing the original and hides its card after cancellation', async () => {
    let jobs: SchemaModelExportJob[] = [];
    server.use(
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.post('/api/models/{model_id}/optimize', ({ params }) => {
            expect(params.model_id).toBe('model-1');
            jobs = [getMockedOptimizeModelJob()];
            return HttpResponse.json(jobs[0], { status: 202 });
        }),
        http.post('/api/jobs/{job_id}:interrupt', () => {
            jobs = [getMockedOptimizeModelJob({ status: 'canceled' })];
            return HttpResponse.json(jobs[0]);
        })
    );
    const user = userEvent.setup();
    render(<ModelFormats model={detail.model} />);
    const original = await screen.findByRole('region', { name: 'OpenVINO model format' });
    const optimize = within(original).getByRole('button', { name: 'Optimize' });
    await waitFor(() => expect(optimize).toBeEnabled());
    await user.click(optimize);
    const dialog = screen.getByRole('dialog', { name: 'Add model format' });
    expect(within(dialog).getByRole('radio', { name: 'OpenVINO · INT8' })).toBeChecked();
    expect(within(dialog).getByText('INT8 compresses weights, not activations.')).toBeInTheDocument();
    await user.click(within(dialog).getByRole('button', { name: 'Optimize to INT8' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    const card = await screen.findByRole('region', { name: 'OpenVINO · INT8' });
    expect(within(card).getByRole('status')).toHaveTextContent('Export queued');
    expect(card).toHaveAttribute('aria-busy', 'true');
    expect(original).toHaveAttribute('aria-busy', 'false');
    expect(within(original).getByRole('button', { name: 'Actions for OpenVINO Original' })).toBeEnabled();
    await user.click(within(card).getByRole('button', { name: 'Cancel export' }));
    await waitFor(() => expect(screen.queryByRole('region', { name: 'OpenVINO · INT8' })).not.toBeInTheDocument());
    expect(original).toBeInTheDocument();
    await waitFor(() => expect(within(original).getByRole('button', { name: 'Optimize' })).toBeEnabled());
});

it('does not queue a duplicate when an INT8 package already exists', async () => {
    server.use(
        http.get('/api/models/{model_id}', () =>
            HttpResponse.json(
                getMockedModelDetailResponse({
                    available_actions: [],
                    exports: [
                        {
                            ...detail.exports[0],
                            variants: [getMockedExportVariant(), getMockedExportVariant({ id: 'int8' })],
                        },
                    ],
                })
            )
        )
    );
    render(<ModelFormats model={detail.model} />);
    expect(await screen.findByRole('region', { name: 'OpenVINO model format' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'OpenVINO · INT8' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Optimize' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Add model format' })).not.toBeInTheDocument();
});

it('blocks optimization while job status is unavailable and supports retry', async () => {
    server.use(http.get('/api/jobs', () => HttpResponse.error()));
    const user = userEvent.setup();
    render(<ModelFormats model={detail.model} />);
    const retry = await screen.findByRole('button', { name: 'Retry job status' });
    expect(screen.getByRole('button', { name: 'Optimize' })).toBeDisabled();
    server.use(http.get('/api/jobs', () => HttpResponse.json([])));
    await user.click(retry);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Optimize' })).toBeEnabled());
});

it('keeps failure history separate for original and INT8 packages on the same backend', async () => {
    server.use(
        http.get('/api/jobs', () =>
            HttpResponse.json([
                getMockedOptimizeModelJob({ id: 'int8-failed', status: 'failed', created_at: '2026-11-18T10:00:00Z' }),
                getMockedOptimizeModelJob({
                    id: 'original-completed',
                    status: 'completed',
                    created_at: '2026-11-19T10:00:00Z',
                    payload: {
                        model_id: 'model-1',
                        backend: 'openvino',
                        operation: 'export',
                        cancellation_requested: false,
                    },
                }),
            ])
        )
    );
    render(<ModelFormats model={detail.model} />);
    const original = await screen.findByRole('region', { name: 'OpenVINO model format' });
    const int8 = await screen.findByRole('region', { name: 'OpenVINO · INT8' });
    expect(within(original).queryByRole('status')).not.toBeInTheDocument();
    expect(within(int8).getByRole('status')).toHaveTextContent('Export failed');
    expect(within(int8).getByRole('button', { name: 'Retry export' })).toBeEnabled();
});

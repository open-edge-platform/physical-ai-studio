import { ComponentProps } from 'react';

import { rs } from '@rstest/core';
import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { createQueryClient } from '../../../query-client/query-client';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { getMockedModelDetailResponse } from '../../../test-utils/mocks/mock-model-detail-response';
import { getMockedOptimizeModelJob } from '../../../test-utils/mocks/mock-optimize-model-job';
import { getMockedTrainJob } from '../../../test-utils/mocks/mock-train-job';
import { render } from '../../../test-utils/render';
import { JOBS_QUERY_KEY } from '../../jobs/use-job-cache';
import { BackendCard } from './backend-card';

const renderCard = (props: Partial<ComponentProps<typeof BackendCard>> = {}, queryClient = createQueryClient()) => {
    render(
        <BackendCard
            model={getMockedModel()}
            modelDetail={getMockedModelDetailResponse()}
            backendType='openvino'
            {...props}
        />,
        { queryClient }
    );
    return queryClient;
};

describe('BackendCard optimization', () => {
    beforeEach(() => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));
    });

    it('submits the source model id once, caches the queued job and prevents duplicates', async () => {
        const user = userEvent.setup();
        const job = getMockedOptimizeModelJob();
        let release = () => {};
        const gate = new Promise<void>((resolve) => {
            release = resolve;
        });
        const submit = rs.fn();
        server.use(
            http.post('/api/models/{model_id}/optimize', async ({ params }) => {
                submit(params.model_id);
                await gate;
                return HttpResponse.json(job, { status: 202 });
            })
        );
        const queryClient = renderCard();
        const button = screen.getByRole('button', { name: 'Optimize to INT8' });
        expect(button).toBeDisabled();
        await waitFor(() => expect(button).toBeEnabled());

        await user.click(button);
        expect(button).toBeDisabled();
        await user.click(button);
        server.use(http.get('/api/jobs', () => HttpResponse.json([job])));
        release();

        expect(await screen.findByRole('status')).toHaveTextContent('Optimization pending: Queued for optimization');
        expect(button).toBeDisabled();
        expect(submit).toHaveBeenCalledTimes(1);
        expect(submit).toHaveBeenCalledWith('model-1');
        expect(queryClient.getQueryData(JOBS_QUERY_KEY)).toEqual([job]);
        expect(screen.getByRole('progressbar', { name: 'Optimization progress' })).toHaveAttribute(
            'aria-valuenow',
            '0'
        );
    });

    it('surfaces the backend error and permits retry', async () => {
        const user = userEvent.setup();
        server.use(
            http.post('/api/models/{model_id}/optimize', ({ response }) =>
                response.untyped(
                    HttpResponse.json(
                        { message: 'OpenVINO export is missing. Export this model first.' },
                        { status: 400 }
                    )
                )
            )
        );
        renderCard();
        const button = screen.getByRole('button', { name: 'Optimize to INT8' });
        await waitFor(() => expect(button).toBeEnabled());
        await user.click(button);

        expect(await screen.findByRole('alert')).toHaveTextContent(
            'OpenVINO export is missing. Export this model first.'
        );
        expect(button).toBeEnabled();
    });

    it.each(['pending', 'running'] as const)(
        'shows %s progress and cancels through the interrupt endpoint',
        async (status) => {
            const user = userEvent.setup();
            const job = getMockedOptimizeModelJob({ status, progress: 42, message: 'Compressing weights' });
            server.use(http.get('/api/jobs', () => HttpResponse.json([job])));
            const cancel = rs.fn(({ params }: { params: { job_id: string } }) => {
                expect(params.job_id).toBe(job.id);
                const canceled = { ...job, status: 'canceled' as const, message: 'Optimization canceled' };
                server.use(http.get('/api/jobs', () => HttpResponse.json([canceled])));
                return HttpResponse.json(null);
            });
            server.use(http.post('/api/jobs/{job_id}:interrupt', cancel));
            renderCard();

            expect(await screen.findByRole('status')).toHaveTextContent(`Optimization ${status}: Compressing weights`);
            expect(screen.getByRole('progressbar', { name: 'Optimization progress' })).toHaveAttribute(
                'aria-valuenow',
                '42'
            );
            expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
            await user.click(screen.getByRole('button', { name: 'Cancel optimization' }));

            await waitFor(() => expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeEnabled());
            expect(screen.getByRole('status')).toHaveTextContent('Optimization canceled: Optimization canceled');
            expect(cancel).toHaveBeenCalledTimes(1);
        }
    );

    it('surfaces cancellation errors while keeping the active job protected from duplicates', async () => {
        const user = userEvent.setup();
        const job = getMockedOptimizeModelJob({ status: 'running' });
        server.use(
            http.get('/api/jobs', () => HttpResponse.json([job])),
            http.post('/api/jobs/{job_id}:interrupt', () =>
                HttpResponse.json({ message: 'Cancellation unavailable. Please retry.' }, { status: 400 })
            )
        );
        renderCard();
        await user.click(await screen.findByRole('button', { name: 'Cancel optimization' }));

        expect(await screen.findByRole('alert')).toHaveTextContent('Cancellation unavailable. Please retry.');
        expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Cancel optimization' })).toBeEnabled();
    });

    it('shows the latest matching failure rather than unrelated or older jobs, then permits retry', async () => {
        const failed = getMockedOptimizeModelJob({
            status: 'failed',
            message: 'Not enough memory. Free memory and retry.',
        });
        server.use(
            http.get('/api/jobs', () =>
                HttpResponse.json([
                    failed,
                    getMockedTrainJob(),
                    getMockedOptimizeModelJob({
                        id: 'other-model-job',
                        payload: { model_id: 'other-model', mode: 'int8_sym' },
                        created_at: '2026-07-14T14:00:00Z',
                    }),
                    getMockedOptimizeModelJob({
                        id: 'older',
                        status: 'completed',
                        created_at: '2026-07-14T12:00:00Z',
                    }),
                ])
            )
        );
        const queryClient = renderCard();
        expect(await screen.findByRole('status')).toHaveTextContent('Optimization failed: Not enough memory.');
        expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeEnabled();

        act(() => queryClient.setQueryData(JOBS_QUERY_KEY, [getMockedOptimizeModelJob({ status: 'running' })]));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled());
        act(() => queryClient.setQueryData(JOBS_QUERY_KEY, [failed]));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeEnabled());
    });

    it('blocks optimization when jobs cannot be loaded and offers a status retry', async () => {
        const user = userEvent.setup();
        server.use(http.get('/api/jobs', () => HttpResponse.error()));
        renderCard();
        expect(await screen.findByRole('alert')).toHaveTextContent('Could not load optimization status.');
        expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeDisabled();
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));
        await user.click(screen.getByRole('button', { name: 'Retry status' }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Optimize to INT8' })).toBeEnabled());
    });

    it('labels optimized precision and does not offer reoptimization', () => {
        renderCard({ model: getMockedModel({ optimization: { source_model_id: 'source', mode: 'int8_sym' } }) });
        expect(screen.getByText('INT8 symmetric weights')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Optimize to INT8' })).not.toBeInTheDocument();
    });

    it('explains weight-only compression and validation without falsely labeling original precision as FP16', () => {
        renderCard();
        expect(screen.getByText('Original')).toBeInTheDocument();
        expect(screen.queryByText('FP16')).not.toBeInTheDocument();
        expect(
            screen.getByText(
                'Compresses weights to symmetric INT8, not activations. The original model remains available. Validate policy accuracy before deployment.'
            )
        ).toBeInTheDocument();
    });

    it('does not offer optimization for a missing OpenVINO export', () => {
        renderCard({ modelDetail: getMockedModelDetailResponse({ exports: [] }) });
        expect(screen.queryByRole('button', { name: 'Optimize to INT8' })).not.toBeInTheDocument();
    });

    it('does not offer optimization for other backends', () => {
        renderCard({
            backendType: 'torch',
            modelDetail: getMockedModelDetailResponse({ exports: [{ type: 'torch', size_bytes: 1, file_count: 1 }] }),
        });
        expect(screen.queryByRole('button', { name: 'Optimize to INT8' })).not.toBeInTheDocument();
    });
});

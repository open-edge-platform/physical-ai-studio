import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaModelExportJob } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { createQueryClient } from '../../../query-client/query-client';
import { getMockedModel } from '../../../test-utils/mocks/mock-model';
import { getMockedBackendExportDetail, getMockedModelDetail } from '../../../test-utils/mocks/mock-model-detail';
import { getMockedModelExportJob } from '../../../test-utils/mocks/mock-model-export-job';
import { render } from '../../../test-utils/render';
import { JOBS_QUERY_KEY } from '../../jobs/use-job-cache';
import { ModelFormats } from './model-formats';

const model = getMockedModel();

describe('ModelFormats', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/models/{model_id}', () => HttpResponse.json(getMockedModelDetail())),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );
    });

    it('hides missing variants and offers only available export actions', async () => {
        render(<ModelFormats model={model} />);
        const user = userEvent.setup();
        await user.click(await screen.findByRole('button', { name: 'Add variant' }));
        expect(screen.getByRole('dialog', { name: 'Add variant' })).toBeInTheDocument();
        expect(screen.getByRole('radiogroup', { name: 'Export backend' })).toBeInTheDocument();
        expect(screen.getByRole('radio', { name: /PyTorch/ })).toBeChecked();
        expect(screen.getByRole('radio', { name: /ONNX/ })).toBeInTheDocument();
        expect(screen.getAllByRole('radio').map((item) => item.getAttribute('value'))).toEqual([
            'torch',
            'openvino',
            'onnx',
        ]);
        expect(screen.queryByRole('radio', { name: /ExecuTorch/ })).not.toBeInTheDocument();
        expect(screen.queryByRole('region')).not.toBeInTheDocument();
        expect(screen.queryByText('Unavailable')).not.toBeInTheDocument();
    });

    it('opens the add-variant dialog with the keyboard and restores focus on dismissal', async () => {
        render(<ModelFormats model={model} />);
        const user = userEvent.setup();
        const addVariant = await screen.findByRole('button', { name: 'Add variant' });
        await waitFor(() => expect(addVariant).toBeEnabled());
        expect(within(addVariant).getByText('Export to another backend')).toBeInTheDocument();
        await user.tab();
        expect(addVariant).toHaveFocus();
        await user.keyboard('{Enter}');
        expect(screen.getByRole('radio', { name: /ONNX/ })).toBeInTheDocument();
        await user.keyboard('{Escape}');
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        await waitFor(() => expect(addVariant).toHaveFocus());
    });

    it('does not request an export until confirmed and allows canceling the selection', async () => {
        const submitted = rs.fn();
        server.use(
            http.post('/api/models/{model_id}/exports/{backend}', () => {
                submitted();
                return HttpResponse.json(getMockedModelExportJob(), { status: 202 });
            })
        );
        render(<ModelFormats model={model} />);
        const user = userEvent.setup();
        await user.click(await screen.findByRole('button', { name: 'Add variant' }));
        await user.click(screen.getByRole('radio', { name: /ONNX/ }));
        expect(screen.getByRole('radio', { name: /ONNX/ })).toBeChecked();
        expect(screen.queryByLabelText('ONNX variant')).not.toBeInTheDocument();
        expect(submitted).not.toHaveBeenCalled();
        await user.click(screen.getByRole('button', { name: 'Cancel' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(submitted).not.toHaveBeenCalled();
    });

    it('excludes existing backends from the dialog', async () => {
        server.use(
            http.get('/api/models/{model_id}', () =>
                HttpResponse.json(getMockedModelDetail({ exports: [getMockedBackendExportDetail()] }))
            )
        );
        render(<ModelFormats model={model} />);
        const user = userEvent.setup();
        await user.click(await screen.findByRole('button', { name: 'Add variant' }));
        expect(screen.queryByRole('radio', { name: /PyTorch/ })).not.toBeInTheDocument();
        expect(screen.getByRole('radio', { name: /OpenVINO/ })).toBeChecked();
        expect(screen.getByRole('radio', { name: /ONNX/ })).toBeInTheDocument();
    });

    it('sorts variant cards by backend rather than API response order', async () => {
        server.use(
            http.get('/api/models/{model_id}', () =>
                HttpResponse.json(
                    getMockedModelDetail({
                        exports: ['executorch', 'onnx', 'torch', 'openvino'].map((type) =>
                            getMockedBackendExportDetail({ type })
                        ),
                        available_actions: [],
                    })
                )
            )
        );
        render(<ModelFormats model={model} />);
        await screen.findByRole('region', { name: 'PyTorch variant' });
        expect(screen.getAllByRole('region').map((card) => card.getAttribute('aria-label'))).toEqual([
            'PyTorch variant',
            'OpenVINO variant',
            'ONNX variant',
            'ExecuTorch variant',
        ]);
    });

    it('offers deleted exports again instead of showing cards for old completed jobs', async () => {
        server.use(
            http.get('/api/models/{model_id}', () =>
                HttpResponse.json(getMockedModelDetail({ exports: [getMockedBackendExportDetail()] }))
            ),
            http.get('/api/jobs', () =>
                HttpResponse.json([
                    getMockedModelExportJob({
                        id: 'old-failed-job',
                        status: 'failed',
                        created_at: '2026-11-18T10:00:00Z',
                        payload: {
                            model_id: 'model-1',
                            backend: 'onnx',
                            operation: 'export',
                            cancellation_requested: false,
                        },
                    }),
                    getMockedModelExportJob({
                        status: 'completed',
                        created_at: '2026-11-19T10:00:00Z',
                        payload: {
                            model_id: 'model-1',
                            backend: 'onnx',
                            operation: 'export',
                            cancellation_requested: false,
                        },
                    }),
                ])
            )
        );
        render(<ModelFormats model={model} />);
        await waitFor(() => expect(screen.getByRole('button', { name: 'Add variant' })).toBeEnabled());
        expect(screen.getByRole('region', { name: 'PyTorch variant' })).toBeInTheDocument();
        expect(screen.queryByRole('region', { name: 'ONNX variant' })).not.toBeInTheDocument();
        expect(screen.queryByText(/Refreshing variant/)).not.toBeInTheDocument();
        await userEvent.setup().click(screen.getByRole('button', { name: 'Add variant' }));
        expect(screen.getByRole('radio', { name: /ONNX/ })).toBeInTheDocument();
    });

    it('can request a new export when a previous job for that backend completed', async () => {
        const oldJob = getMockedModelExportJob({ status: 'completed', message: 'Old export created' });
        let accept: () => void = () => undefined;
        const accepted = new Promise<void>((resolve) => {
            accept = resolve;
        });
        let jobs = [oldJob];
        server.use(
            http.get('/api/jobs', () => HttpResponse.json(jobs)),
            http.post('/api/models/{model_id}/exports/{backend}', async () => {
                await accepted;
                jobs = [getMockedModelExportJob({ id: 'new-export-job', created_at: '2026-11-19T11:00:00Z' }), oldJob];
                return HttpResponse.json(jobs[0], { status: 202 });
            })
        );
        render(<ModelFormats model={model} />);
        const user = userEvent.setup();
        await waitFor(() => expect(screen.getByRole('button', { name: 'Add variant' })).toBeEnabled());
        await user.click(screen.getByRole('button', { name: 'Add variant' }));
        await user.click(screen.getByRole('button', { name: 'Export variant' }));
        expect(within(screen.getByRole('dialog')).getByRole('status')).toHaveTextContent('Requesting export');
        await act(async () => {
            accept();
        });
        const card = await screen.findByRole('region', { name: 'PyTorch variant' });
        expect(await within(card).findByText('Export queued')).toBeInTheDocument();
        expect(within(card).queryByText(/Old export created/)).not.toBeInTheDocument();
    });

    it('immediately shows only the requested variant, then follows its job to completion', async () => {
        let accept: () => void = () => undefined;
        const accepted = new Promise<void>((resolve) => {
            accept = resolve;
        });
        let jobs: SchemaModelExportJob[] = [];
        const submitted = rs.fn();
        server.use(
            http.get('/api/jobs', () => HttpResponse.json(jobs)),
            http.post('/api/models/{model_id}/exports/{backend}', async ({ params }) => {
                submitted(params);
                await accepted;
                jobs = [getMockedModelExportJob()];
                return HttpResponse.json(jobs[0], { status: 202 });
            })
        );
        const queryClient = createQueryClient();
        render(<ModelFormats model={model} />, { queryClient });
        const user = userEvent.setup();
        await user.click(await screen.findByRole('button', { name: 'Add variant' }));
        await user.click(screen.getByRole('button', { name: 'Export variant' }));
        const dialog = screen.getByRole('dialog', { name: 'Add variant' });
        expect(within(dialog).getByRole('status')).toHaveTextContent('Requesting export');
        expect(within(dialog).getByRole('button', { name: /Export variant/ })).toBeDisabled();
        expect(within(dialog).getByRole('button', { name: 'Cancel' })).toBeDisabled();
        expect(within(dialog).getByRole('radio', { name: /ONNX/ })).toBeDisabled();
        await user.keyboard('{Escape}');
        expect(dialog).toBeInTheDocument();
        const card = await screen.findByLabelText('PyTorch variant');
        expect(card).toHaveAttribute('aria-busy', 'true');
        expect(screen.queryByLabelText('ONNX variant')).not.toBeInTheDocument();
        expect(screen.getByLabelText('Add variant', { selector: 'button' })).toBeDisabled();
        expect(submitted).toHaveBeenCalledWith({ model_id: 'model-1', backend: 'torch' });
        await act(async () => {
            accept();
        });
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(await screen.findByText('Export queued')).toBeInTheDocument();
        act(() => {
            queryClient.setQueryData(JOBS_QUERY_KEY, [getMockedModelExportJob({ status: 'running', progress: 40 })]);
        });
        await waitFor(() => expect(within(card).getByRole('progressbar')).toHaveAttribute('aria-valuenow', '40'));

        act(() => {
            queryClient.setQueryData(
                ['get', '/api/models/{model_id}', { params: { path: { model_id: model.id } } }],
                getMockedModelDetail({
                    exports: [getMockedBackendExportDetail()],
                })
            );
            queryClient.setQueryData(JOBS_QUERY_KEY, [getMockedModelExportJob({ status: 'completed', progress: 100 })]);
        });
        expect(await screen.findByRole('button', { name: 'Download PyTorch export' })).toBeInTheDocument();
        expect(within(card).queryByRole('progressbar')).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Add variant' })).toBeEnabled();
    });

    it('scopes an existing export to its backend and model', async () => {
        server.use(
            http.get('/api/jobs', () =>
                HttpResponse.json([
                    getMockedModelExportJob({ status: 'running' }),
                    getMockedModelExportJob({
                        id: 'other-model-job',
                        status: 'running',
                        payload: {
                            model_id: 'other-model',
                            backend: 'onnx',
                            operation: 'export',
                            cancellation_requested: false,
                        },
                    }),
                ])
            )
        );
        render(<ModelFormats model={model} />);
        const card = await screen.findByRole('region', { name: 'PyTorch variant' });
        expect(within(card).getByRole('button', { name: 'Cancel export' })).toBeInTheDocument();
        expect(screen.queryByRole('region', { name: 'ONNX variant' })).not.toBeInTheDocument();
        expect(screen.queryByRole('region', { name: 'OpenVINO variant' })).not.toBeInTheDocument();
    });

    it('keeps existing variants usable while another backend exports', async () => {
        server.use(
            http.get('/api/models/{model_id}', () =>
                HttpResponse.json(
                    getMockedModelDetail({
                        exports: [getMockedBackendExportDetail({ type: 'onnx' })],
                    })
                )
            ),
            http.get('/api/jobs', () => HttpResponse.json([getMockedModelExportJob({ status: 'running' })]))
        );
        render(<ModelFormats model={model} />);
        const card = await screen.findByRole('region', { name: 'ONNX variant' });
        expect(within(card).getByRole('button', { name: 'Download ONNX export' })).toBeEnabled();
        expect(within(card).queryByRole('status')).not.toBeInTheDocument();
        expect(card).toHaveAttribute('aria-busy', 'false');
        expect(await screen.findByRole('region', { name: 'PyTorch variant' })).toHaveAttribute('aria-busy', 'true');
    });

    it('does not offer an export when no backend is available', async () => {
        server.use(
            http.get('/api/models/{model_id}', () => HttpResponse.json(getMockedModelDetail({ available_actions: [] })))
        );
        render(<ModelFormats model={model} />);
        expect(await screen.findByText('No exported variants are available for this model.')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Add variant' })).not.toBeInTheDocument();
    });

    it('blocks new exports and offers recovery if job status cannot be loaded', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.error()));
        render(<ModelFormats model={model} />);
        expect(await screen.findByText('Unable to load export status.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Add variant' })).toBeDisabled();
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));
        await userEvent.setup().click(screen.getByRole('button', { name: 'Retry job status' }));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Add variant' })).toBeEnabled());
    });

    it('keeps a failed request in the dialog and on its own card and allows retrying', async () => {
        server.use(
            http.post('/api/models/{model_id}/exports/{backend}', () =>
                HttpResponse.json(
                    { detail: [{ loc: [], msg: 'Checkpoint unavailable', type: 'value_error' }] },
                    { status: 422 }
                )
            )
        );
        render(<ModelFormats model={model} />);
        const user = userEvent.setup();
        await user.click(await screen.findByRole('button', { name: 'Add variant' }));
        await user.click(screen.getByRole('radio', { name: /ONNX/ }));
        await user.click(screen.getByRole('button', { name: 'Export variant' }));
        const dialog = screen.getByRole('dialog', { name: 'Add variant' });
        expect(await within(dialog).findByRole('alert')).toHaveTextContent('Checkpoint unavailable');
        expect(within(dialog).getByRole('button', { name: 'Export variant' })).toBeEnabled();
        expect(within(dialog).getByRole('radio', { name: /ONNX/ })).toBeChecked();
        await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));
        const card = await screen.findByRole('region', { name: 'ONNX variant' });
        expect(await within(card).findByRole('alert')).toHaveTextContent('Checkpoint unavailable');
        expect(within(card).getByRole('button', { name: 'Retry export' })).toBeEnabled();
        expect(card).toHaveAttribute('aria-busy', 'false');
        expect(screen.queryByRole('region', { name: 'PyTorch variant' })).not.toBeInTheDocument();
        await waitFor(() => expect(screen.getByRole('button', { name: 'Add variant' })).toBeEnabled());
    });

    it('retries the selected backend in the dialog and closes only after the request succeeds', async () => {
        let jobs: SchemaModelExportJob[] = [];
        const submitted = rs.fn();
        server.use(
            http.get('/api/jobs', () => HttpResponse.json(jobs)),
            http.post('/api/models/{model_id}/exports/{backend}', ({ params }) => {
                submitted(params.backend);
                if (submitted.mock.calls.length === 1) {
                    return HttpResponse.json(
                        { detail: [{ loc: [], msg: 'Please try again', type: 'value_error' }] },
                        { status: 422 }
                    );
                }
                jobs = [getMockedModelExportJob()];
                return HttpResponse.json(jobs[0], { status: 202 });
            })
        );
        render(<ModelFormats model={model} />);
        const user = userEvent.setup();
        await user.click(await screen.findByRole('button', { name: 'Add variant' }));
        await user.click(screen.getByRole('button', { name: 'Export variant' }));
        const dialog = screen.getByRole('dialog');
        expect(await within(dialog).findByRole('alert')).toHaveTextContent('Please try again');
        expect(within(dialog).getByRole('radio', { name: /PyTorch/ })).toBeChecked();
        await user.click(within(dialog).getByRole('button', { name: 'Export variant' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(submitted.mock.calls).toEqual([['torch'], ['torch']]);
        const card = await screen.findByRole('region', { name: 'PyTorch variant' });
        expect(within(card).getByRole('status')).toHaveTextContent('Export queued');
    });
});

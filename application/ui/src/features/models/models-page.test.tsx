import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaTrainJob } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { getMockedTrainJob } from '../../test-utils/mocks/mock-train-job';
import { render } from '../../test-utils/render';
import { JobsButton } from '../jobs/footer/jobs-button';
import { useJobUpdates } from '../jobs/use-job-updates';
import { ModelsPage } from './models-page';

const projectId = 'project-1';
const datasetId = 'dataset-1';

// The page opens a live WebSocket (job updates) and job rows open an
// EventSource (metrics). jsdom provides neither, so stub both with no-op
// implementations, restored per-test.
class FakeWebSocket extends EventTarget {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSING = 2;
    static readonly CLOSED = 3;
    readyState = FakeWebSocket.OPEN;
    constructor(
        public url: string | URL,
        public protocols?: string | string[]
    ) {
        super();
    }
    close() {
        this.readyState = FakeWebSocket.CLOSED;
    }
    send() {}
}

class ImmediatelyClosingEventSource {
    onmessage: ((event: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;
    constructor() {
        queueMicrotask(() => this.onmessage?.({ data: 'DONE' }));
    }
    close() {}
}

const runningJob = getMockedTrainJob({
    id: 'job-running',
    status: 'running',
    payload: { model_name: 'running-model' } as never,
});
const pendingJob = getMockedTrainJob({
    id: 'job-pending',
    status: 'pending',
    payload: { model_name: 'pending-model' } as never,
});
const canceledJob = getMockedTrainJob({
    id: 'job-canceled',
    status: 'canceled',
    end_time: '2026-07-14T10:05:00Z',
    payload: { model_name: 'canceled-model' } as never,
});
const failedJob = getMockedTrainJob({
    id: 'job-failed',
    status: 'failed',
    payload: { model_name: 'failed-model' } as never,
});

const mockApi = (initialJobs: SchemaTrainJob[]) => {
    let jobs = initialJobs;

    server.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([])),
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.get('/api/dataset/{dataset_id}', () => HttpResponse.error()),
        http.get('/api/projects/{project_id}/environments/{environment_id}', () => HttpResponse.error()),
        http.get('/api/remote-trainers', () => HttpResponse.json([])),
        http.get('/api/projects/{project_id}', () =>
            HttpResponse.json({
                id: projectId,
                name: 'Test project',
                datasets: [
                    {
                        id: datasetId,
                        name: 'Test dataset',
                        default_task: 'test',
                        project_id: projectId,
                        environment_id: 'environment-1',
                    },
                ],
            })
        ),
        http.post('/api/jobs:train', () => {
            jobs = [...jobs, pendingJob];
            return HttpResponse.json(pendingJob, { status: 201 });
        })
    );
};

const renderPage = () =>
    render(<ModelsPage />, {
        route: `/projects/${projectId}/models`,
        path: '/projects/:project_id/models',
    });

describe('ModelsPage - Current Training', () => {
    beforeEach(() => {
        vi.stubGlobal('WebSocket', FakeWebSocket);
        vi.stubGlobal('EventSource', ImmediatelyClosingEventSource);
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('shows running and pending jobs in Current Training, but not terminal jobs', async () => {
        mockApi([runningJob, pendingJob, canceledJob, failedJob]);

        renderPage();

        expect(await screen.findByText('running-model')).toBeInTheDocument();
        expect(screen.getByText('pending-model')).toBeInTheDocument();
        expect(screen.queryByText('canceled-model')).not.toBeInTheDocument();
        expect(screen.queryByText('failed-model')).not.toBeInTheDocument();
    });

    it('shows the "No trained models" placeholder, with no All Jobs button, when there are no models and no active jobs', async () => {
        mockApi([canceledJob, failedJob]);

        renderPage();

        expect(await screen.findByText('No trained models')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /all jobs/i })).not.toBeInTheDocument();
    });

    it('shows the submitted job as pending immediately when training from the empty placeholder', async () => {
        const user = userEvent.setup();
        mockApi([]);

        renderPage();

        await user.click(await screen.findByRole('button', { name: 'Train model' }));
        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(screen.getByRole('button', { name: 'Train' }));

        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(await screen.findByText('pending-model')).toBeInTheDocument();
    });
});

let capturedOnMessage: ((event: { data: string }) => void) | undefined;

vi.mock('react-use-websocket', () => ({
    default: (_url: unknown, options: { onMessage?: (event: { data: string }) => void }) => {
        capturedOnMessage = options.onMessage;
        return { readyState: 1 };
    },
}));

const FooterStandIn = () => {
    useJobUpdates(projectId);
    return <JobsButton projectId={projectId} />;
};

const deliverJobUpdate = (job: SchemaTrainJob) => {
    act(() => {
        capturedOnMessage?.({ data: JSON.stringify({ event: 'JOB_UPDATE', data: job }) });
    });
};

describe('ModelsPage + JobsButton - shared query-cache integration', () => {
    beforeEach(() => {
        vi.stubGlobal('WebSocket', FakeWebSocket);
        vi.stubGlobal('EventSource', ImmediatelyClosingEventSource);
        capturedOnMessage = undefined;
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('reflects a single job-cache update in both ModelsPage and the footer-owned Jobs dialog', async () => {
        const user = userEvent.setup();
        mockApi([runningJob]);

        render(
            <>
                <ModelsPage />
                <FooterStandIn />
            </>,
            { route: `/projects/${projectId}/models`, path: '/projects/:project_id/models' }
        );

        expect(await screen.findAllByText('running-model')).toHaveLength(1);

        await user.click(await screen.findByRole('button', { name: 'Jobs' }));
        const dialog = await screen.findByRole('dialog');
        expect(await screen.findAllByText('running-model')).toHaveLength(2);
        expect(within(dialog).getByText('running')).toBeInTheDocument();

        deliverJobUpdate({ ...runningJob, status: 'completed' });

        await waitFor(() => expect(screen.getAllByText('running-model')).toHaveLength(1));
        expect(within(dialog).getByText('running-model')).toBeInTheDocument();
        expect(within(dialog).getByText('completed')).toBeInTheDocument();
    });
});

import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { SchemaTrainJob } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { getMockedTrainJob } from '../../test-utils/mocks/mock-train-job';
import { render } from '../../test-utils/render';
import { ModelsPage } from './models-page';

const projectId = 'project-1';

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

const mockApi = (jobs: SchemaTrainJob[]) => {
    server.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([])),
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.get('/api/dataset/{dataset_id}', () => HttpResponse.error()),
        http.get('/api/projects/{project_id}/environments/{environment_id}', () => HttpResponse.error()),
        http.get('/api/remote-trainers', () => HttpResponse.json([]))
    );
};

const renderPage = () =>
    render(<ModelsPage />, {
        route: `/projects/${projectId}/models`,
        path: '/projects/:project_id/models',
    });

describe('ModelsPage - Current Training vs All jobs', () => {
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

    it('lists every job regardless of status in the All jobs dialog', async () => {
        const user = userEvent.setup();
        mockApi([runningJob, pendingJob, canceledJob, failedJob]);

        renderPage();

        await user.click(await screen.findByRole('button', { name: /all jobs/i }));

        const dialog = await screen.findByRole('dialog');
        expect(within(dialog).getByText('running-model')).toBeInTheDocument();
        expect(within(dialog).getByText('pending-model')).toBeInTheDocument();
        expect(within(dialog).getByText('canceled-model')).toBeInTheDocument();
        expect(within(dialog).getByText('failed-model')).toBeInTheDocument();
    });

    it('disables the All jobs button when there are no jobs', async () => {
        mockApi([]);

        renderPage();

        // No models and no active jobs -> illustrated placeholder, no All jobs button.
        expect(await screen.findByText('No trained models')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /all jobs/i })).not.toBeInTheDocument();
    });

    it('surfaces All jobs from the placeholder when only terminal jobs exist and no models', async () => {
        const user = userEvent.setup();
        mockApi([canceledJob, failedJob]);

        renderPage();

        expect(await screen.findByText('No trained models')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /all jobs/i }));

        const dialog = await screen.findByRole('dialog');
        expect(within(dialog).getByText('canceled-model')).toBeInTheDocument();
        expect(within(dialog).getByText('failed-model')).toBeInTheDocument();
    });
});

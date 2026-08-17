import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { Index } from './index';
import { SchemaTrainJob } from './train-model-dialog';

const projectId = 'b8b28d4f-e78f-48ad-afb8-03d060178a3c';
const jobId = 'f1b3f8f0-1c1a-4b1a-9c1a-1c1a1c1a1c1a';

// `Index` opens a live WebSocket for job updates. jsdom has no server to
// connect to, so stub out `WebSocket` with a minimal no-op implementation
// that never attempts a real network connection. Matches the constructor
// signature of the real WebSocket so `react-use-websocket` can instantiate
// it safely, and is stubbed/restored per-test so it can't leak into other
// test files.
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

beforeEach(() => {
    vi.stubGlobal('WebSocket', FakeWebSocket);
});

afterEach(() => {
    vi.unstubAllGlobals();
});

let jobs: SchemaTrainJob[];

const baseJob: SchemaTrainJob = {
    id: jobId,
    project_id: projectId,
    type: 'training',
    status: 'canceled',
    progress: 42,
    message: 'Cancelled by user',
    start_time: '2026-08-01T00:00:00Z',
    end_time: '2026-08-01T00:05:00Z',
    created_at: '2026-08-01T00:00:00Z',
    extra_info: {},
    payload: {
        training_target: 'local',
        model_name: 'Cancelled model',
        policy: 'act',
        dataset_id: 'd1',
    },
} as unknown as SchemaTrainJob;

const mockRoutes = () => {
    jobs = [baseJob];

    server.use(
        http.get('/api/projects/{project_id}/models', () => HttpResponse.json([])),
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.delete('/api/jobs/{job_id}', () => {
            jobs = jobs.filter((job) => job.id !== jobId);
            return HttpResponse.json(null);
        })
    );
};

describe('Index - deleting a canceled training job', () => {
    it('removes the job row from the list once deletion succeeds, without a page refresh', async () => {
        const user = userEvent.setup();
        mockRoutes();

        render(<Index />, {
            route: `/projects/${projectId}/models`,
            path: '/projects/:project_id/models',
        });

        expect(await screen.findByText('Cancelled model')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Job options' }));
        await user.click(screen.getByRole('menuitem', { name: 'Delete' }));

        await waitFor(() => {
            expect(screen.queryByText('Cancelled model')).not.toBeInTheDocument();
        });
    });
});

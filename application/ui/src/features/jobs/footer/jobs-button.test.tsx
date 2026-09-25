import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { SchemaTrainJob } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedTrainJob } from '../../../test-utils/mocks/mock-train-job';
import { render } from '../../../test-utils/render';
import { JobsButton } from './jobs-button';

// LogsDialog streams via the browser-native EventSource, which jsdom does not
// implement. Stub it so opening the logs view resolves its stream immediately
// instead of throwing `ReferenceError: EventSource is not defined`.
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

const mockApi = (jobs: SchemaTrainJob[]) => {
    server.use(
        http.get('/api/jobs', () => HttpResponse.json(jobs)),
        http.get('/api/logs/sources', () =>
            HttpResponse.json([{ id: 'application', name: 'Application', type: 'application', created_at: null }])
        )
    );
};

const renderButton = () => render(<JobsButton />, { route: '/', path: '/' });

describe('JobsButton', () => {
    beforeEach(() => {
        vi.stubGlobal('EventSource', ImmediatelyClosingEventSource);
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('renders an enabled "Jobs" trigger even with an empty jobs list', async () => {
        mockApi([]);

        renderButton();

        const trigger = await screen.findByRole('button', { name: 'Jobs' });
        expect(trigger).toBeEnabled();
    });

    it('opens the dialog defaulting to the "All jobs" tab', async () => {
        const user = userEvent.setup();
        mockApi([runningJob]);

        renderButton();

        await user.click(await screen.findByRole('button', { name: 'Jobs' }));

        const allTab = await screen.findByRole('tab', { name: /all jobs/i });
        expect(allTab).toHaveAttribute('aria-selected', 'true');
    });

    it('opens fullscreen logs for a row, hiding Jobs, then returns to Jobs with the tab preserved', async () => {
        const user = userEvent.setup();
        mockApi([runningJob]);

        renderButton();

        await user.click(await screen.findByRole('button', { name: 'Jobs' }));
        await user.click(await screen.findByRole('tab', { name: /running jobs/i }));

        await user.click(await screen.findByRole('button', { name: 'Job options' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Logs' }));

        // Logs is fullscreen and Jobs is gone.
        expect(await screen.findByRole('heading', { name: 'Logs' })).toBeInTheDocument();
        expect(screen.queryByRole('tab', { name: /all jobs/i })).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Close' }));

        // Back on Jobs, with the previously-selected tab still selected.
        const runningTab = await screen.findByRole('tab', { name: /running jobs/i });
        expect(runningTab).toHaveAttribute('aria-selected', 'true');
        expect(screen.queryByRole('heading', { name: 'Logs' })).not.toBeInTheDocument();
    });

    it('resets to "All jobs" when Jobs is closed and reopened', async () => {
        const user = userEvent.setup();
        mockApi([runningJob]);

        renderButton();

        await user.click(await screen.findByRole('button', { name: 'Jobs' }));
        await user.click(await screen.findByRole('tab', { name: /running jobs/i }));

        await user.click(screen.getByRole('button', { name: 'Dismiss' }));
        await waitFor(() => expect(screen.queryByRole('tab', { name: /all jobs/i })).not.toBeInTheDocument());

        await user.click(await screen.findByRole('button', { name: 'Jobs' }));

        const allTab = await screen.findByRole('tab', { name: /all jobs/i });
        expect(allTab).toHaveAttribute('aria-selected', 'true');
    });
});

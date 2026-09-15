import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaTrainJob } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { getMockedTrainJob } from '../../test-utils/mocks/mock-train-job';
import { render } from '../../test-utils/render';
import { JobsDialog } from './jobs-dialog';

const PROJECT_A = 'project-a';
const PROJECT_B = 'project-b';

const runningJob = getMockedTrainJob({
    id: 'job-running',
    status: 'running',
    project_id: PROJECT_A,
    created_at: '2026-08-01T00:00:00Z',
    payload: { model_name: 'running-model' } as never,
});
const completedJob = getMockedTrainJob({
    id: 'job-completed',
    status: 'completed',
    project_id: PROJECT_A,
    created_at: '2026-07-01T00:00:00Z',
    end_time: '2026-07-01T01:00:00Z',
    payload: { model_name: 'completed-model' } as never,
});
const otherProjectJob = getMockedTrainJob({
    id: 'job-other-project',
    status: 'running',
    project_id: PROJECT_B,
    created_at: '2026-08-02T00:00:00Z',
    payload: { model_name: 'other-project-model' } as never,
});

const mockJobs = (jobs: SchemaTrainJob[]) => {
    server.use(http.get('/api/jobs', () => HttpResponse.json(jobs)));
};

const renderDialog = (props: Partial<Parameters<typeof JobsDialog>[0]> = {}) =>
    render(<JobsDialog onViewLogs={() => {}} close={() => {}} {...props} />, {
        route: '/projects/project-a/models',
        path: '/projects/:project_id/models',
    });

describe('JobsDialog', () => {
    it('renders all six status tabs in the agreed order with the exact labels', async () => {
        mockJobs([runningJob, completedJob]);

        renderDialog({ projectId: PROJECT_A });

        const tabs = await screen.findAllByRole('tab');
        const labels = tabs.map((tab) => tab.textContent);

        expect(labels).toEqual([
            expect.stringContaining('All jobs'),
            expect.stringContaining('Running jobs'),
            expect.stringContaining('Finished jobs'),
            expect.stringContaining('Scheduled jobs'),
            expect.stringContaining('Cancelled jobs'),
            expect.stringContaining('Failed jobs'),
        ]);
    });

    it('shows a count badge only for tabs with a positive count', async () => {
        mockJobs([runningJob, completedJob]);

        renderDialog({ projectId: PROJECT_A });

        const runningTab = await screen.findByRole('tab', { name: /^running jobs\d+$/i });
        const finishedTab = screen.getByRole('tab', { name: /^finished jobs\d+$/i });
        const failedTab = screen.getByRole('tab', { name: /^failed jobs$/i });

        expect(within(runningTab).getByText('1')).toBeInTheDocument();
        expect(within(finishedTab).getByText('1')).toBeInTheDocument();
        expect(within(failedTab).queryByText(/\d/)).not.toBeInTheDocument();
    });

    it("highlights only the currently selected tab's count badge in energy blue", async () => {
        const user = userEvent.setup();
        mockJobs([runningJob, completedJob]);

        renderDialog({ projectId: PROJECT_A });

        const allTab = await screen.findByRole('tab', { name: /^all jobs\d+$/i });
        const runningTab = screen.getByRole('tab', { name: /^running jobs\d+$/i });

        // The Badge's inline style lives on its `role="presentation"` wrapper,
        // one level up from the digit text itself.
        const badgeOf = (tab: HTMLElement) => within(tab).getByRole('presentation');

        // "All jobs" is selected by default.
        expect(badgeOf(allTab)).toHaveStyle({ backgroundColor: 'var(--energy-blue)' });
        expect(badgeOf(runningTab)).not.toHaveStyle({ backgroundColor: 'var(--energy-blue)' });

        await user.click(runningTab);

        expect(badgeOf(runningTab)).toHaveStyle({ backgroundColor: 'var(--energy-blue)' });
        expect(badgeOf(allTab)).not.toHaveStyle({ backgroundColor: 'var(--energy-blue)' });
    });

    it('keeps zero-count tabs selectable and shows an illustrated empty message', async () => {
        const user = userEvent.setup();
        mockJobs([runningJob]);

        const { container } = renderDialog({ projectId: PROJECT_A });

        const failedTab = await screen.findByRole('tab', { name: /^failed jobs$/i });
        await user.click(failedTab);

        expect(await screen.findByText('No failed jobs.')).toBeInTheDocument();
        expect(container.querySelector('svg')).toBeInTheDocument();
    });

    it('filters visible rows per tab', async () => {
        const user = userEvent.setup();
        mockJobs([runningJob, completedJob]);

        renderDialog({ projectId: PROJECT_A });

        await user.click(await screen.findByRole('tab', { name: /running jobs/i }));

        expect(await screen.findByText('running-model')).toBeInTheDocument();
        expect(screen.queryByText('completed-model')).not.toBeInTheDocument();

        await user.click(screen.getByRole('tab', { name: /finished jobs/i }));

        expect(await screen.findByText('completed-model')).toBeInTheDocument();
        expect(screen.queryByText('running-model')).not.toBeInTheDocument();
    });

    it('scopes jobs to the given project and titles the heading "Current project jobs"', async () => {
        mockJobs([runningJob, otherProjectJob]);

        renderDialog({ projectId: PROJECT_A });

        expect(await screen.findByText('Current project jobs')).toBeInTheDocument();
        expect(await screen.findByText('running-model')).toBeInTheDocument();
        expect(screen.queryByText('other-project-model')).not.toBeInTheDocument();
    });

    it('shows every project\'s jobs and titles the heading "All projects jobs" when projectId is omitted', async () => {
        mockJobs([runningJob, otherProjectJob]);

        renderDialog();

        expect(await screen.findByText('All projects jobs')).toBeInTheDocument();
        expect(await screen.findByText('running-model')).toBeInTheDocument();
        expect(screen.getByText('other-project-model')).toBeInTheDocument();
    });

    it('shows a loading indicator before jobs render, then shows the loaded content', async () => {
        server.use(
            http.get('/api/jobs', async () => {
                await new Promise((resolve) => setTimeout(resolve, 20));
                return HttpResponse.json([runningJob]);
            })
        );

        renderDialog({ projectId: PROJECT_A });

        // The Loading spinner is indeterminate (no `aria-valuenow`), unlike a
        // running job's own determinate `ProgressBar`.
        const findSpinner = () =>
            screen.queryAllByRole('progressbar').filter((el) => !el.hasAttribute('aria-valuenow'));

        expect(findSpinner().length).toBeGreaterThan(0);
        expect(screen.queryByText('running-model')).not.toBeInTheDocument();

        expect(await screen.findByText('running-model')).toBeInTheDocument();
        expect(findSpinner()).toHaveLength(0);
    });

    it('shows an error message and a Retry button when the initial fetch fails, and retrying recovers', async () => {
        const user = userEvent.setup();
        let callCount = 0;
        server.use(
            http.get('/api/jobs', () => {
                callCount += 1;
                if (callCount === 1) {
                    return HttpResponse.error();
                }
                return HttpResponse.json([runningJob]);
            })
        );

        renderDialog({ projectId: PROJECT_A });

        expect(await screen.findByText('Failed to load jobs.')).toBeInTheDocument();
        const retryButton = screen.getByRole('button', { name: /retry/i });

        await user.click(retryButton);

        expect(await screen.findByText('running-model')).toBeInTheDocument();
    });

    it('shows the "All jobs" empty message on a successful empty result', async () => {
        mockJobs([]);

        const { container } = renderDialog({ projectId: PROJECT_A });

        expect(await screen.findByText('No jobs yet.')).toBeInTheDocument();
        expect(container.querySelector('svg')).toBeInTheDocument();
        expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
        expect(screen.queryByText('Failed to load jobs.')).not.toBeInTheDocument();
    });

    it('sorts jobs newest-created-first', async () => {
        mockJobs([completedJob, runningJob]);

        renderDialog({ projectId: PROJECT_A });

        const table = await screen.findByTestId('table');
        const names = within(table)
            .getAllByText(/-model$/)
            .map((el) => el.textContent);

        expect(names.indexOf('running-model')).toBeLessThan(names.indexOf('completed-model'));
    });
});

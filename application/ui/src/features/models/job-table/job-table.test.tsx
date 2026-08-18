import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { SchemaTrainJob } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { TrainingRow } from './job-table';

/**
 * jsdom does not implement `EventSource`, and MSW does not intercept it (no
 * EventSource interceptor exists in `@mswjs/interceptors`). `JobMetricsContent`
 * calls the real browser `EventSource` via `fetchSSE`, and it is the default
 * (and therefore immediately-mounted) tab panel for a job row, so it must be
 * stubbed here or expanding a job row would throw
 * `ReferenceError: EventSource is not defined`. The stub resolves the stream
 * immediately so the async iterator in `fetchSSE` terminates without hanging.
 */
class ImmediatelyClosingEventSource {
    onmessage: ((event: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;

    constructor() {
        queueMicrotask(() => this.onmessage?.({ data: 'DONE' }));
    }

    close() {}
}

const localJob: SchemaTrainJob = {
    id: 'job-1',
    project_id: 'project-1',
    progress: 42,
    status: 'running',
    message: 'Training',
    start_time: '2026-07-14T10:00:00Z',
    end_time: null,
    created_at: '2026-07-14T09:00:00Z',
    extra_info: { 'train/loss_step': 0.123456 },
    type: 'training',
    payload: {
        project_id: 'project-1',
        dataset_id: 'dataset-1',
        policy: 'act',
        model_name: 'pick-and-place',
        batch_size: 8,
        num_workers: 'auto',
        auto_scale_batch_size: false,
        val_split: 0.1,
        precision: 'bf16-mixed',
        compile_model: false,
        training_target: 'local',
    },
};

const remoteTrainer = {
    id: 'trainer-1',
    name: 'managed-trainer',
    url: 'https://trainer.example.test/api',
    created_at: '2026-07-14T12:00:00Z',
};

const renderTrainingRow = (
    trainJobOverride: Partial<SchemaTrainJob> = {},
    { onInterrupt = vi.fn(), onViewLogs = vi.fn() }: { onInterrupt?: () => void; onViewLogs?: () => void } = {}
) => {
    const trainJob: SchemaTrainJob = { ...localJob, ...trainJobOverride };
    return render(<TrainingRow trainJob={trainJob} onInterrupt={onInterrupt} onViewLogs={onViewLogs} />);
};

describe('TrainingRow', () => {
    beforeAll(() => {
        vi.stubGlobal('EventSource', ImmediatelyClosingEventSource);
    });

    afterAll(() => {
        vi.unstubAllGlobals();
    });

    beforeEach(() => {
        server.use(http.get('/api/remote-trainers', () => HttpResponse.json([remoteTrainer])));
    });

    it('renders the model name, loss to two decimal places, and the uppercased architecture', () => {
        renderTrainingRow();

        expect(screen.getByText('pick-and-place')).toBeInTheDocument();
        expect(screen.getByText('0.12')).toBeInTheDocument();
        expect(screen.getByText('ACT')).toBeInTheDocument();
    });

    it('shows "..." when extra_info[\'train/loss_step\'] is missing', () => {
        renderTrainingRow({ extra_info: null });

        expect(screen.getByText('...')).toBeInTheDocument();
    });

    it('renders the split status badge and a ProgressBar for a running job', () => {
        renderTrainingRow({ status: 'running' });

        expect(screen.getByText('running')).toBeInTheDocument();
        expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    it('renders no ProgressBar for a non-running job', () => {
        renderTrainingRow({ status: 'completed', end_time: '2026-07-14T10:30:00Z' });

        expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    });

    it('renders a Remote · {name} badge for a remote job', async () => {
        renderTrainingRow({
            payload: { ...localJob.payload, training_target: 'remote', remote_trainer_id: remoteTrainer.id },
        });

        expect(await screen.findByText(`Remote · ${remoteTrainer.name}`)).toBeInTheDocument();
    });

    it('renders no location badge for a local job', () => {
        renderTrainingRow();

        expect(screen.queryByText(/^Remote ·/)).not.toBeInTheDocument();
    });

    it('reveals the panel tabs when the row is clicked', async () => {
        const user = userEvent.setup();
        renderTrainingRow();

        await user.click(screen.getByText('pick-and-place'));

        expect(await screen.findByRole('tab', { name: 'Model Metrics' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Training Datasets' })).toBeInTheDocument();
    });

    it('does not collapse the row when a tab inside the panel is clicked', async () => {
        const user = userEvent.setup();
        renderTrainingRow();

        await user.click(screen.getByText('pick-and-place'));
        expect(await screen.findByRole('tab', { name: 'Model Metrics' })).toBeInTheDocument();

        await user.click(screen.getByRole('tab', { name: 'Training Datasets' }));

        expect(screen.getByRole('tab', { name: 'Model Metrics' })).toBeInTheDocument();
    });

    it('two job rows can be expanded simultaneously', async () => {
        const user = userEvent.setup();
        const secondJob: SchemaTrainJob = {
            ...localJob,
            id: 'job-2',
            payload: { ...localJob.payload, model_name: 'stack-blocks' },
        };

        render(
            <>
                <TrainingRow trainJob={localJob} onInterrupt={vi.fn()} onViewLogs={vi.fn()} />
                <TrainingRow trainJob={secondJob} onInterrupt={vi.fn()} onViewLogs={vi.fn()} />
            </>
        );

        await user.click(screen.getByText('pick-and-place'));
        await user.click(screen.getByText('stack-blocks'));

        expect(await screen.findAllByRole('tab', { name: 'Model Metrics' })).toHaveLength(2);
    });

    it('shows Logs and Delete in the overflow menu, with Delete disabled unless status is failed', async () => {
        const user = userEvent.setup();
        renderTrainingRow({ status: 'running' });

        await user.click(screen.getByRole('button', { name: 'Job options' }));

        expect(screen.getByRole('menuitem', { name: 'Logs' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete' }).closest('[aria-disabled]')).toHaveAttribute(
            'aria-disabled',
            'true'
        );
    });

    it('enables Delete in the overflow menu when status is failed', async () => {
        const user = userEvent.setup();
        renderTrainingRow({ status: 'failed' });

        await user.click(screen.getByRole('button', { name: 'Job options' }));

        expect(screen.getByRole('menuitem', { name: 'Delete' }).closest('[aria-disabled]')).toBeNull();
    });

    it('renders the Stop button only when status is running', () => {
        renderTrainingRow({ status: 'running' });
        expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument();
    });

    it('renders no Stop button when status is not running', () => {
        renderTrainingRow({ status: 'completed', end_time: '2026-07-14T10:30:00Z' });
        expect(screen.queryByRole('button', { name: 'Stop' })).not.toBeInTheDocument();
    });

    it('names the disclosure button "Show details for {model_name}" and exposes aria-expanded', async () => {
        const user = userEvent.setup();
        renderTrainingRow();

        const disclosureButton = screen.getByRole('button', { name: 'Show details for pick-and-place' });
        expect(disclosureButton).toHaveAttribute('aria-expanded', 'false');

        await user.click(disclosureButton);

        expect(await screen.findByRole('tab', { name: 'Model Metrics' })).toBeInTheDocument();
        expect(disclosureButton).toHaveAttribute('aria-expanded', 'true');
    });

    it('is expandable via the keyboard', async () => {
        const user = userEvent.setup();
        renderTrainingRow();

        await user.tab();
        expect(screen.getByRole('button', { name: 'Show details for pick-and-place' })).toHaveFocus();

        await user.keyboard('{Enter}');

        expect(await screen.findByRole('tab', { name: 'Model Metrics' })).toBeInTheDocument();
    });

    it('renders the ProgressBar outside the detail panel, visible while the row is collapsed', () => {
        renderTrainingRow({ status: 'running' });

        expect(screen.getByRole('progressbar')).toBeInTheDocument();
        expect(screen.queryByRole('tab', { name: 'Model Metrics' })).not.toBeInTheDocument();
    });
});

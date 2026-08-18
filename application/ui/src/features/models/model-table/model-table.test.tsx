import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { SchemaModel, SchemaTrainJob } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { durationBetween } from '../shared/duration';
import { ModelRow } from './model-table';

/**
 * jsdom does not implement `EventSource`, and MSW does not intercept it (no
 * EventSource interceptor exists in `@mswjs/interceptors`). `MetricsContent`
 * calls the real browser `EventSource` via `fetchSSE`, so it must be stubbed
 * here or the panel would throw `ReferenceError: EventSource is not defined`.
 * The stub resolves the stream immediately so the async iterator in
 * `fetchSSE` terminates without hanging the test.
 */
class ImmediatelyClosingEventSource {
    onmessage: ((event: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;

    constructor() {
        queueMicrotask(() => this.onmessage?.({ data: 'DONE' }));
    }

    close() {}
}

const model: SchemaModel = {
    id: 'model-1',
    name: 'pick-and-place',
    path: '/models/pick-and-place',
    policy: 'act',
    properties: {},
    project_id: 'project-1',
    dataset_id: null,
    snapshot_id: null,
    train_job_id: 'job-1',
    parent_model_id: null,
    version: 1,
    created_at: '2026-07-14T12:00:00Z',
    available_backends: [],
};

const trainingJob: SchemaTrainJob = {
    id: 'job-1',
    project_id: 'project-1',
    progress: 100,
    status: 'completed',
    message: 'Job created',
    start_time: '2026-07-14T10:00:00Z',
    end_time: '2026-07-14T10:30:00Z',
    created_at: '2026-07-14T09:00:00Z',
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

const modelDetailResponse = {
    model,
    exports: [],
    training_summary: null,
    hparams: null,
};

const renderModelRow = ({
    modelOverride,
    trainingJobOverride,
    onDelete = vi.fn(),
    onRetrain = vi.fn(),
    onViewLogs = vi.fn(),
}: {
    modelOverride?: Partial<SchemaModel>;
    trainingJobOverride?: SchemaTrainJob;
    onDelete?: () => void;
    onRetrain?: () => void;
    onViewLogs?: () => void;
} = {}) => {
    return render(
        <ModelRow
            model={{ ...model, ...modelOverride }}
            trainingJob={trainingJobOverride}
            onDelete={onDelete}
            onRetrain={onRetrain}
            onViewLogs={onViewLogs}
        />
    );
};

describe('ModelRow', () => {
    beforeAll(() => {
        vi.stubGlobal('EventSource', ImmediatelyClosingEventSource);
    });

    afterAll(() => {
        vi.unstubAllGlobals();
    });

    beforeEach(() => {
        server.use(
            http.get('/api/models/{model_id}', () => HttpResponse.json(modelDetailResponse)),
            http.get('/api/policies/backends', () => HttpResponse.json({}))
        );
    });

    it('renders the model name, formatted created date, duration, and uppercased architecture', () => {
        renderModelRow({ trainingJobOverride: trainingJob });

        const expectedDuration = durationBetween(trainingJob.start_time as string, trainingJob.end_time as string);

        expect(screen.getByText('pick-and-place')).toBeInTheDocument();
        expect(screen.getByText(new Date(model.created_at as string).toLocaleString())).toBeInTheDocument();
        expect(screen.getByText(expectedDuration)).toBeInTheDocument();
        expect(screen.getByText('ACT')).toBeInTheDocument();
    });

    it('falls back to the — placeholder when the training job has no start/end time', () => {
        renderModelRow({ trainingJobOverride: { ...trainingJob, start_time: null, end_time: null } });

        expect(screen.getByText('—')).toBeInTheDocument();
    });

    it('falls back to the — placeholder when there is no training job at all', () => {
        renderModelRow();

        expect(screen.getByText('—')).toBeInTheDocument();
    });

    it('renders the v{n} suffix only when version > 1', () => {
        const { rerender } = renderModelRow({ modelOverride: { version: 1 } });

        expect(screen.queryByText(/^v\d+$/)).not.toBeInTheDocument();

        rerender(
            <ModelRow
                model={{ ...model, version: 2 }}
                trainingJob={undefined}
                onDelete={vi.fn()}
                onRetrain={vi.fn()}
                onViewLogs={vi.fn()}
            />
        );

        expect(screen.getByText('v2')).toBeInTheDocument();
    });

    it('does not render the detail panel initially', () => {
        renderModelRow();

        expect(screen.queryByRole('tab', { name: 'Model formats' })).not.toBeInTheDocument();
    });

    it('reveals the panel tabs when the row is clicked', async () => {
        const user = userEvent.setup();
        renderModelRow();

        await user.click(screen.getByText('pick-and-place'));

        expect(await screen.findByRole('tab', { name: 'Model formats' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Model Metrics' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Training Datasets' })).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: 'Training Details' })).toBeInTheDocument();
    });

    it('does not collapse the row when a tab inside the panel is clicked', async () => {
        const user = userEvent.setup();
        renderModelRow();

        await user.click(screen.getByText('pick-and-place'));
        expect(await screen.findByRole('tab', { name: 'Model formats' })).toBeInTheDocument();

        await user.click(screen.getByRole('tab', { name: 'Model Metrics' }));

        expect(screen.getByRole('tab', { name: 'Model formats' })).toBeInTheDocument();
    });

    it('collapses an expanded row when it is clicked again', async () => {
        const user = userEvent.setup();
        renderModelRow();

        await user.click(screen.getByText('pick-and-place'));
        expect(await screen.findByRole('tab', { name: 'Model formats' })).toBeInTheDocument();

        await user.click(screen.getByText('pick-and-place'));

        expect(screen.queryByRole('tab', { name: 'Model formats' })).not.toBeInTheDocument();
    });

    it('shows Logs, Download, Retrain, Delete in the overflow menu, with Logs disabled when train_job_id is absent', async () => {
        const user = userEvent.setup();
        renderModelRow({ modelOverride: { train_job_id: null } });

        await user.click(screen.getByRole('button', { name: 'options' }));

        expect(screen.getByRole('menuitem', { name: 'Logs' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Download' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Retrain' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Logs' }).closest('[aria-disabled]')).toHaveAttribute(
            'aria-disabled',
            'true'
        );
    });

    it('renders the Run model button', () => {
        renderModelRow();

        expect(screen.getByRole('button', { name: 'Run model' })).toBeInTheDocument();
    });

    it('two rows can be expanded simultaneously', async () => {
        const user = userEvent.setup();
        const secondModel: SchemaModel = { ...model, id: 'model-2', name: 'stack-blocks' };

        render(
            <>
                <ModelRow
                    model={model}
                    trainingJob={undefined}
                    onDelete={vi.fn()}
                    onRetrain={vi.fn()}
                    onViewLogs={vi.fn()}
                />
                <ModelRow
                    model={secondModel}
                    trainingJob={undefined}
                    onDelete={vi.fn()}
                    onRetrain={vi.fn()}
                    onViewLogs={vi.fn()}
                />
            </>
        );

        await user.click(screen.getByText('pick-and-place'));
        await user.click(screen.getByText('stack-blocks'));

        expect(await screen.findAllByRole('tab', { name: 'Model formats' })).toHaveLength(2);
    });

    it('names the disclosure button "Show details for {model.name}" and exposes aria-expanded', async () => {
        const user = userEvent.setup();
        renderModelRow();

        const disclosureButton = screen.getByRole('button', { name: 'Show details for pick-and-place' });
        expect(disclosureButton).toHaveAttribute('aria-expanded', 'false');

        await user.click(disclosureButton);

        expect(await screen.findByRole('tab', { name: 'Model formats' })).toBeInTheDocument();
        expect(disclosureButton).toHaveAttribute('aria-expanded', 'true');
    });

    it('is expandable via the keyboard', async () => {
        const user = userEvent.setup();
        renderModelRow();

        await user.tab();
        expect(screen.getByRole('button', { name: 'Show details for pick-and-place' })).toHaveFocus();

        await user.keyboard('{Enter}');

        expect(await screen.findByRole('tab', { name: 'Model formats' })).toBeInTheDocument();
    });
});

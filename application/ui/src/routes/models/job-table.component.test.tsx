import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { render } from '../../test-utils/render';
import { TrainingRow } from './job-table.component';
import { SchemaTrainJob } from './train-model-dialog';

const baseJob: SchemaTrainJob = {
    id: 'f1b3f8f0-1c1a-4b1a-9c1a-1c1a1c1a1c1a',
    status: 'failed',
    progress: 0,
    message: '',
    start_time: null,
    end_time: null,
    extra_info: {},
    payload: {
        training_target: 'local',
        model_name: 'Test model',
        policy: 'act',
    },
} as unknown as SchemaTrainJob;

const renderRow = (job: SchemaTrainJob) =>
    render(<TrainingRow trainJob={job} onInterrupt={() => undefined} onViewLogs={() => undefined} />, {
        route: '/projects/p1/models',
        path: '/projects/:project_id/models',
    });

describe('JobMenu delete option', () => {
    it.each(['failed', 'canceled'] as const)('is enabled when job status is %s', async (status) => {
        const user = userEvent.setup();
        renderRow({ ...baseJob, status });

        await user.click(screen.getByRole('button', { name: 'Job options' }));

        expect(screen.getByRole('menuitem', { name: 'Delete' })).not.toHaveAttribute('aria-disabled', 'true');
    });

    it.each(['running', 'completed', 'pending'] as const)('is disabled when job status is %s', async (status) => {
        const user = userEvent.setup();
        renderRow({ ...baseJob, status });

        await user.click(screen.getByRole('button', { name: 'Job options' }));

        expect(screen.getByRole('menuitem', { name: 'Delete' })).toHaveAttribute('aria-disabled', 'true');
    });
});

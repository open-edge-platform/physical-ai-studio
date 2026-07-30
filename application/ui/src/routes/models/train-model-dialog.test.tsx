import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaModel } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { TrainModelDialog } from './train-model-dialog';

const projectId = 'b8b28d4f-e78f-48ad-afb8-03d060178a3c';
const remoteTrainerId = '16ea95a7-6f49-4c19-b22b-91cf89f8d34b';
const datasetId = '9f4e20fb-7dd8-4d2c-b207-a6d554311a12';

const remoteTrainer = {
    id: remoteTrainerId,
    name: 'managed-trainer',
    url: 'https://trainer.example.test/api',
    created_at: '2026-07-14T12:00:00Z',
};

const healthyRemoteTrainer = {
    remote_trainer_id: remoteTrainerId,
    status: 'healthy' as const,
    checked_at: '2026-07-16T12:00:00Z',
    latency_ms: 24,
    devices: [],
    reason_code: null,
};

const baseModel = {
    id: '9340adfd-9632-4c54-8acd-8304f9dfda91',
    name: 'Test model',
    dataset_id: datasetId,
    policy: 'act',
} as SchemaModel;

describe('TrainModelDialog', () => {
    it('does not submit a remote job when the final health check fails', async () => {
        const user = userEvent.setup();
        let healthCheckCount = 0;
        let jobSubmitted = false;

        server.use(
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
                            environment_id: 'ad5c311d-bdd7-4a1c-ad27-26c2775901e9',
                        },
                    ],
                })
            ),
            http.get('/api/system/devices/training', () =>
                HttpResponse.json({ mode: 'local', remote_available: true, devices: [] })
            ),
            http.get('/api/remote-trainers', () => HttpResponse.json([remoteTrainer])),
            http.get('/api/remote-trainers/{remote_trainer_id}/health', () => {
                healthCheckCount += 1;
                return healthCheckCount === 1
                    ? HttpResponse.json(healthyRemoteTrainer)
                    : HttpResponse.json({ detail: [] }, { status: 503 });
            }),
            http.post('/api/jobs:train', () => {
                jobSubmitted = true;
                return HttpResponse.json({}, { status: 201 });
            })
        );

        render(<TrainModelDialog baseModel={baseModel} close={() => undefined} />, {
            route: `/projects/${projectId}/models`,
            path: '/projects/:project_id/models',
        });

        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));
        await user.click(await screen.findByRole('option', { name: remoteTrainer.name }));
        await screen.findByText('Remote trainer selected');
        await user.click(screen.getByRole('button', { name: 'Train' }));

        await waitFor(() => expect(healthCheckCount).toBeGreaterThan(1));
        expect(jobSubmitted).toBe(false);
    });
});

import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { RemoteTrainersPage } from './remote-trainers-page';

const REMOTE_TRAINERS_PATH = '/api/remote-trainers';
const REMOTE_TRAINER_PATH = '/api/remote-trainers/{remote_trainer_id}';
const REMOTE_TRAINER_HEALTH_PATH = '/api/remote-trainers/{remote_trainer_id}/health';

const remoteTrainer = {
    id: 'b8b28d4f-e78f-48ad-afb8-03d060178a3c',
    name: 'managed-trainer',
    url: 'https://trainer.example.test/api',
    created_at: '2026-07-14T12:00:00Z',
};

const healthyTrainer = {
    remote_trainer_id: remoteTrainer.id,
    status: 'healthy' as const,
    checked_at: '2026-07-16T12:00:00Z',
    latency_ms: 24,
    devices: [{ type: 'cuda' as const, name: 'NVIDIA A100', memory: 85899345920, index: 0 }],
    storage: { total_bytes: 1_000_000_000_000, free_bytes: 600_000_000_000 },
    reason_code: null,
};

const unavailableTrainer = {
    ...remoteTrainer,
    id: 'b5a0da22-7066-426d-b0ae-6dfae2d983dc',
    name: 'unavailable-trainer',
};

describe('RemoteTrainersPage', () => {
    beforeEach(() => {
        server.use(http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json(healthyTrainer)));
    });

    it('shows configured remote trainers and their connection details', async () => {
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([remoteTrainer])));

        render(<RemoteTrainersPage />);

        expect(await screen.findByRole('heading', { name: 'Remote Trainers' })).toBeInTheDocument();
        expect(await screen.findAllByText('managed-trainer')).not.toHaveLength(0);
        expect(await screen.findByRole('heading', { name: 'managed-trainer' })).toBeInTheDocument();
        expect(await screen.findAllByText('Healthy')).not.toHaveLength(0);
        expect(screen.getByText(/ready for training requests/)).toBeInTheDocument();
        expect(screen.getByText('Trainer health endpoint')).toBeInTheDocument();
        expect(screen.getByText('Compute capability')).toBeInTheDocument();
        expect(screen.getAllByText('CUDA')).not.toHaveLength(0);
        expect(screen.getAllByText(/NVIDIA A100/)).not.toHaveLength(0);
        expect(screen.getByText('Storage capacity')).toBeInTheDocument();
        expect(screen.getAllByText(/558\.8 GB free of 931\.3 GB/)).not.toHaveLength(0);
    });

    it('attributes an invalid device report to compute capability', async () => {
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([remoteTrainer])),
            http.get(REMOTE_TRAINER_HEALTH_PATH, () =>
                HttpResponse.json({
                    ...healthyTrainer,
                    status: 'degraded',
                    devices: [],
                    reason_code: 'invalid_devices_response',
                })
            )
        );

        render(<RemoteTrainersPage />);

        expect(await screen.findAllByText('Healthy')).not.toHaveLength(0);
        const trainerHealthRow = (await screen.findByText('Trainer health endpoint')).closest('div');
        const computeRow = screen.getByText('Compute capability').closest('div');

        if (trainerHealthRow === null || computeRow === null) {
            throw new Error('Expected trainer health and compute capability rows to be rendered.');
        }

        expect(within(trainerHealthRow).getByText('Healthy')).toBeInTheDocument();
        expect(within(computeRow).getByText('Unknown')).toBeInTheDocument();
    });

    it('distinguishes a failed health request from an unreachable trainer', async () => {
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([remoteTrainer])),
            http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json({ detail: [] }, { status: 422 }))
        );

        render(<RemoteTrainersPage />);

        expect(await screen.findAllByText('Check failed')).not.toHaveLength(0);
        expect(screen.getAllByText('Studio could not complete the health check. Try again.')).not.toHaveLength(0);
    });

    it('shows a failed health check for an unselected trainer', async () => {
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([remoteTrainer, unavailableTrainer])),
            http.get(REMOTE_TRAINER_HEALTH_PATH, ({ params }) =>
                params.remote_trainer_id === unavailableTrainer.id
                    ? HttpResponse.json({ detail: [] }, { status: 503 })
                    : HttpResponse.json(healthyTrainer)
            )
        );

        render(<RemoteTrainersPage />);

        const unavailableCard = await screen.findByRole('button', { name: /unavailable-trainer/i });
        expect(await within(unavailableCard).findByText('Check failed')).toBeInTheDocument();
    });

    it('creates a configured remote trainer URL', async () => {
        const user = userEvent.setup();
        let trainers: (typeof remoteTrainer)[] = [];
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json(trainers)),
            http.post(REMOTE_TRAINERS_PATH, async ({ request }) => {
                const body = (await request.json()) as Pick<typeof remoteTrainer, 'name' | 'url'>;
                trainers = [{ ...body, id: remoteTrainer.id, created_at: remoteTrainer.created_at }];
                return HttpResponse.json(trainers[0], { status: 201 });
            })
        );

        render(<RemoteTrainersPage />);

        expect(await screen.findByText('No remote trainers are configured.')).toBeInTheDocument();
        await user.click(await screen.findByRole('button', { name: /new remote trainer/i }));
        const dialog = await screen.findByRole('dialog');
        const inputs = dialog.querySelectorAll('input');
        await user.type(inputs[0], remoteTrainer.name);
        await user.type(inputs[1], remoteTrainer.url);
        await user.click(screen.getByRole('button', { name: 'Add trainer' }));

        expect(await screen.findByRole('heading', { name: remoteTrainer.name })).toBeInTheDocument();
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });

    it('edits a configured remote trainer', async () => {
        const user = userEvent.setup();
        let trainer = remoteTrainer;
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([trainer])),
            http.patch(REMOTE_TRAINER_PATH, async ({ request }) => {
                const update = (await request.json()) as Partial<typeof remoteTrainer>;
                trainer = { ...trainer, ...update };
                return HttpResponse.json(trainer);
            })
        );

        render(<RemoteTrainersPage />);

        await user.click(await screen.findByRole('button', { name: `Edit ${remoteTrainer.name}` }));
        const dialog = await screen.findByRole('dialog');
        const nameInput = dialog.querySelectorAll('input')[0];
        await user.clear(nameInput);
        await user.type(nameInput, 'renamed-trainer');
        await user.click(screen.getByRole('button', { name: 'Save changes' }));

        expect(await screen.findByRole('heading', { name: 'renamed-trainer' })).toBeInTheDocument();
    });

    it('deletes a configured remote trainer', async () => {
        const user = userEvent.setup();
        let trainers: (typeof remoteTrainer)[] = [remoteTrainer];
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json(trainers)),
            http.delete(REMOTE_TRAINER_PATH, () => {
                trainers = [];
                return new HttpResponse(null, { status: 204 });
            })
        );

        render(<RemoteTrainersPage />);

        await user.click(await screen.findByRole('button', { name: `Delete ${remoteTrainer.name}` }));
        await user.click(await screen.findByRole('button', { name: 'Delete' }));
    });
});

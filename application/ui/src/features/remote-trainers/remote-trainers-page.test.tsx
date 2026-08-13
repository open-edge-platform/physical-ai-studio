import { screen, waitFor } from '@testing-library/react';
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

describe('RemoteTrainersPage', () => {
    beforeEach(() => {
        server.use(http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json(healthyTrainer)));
    });

    it('shows configured remote trainers', async () => {
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([remoteTrainer])));

        render(<RemoteTrainersPage />);

        expect(await screen.findByRole('heading', { name: 'Remote Trainers' })).toBeInTheDocument();
        expect(await screen.findAllByText('managed-trainer')).not.toHaveLength(0);
        expect(await screen.findByRole('button', { name: /show details for managed-trainer/i })).toBeInTheDocument();
    });

    it('shows an empty state when no remote trainers are configured', async () => {
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([])));

        render(<RemoteTrainersPage />);

        expect(await screen.findByText('No remote trainers are configured.')).toBeInTheDocument();
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

        expect(await screen.findByRole('button', { name: /show details for managed-trainer/i })).toBeInTheDocument();
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

        await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
        await user.click(await screen.findByRole('menuitem', { name: 'Edit' }));
        const dialog = await screen.findByRole('dialog');
        const nameInput = dialog.querySelectorAll('input')[0];
        await user.clear(nameInput);
        await user.type(nameInput, 'renamed-trainer');
        await user.click(screen.getByRole('button', { name: 'Save changes' }));

        expect(await screen.findByRole('button', { name: /show details for renamed-trainer/i })).toBeInTheDocument();
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

        await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));
        await user.click(await screen.findByRole('button', { name: 'Delete' }));

        expect(await screen.findByText('No remote trainers are configured.')).toBeInTheDocument();
        expect(screen.queryByText(remoteTrainer.name)).not.toBeInTheDocument();
    });
});

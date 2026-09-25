import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { TrainingTargetsPage } from './training-targets-page';

const REMOTE_TRAINERS_PATH = '/api/remote-trainers';
const REMOTE_TRAINER_PATH = '/api/remote-trainers/{remote_trainer_id}';
const REMOTE_TRAINER_HEALTH_PATH = '/api/remote-trainers/{remote_trainer_id}/health';

const remoteTrainer = {
    id: 'b8b28d4f-e78f-48ad-afb8-03d060178a3c',
    name: 'managed-trainer',
    connection_mode: 'direct' as const,
    url: 'https://trainer.example.test/api',
    ssh_remote_port: null,
    ssh_local_port: null,
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

describe('TrainingTargetsPage', () => {
    beforeEach(() => {
        server.use(
            http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json(healthyTrainer)),
            http.get('/api/remote-servers/feature-status', () => HttpResponse.json({ network_exposed: false }))
        );
    });

    it('shows configured remote trainers', async () => {
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([remoteTrainer])));

        render(<TrainingTargetsPage />);

        expect(await screen.findByText('Configure and monitor where training jobs run.')).toBeInTheDocument();
        expect(await screen.findAllByText('managed-trainer')).not.toHaveLength(0);
        expect(await screen.findByRole('button', { name: /show details for managed-trainer/i })).toBeInTheDocument();
    });

    it('opens the remote trainer form without a target type switch', async () => {
        const user = userEvent.setup();
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([])));

        render(<TrainingTargetsPage />);

        await user.click(await screen.findByRole('button', { name: /new training target/i }));
        const dialog = await screen.findByRole('dialog', { name: /add remote trainer/i });
        expect(within(dialog).queryByText('Target type')).not.toBeInTheDocument();
        expect(within(dialog).getByRole('tab', { name: 'SSH tunnel' })).toBeInTheDocument();
    });

    it('disables SSH and defaults to a direct URL when SSH is unavailable', async () => {
        const user = userEvent.setup();
        let created: Record<string, unknown> | undefined;
        let aliasRequests = 0;
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([])),
            http.get('/api/remote-servers/feature-status', () => HttpResponse.json({ network_exposed: true })),
            http.get('/api/remote-servers/aliases', () => {
                aliasRequests++;
                return HttpResponse.json([], { status: 503 });
            }),
            http.post(REMOTE_TRAINERS_PATH, async ({ request }) => {
                created = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(remoteTrainer, { status: 201 });
            })
        );

        render(<TrainingTargetsPage />);

        expect(await screen.findByText(/SSH training targets are unavailable/i)).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: /new training target/i }));
        const dialog = await screen.findByRole('dialog', { name: /add remote trainer/i });
        expect(within(dialog).getByRole('tab', { name: 'SSH tunnel' })).toHaveAttribute('aria-disabled', 'true');
        expect(within(dialog).getByRole('tab', { name: 'Trainer URL' })).toHaveAttribute('aria-selected', 'true');
        expect(within(dialog).queryByRole('button', { name: /add ssh connection/i })).not.toBeInTheDocument();
        await user.type(within(dialog).getByRole('textbox', { name: /^Name/ }), 'direct-trainer');
        await user.type(within(dialog).getByRole('textbox', { name: /trainer url/i }), 'https://trainer.example.test');
        await user.click(within(dialog).getByRole('button', { name: 'Add trainer' }));
        await waitFor(() => expect(created).toBeDefined());
        expect(created).toMatchObject({ connection_mode: 'direct', ssh_connection: null });
        expect(aliasRequests).toBe(0);
    });

    it('fails closed when SSH availability cannot be checked', async () => {
        const user = userEvent.setup();
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([])),
            http.get('/api/remote-servers/feature-status', () =>
                HttpResponse.json({ network_exposed: true }, { status: 503 })
            )
        );

        render(<TrainingTargetsPage />);

        await user.click(await screen.findByRole('button', { name: /new training target/i }));
        const dialog = await screen.findByRole('dialog', { name: /add remote trainer/i });
        expect(within(dialog).getByRole('tab', { name: 'SSH tunnel' })).toHaveAttribute('aria-disabled', 'true');
        expect(within(dialog).getByRole('tab', { name: 'Trainer URL' })).toHaveAttribute('aria-selected', 'true');
    });

    it('creates a configured remote trainer URL', async () => {
        const user = userEvent.setup();
        let trainers: (typeof remoteTrainer)[] = [];
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json(trainers)),
            http.post(REMOTE_TRAINERS_PATH, async ({ request }) => {
                const body = (await request.json()) as Pick<typeof remoteTrainer, 'name' | 'url'>;
                trainers = [
                    {
                        ...body,
                        id: remoteTrainer.id,
                        connection_mode: 'direct',
                        ssh_remote_port: null,
                        ssh_local_port: null,
                        created_at: remoteTrainer.created_at,
                    },
                ];
                return HttpResponse.json(trainers[0], { status: 201 });
            })
        );

        render(<TrainingTargetsPage />);

        await user.click(await screen.findByRole('button', { name: /new training target/i }));
        const dialog = await screen.findByRole('dialog');
        await user.type(within(dialog).getByLabelText(/name/i), remoteTrainer.name);
        await user.click(within(dialog).getByRole('tab', { name: 'Trainer URL' }));
        await user.type(within(dialog).getByRole('textbox', { name: /trainer url/i }), remoteTrainer.url);
        await user.click(within(dialog).getByRole('button', { name: 'Add trainer' }));

        expect(await screen.findByRole('button', { name: /show details for managed-trainer/i })).toBeInTheDocument();
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
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

        render(<TrainingTargetsPage />);

        await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));
        await user.click(await screen.findByRole('button', { name: 'Delete' }));

        expect(await screen.findByText('No training targets are configured.')).toBeInTheDocument();
    });
});

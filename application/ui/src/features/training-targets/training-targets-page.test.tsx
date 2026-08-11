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
const REMOTE_SERVERS_PATH = '/api/remote-servers';
const REMOTE_SERVER_PATH = '/api/remote-servers/{remote_server_id}';
const REMOTE_SERVER_ALIASES_PATH = '/api/remote-servers/aliases';
const REMOTE_SERVER_STATUS_PATH = '/api/remote-servers/{remote_server_id}/status';

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

const remoteServer = {
    id: 'f1a2b3c4-d5e6-47a8-99b0-1234567890ab',
    name: 'lambda-a100',
    ssh_host_alias: 'gpu-01',
    device_type: 'cuda' as const,
    last_check_status: 'unknown' as const,
};

const aliasOption = { alias: 'gpu-01', hostname: 'gpu-01.lab.internal', port: 22, user: 'ubuntu' };

const healthyServerStatus = {
    remote_server_id: remoteServer.id,
    status: 'healthy' as const,
    device_type: 'cuda',
    checks: [],
    checked_at: '2026-08-07T12:00:00Z',
    waiting_for_gpu: false,
};

describe('TrainingTargetsPage', () => {
    beforeEach(() => {
        server.use(
            http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json(healthyTrainer)),
            http.get(REMOTE_SERVERS_PATH, () => HttpResponse.json([])),
            http.get(REMOTE_SERVER_ALIASES_PATH, () => HttpResponse.json([aliasOption])),
            http.get(REMOTE_SERVER_STATUS_PATH, () => HttpResponse.json(healthyServerStatus))
        );
    });

    it('shows configured remote trainers', async () => {
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([remoteTrainer])));

        render(<TrainingTargetsPage />);

        expect(await screen.findByText('Configure and monitor where training jobs run.')).toBeInTheDocument();
        expect(await screen.findAllByText('managed-trainer')).not.toHaveLength(0);
        expect(await screen.findByRole('button', { name: /show details for managed-trainer/i })).toBeInTheDocument();
    });

    it('shows configured SSH servers alongside direct-URL trainers', async () => {
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([remoteTrainer])),
            http.get(REMOTE_SERVERS_PATH, () => HttpResponse.json([remoteServer]))
        );

        render(<TrainingTargetsPage />);

        expect(await screen.findByText('managed-trainer')).toBeInTheDocument();
        expect(await screen.findByText('lambda-a100')).toBeInTheDocument();
    });

    it('shows an empty state when no training targets are configured', async () => {
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([])));

        render(<TrainingTargetsPage />);

        expect(await screen.findByText('No training targets are configured.')).toBeInTheDocument();
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

        render(<TrainingTargetsPage />);

        expect(await screen.findByText('No training targets are configured.')).toBeInTheDocument();
        await user.click(await screen.findByRole('button', { name: /new training target/i }));
        const dialog = await screen.findByRole('dialog');
        await user.type(within(dialog).getByLabelText(/name/i), remoteTrainer.name);
        await user.click(within(dialog).getByRole('button', { name: 'Direct trainer URL' }));
        await user.type(within(dialog).getByLabelText(/trainer url/i), remoteTrainer.url);
        await user.click(within(dialog).getByRole('button', { name: 'Add trainer' }));

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

        render(<TrainingTargetsPage />);

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

        render(<TrainingTargetsPage />);

        await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));
        await user.click(await screen.findByRole('button', { name: 'Delete' }));

        expect(await screen.findByText('No training targets are configured.')).toBeInTheDocument();
        expect(screen.queryByText(remoteTrainer.name)).not.toBeInTheDocument();
    });

    it('creates an SSH server training target', async () => {
        const user = userEvent.setup();
        let servers: (typeof remoteServer)[] = [];
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([])),
            http.get(REMOTE_SERVERS_PATH, () => HttpResponse.json(servers)),
            http.post(REMOTE_SERVERS_PATH, async ({ request }) => {
                const body = (await request.json()) as Pick<
                    typeof remoteServer,
                    'name' | 'ssh_host_alias' | 'device_type'
                >;
                servers = [{ ...body, id: remoteServer.id, last_check_status: 'unknown' }];
                return HttpResponse.json(servers[0], { status: 201 });
            })
        );

        render(<TrainingTargetsPage />);

        expect(await screen.findByText('No training targets are configured.')).toBeInTheDocument();
        await user.click(await screen.findByRole('button', { name: /new training target/i }));

        const dialog = await screen.findByRole('dialog');
        await user.type(within(dialog).getByLabelText(/name/i), remoteServer.name);
        await user.click(within(dialog).getByRole('button', { name: /ssh host/i }));
        await user.click(await screen.findByRole('option', { name: aliasOption.alias }));
        await user.click(within(dialog).getByRole('button', { name: /device type/i }));
        await user.click(await screen.findByRole('option', { name: 'CUDA' }));
        await user.click(within(dialog).getByRole('button', { name: 'Verify & save' }));

        expect(await screen.findByRole('button', { name: /show details for lambda-a100/i })).toBeInTheDocument();
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });

    it('deletes a configured SSH server', async () => {
        const user = userEvent.setup();
        let servers: (typeof remoteServer)[] = [remoteServer];
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([])),
            http.get(REMOTE_SERVERS_PATH, () => HttpResponse.json(servers)),
            http.delete(REMOTE_SERVER_PATH, () => {
                servers = [];
                return new HttpResponse(null, { status: 204 });
            })
        );

        render(<TrainingTargetsPage />);

        await user.click(await screen.findByRole('button', { name: `More actions ${remoteServer.name}` }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));
        await user.click(await screen.findByRole('button', { name: 'Delete' }));

        expect(await screen.findByText('No training targets are configured.')).toBeInTheDocument();
        expect(screen.queryByText(remoteServer.name)).not.toBeInTheDocument();
    });
});

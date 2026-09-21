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
                const body = (await request.json()) as Pick<
                    typeof remoteTrainer,
                    'name' | 'connection_mode' | 'url' | 'ssh_remote_port' | 'ssh_local_port'
                >;
                trainers = [{ ...body, id: remoteTrainer.id, created_at: remoteTrainer.created_at }];
                return HttpResponse.json(trainers[0], { status: 201 });
            })
        );

        render(<RemoteTrainersPage />);

        expect(await screen.findByText('No remote trainers are configured.')).toBeInTheDocument();
        await user.click(await screen.findByRole('button', { name: /new remote trainer/i }));
        await screen.findByRole('dialog');
        const deployStackLink = screen.getByRole('link', { name: 'Deploy AWS stack' });
        expect(screen.getByText('Need a new remote trainer?')).toBeInTheDocument();
        expect(deployStackLink).toHaveAttribute('target', '_blank');
        expect(deployStackLink).toHaveAttribute('rel', 'noopener noreferrer');
        expect(deployStackLink).toHaveAttribute(
            'href',
            'https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1' +
                '#/stacks/create/review?' +
                'templateURL=https%3A%2F%2Fphysical-ai-studio.s3.eu-west-1.amazonaws.com%2Faws-cf-templates%2Fremote-trainer.yaml' +
                '&stackName=physical-ai-studio-remote-trainer'
        );
        expect(screen.getByText('Address exposed by the remote trainer.')).toBeInTheDocument();
        expect(screen.getByText('Enter the URL of a trainer that Studio can reach directly.')).toBeInTheDocument();
        await user.type(screen.getByRole('textbox', { name: /^Name/ }), remoteTrainer.name);
        await user.type(screen.getByRole('textbox', { name: /^Trainer URL/ }), remoteTrainer.url);
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
        expect(screen.queryByRole('link', { name: 'Deploy AWS stack' })).not.toBeInTheDocument();
        const nameInput = dialog.querySelectorAll('input')[0];
        await user.clear(nameInput);
        await user.type(nameInput, 'renamed-trainer');
        await user.click(screen.getByRole('button', { name: 'Save changes' }));

        expect(await screen.findByRole('button', { name: /show details for renamed-trainer/i })).toBeInTheDocument();
    });

    it('shows the SSH tunnel connection fields', async () => {
        const user = userEvent.setup();
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([])));

        render(<RemoteTrainersPage />);

        await user.click(await screen.findByRole('button', { name: /new remote trainer/i }));
        await user.click(screen.getByRole('tab', { name: /ssh tunnel/i }));

        expect(screen.getByText('Connect through SSH when the trainer is not directly reachable.')).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: /connection details/i })).toHaveAttribute('aria-selected', 'true');
        expect(screen.getByRole('textbox', { name: /^User/ })).toHaveValue('ec2-user');
        expect(screen.getByRole('textbox', { name: /local port/i })).toHaveValue('8001');
        expect(screen.queryByRole('textbox', { name: /trainer url/i })).not.toBeInTheDocument();
    });

    it('creates a remote trainer with an SSH tunnel configured via a new manual host', async () => {
        const user = userEvent.setup();
        const fingerprint = 'SHA256:first-seen-host-key';
        let trainers: Record<string, unknown>[] = [];
        let aliasCreateCount = 0;
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json(trainers as (typeof remoteTrainer)[])),
            http.post(REMOTE_TRAINERS_PATH, async ({ request }) => {
                if (request.headers.get('accepted-host-key-fingerprint') !== fingerprint) {
                    return HttpResponse.json<Record<string, unknown>>(
                        {
                            error_code: 'ssh_host_key_confirmation_required',
                            message: 'Confirm the SSH host key fingerprint before connecting.',
                            http_status: 428,
                            fingerprint,
                        },
                        { status: 428 }
                    );
                }
                const body = (await request.json()) as Record<string, unknown>;
                trainers = [{ ...body, id: remoteTrainer.id, created_at: remoteTrainer.created_at }];
                return HttpResponse.json(trainers[0], { status: 201 });
            }),
            http.post('/api/remote-servers/aliases', () => {
                aliasCreateCount += 1;
                return HttpResponse.json({}, { status: 201 });
            })
        );

        render(<RemoteTrainersPage />);

        expect(await screen.findByText('No remote trainers are configured.')).toBeInTheDocument();
        await user.click(await screen.findByRole('button', { name: /new remote trainer/i }));
        await user.type(screen.getByRole('textbox', { name: /^Name/ }), remoteTrainer.name);
        await user.click(screen.getByRole('tab', { name: /ssh tunnel/i }));
        expect(screen.queryByRole('textbox', { name: /ssh host alias/i })).not.toBeInTheDocument();
        await user.type(screen.getByRole('textbox', { name: /^Host/ }), 'gpu.example.test');
        await user.clear(screen.getByRole('textbox', { name: /^Port/ }));
        await user.type(screen.getByRole('textbox', { name: /^Port/ }), '2222');
        await user.clear(screen.getByRole('textbox', { name: /^User/ }));
        await user.type(screen.getByRole('textbox', { name: /^User/ }), 'trainer');
        await user.type(screen.getByRole('textbox', { name: /key path/i }), '~/.ssh/trainer');

        expect(screen.getByRole('button', { name: 'Add trainer' })).toBeEnabled();
        await user.click(screen.getByRole('button', { name: 'Add trainer' }));
        expect(await screen.findByRole('dialog', { name: 'Verify SSH server' })).toHaveTextContent(fingerprint);
        await user.click(screen.getByRole('button', { name: 'Trust host' }));

        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(aliasCreateCount).toBe(0);
        expect(trainers[0]).toMatchObject({
            connection_mode: 'ssh',
            url: null,
            ssh_host_alias: null,
            ssh_connection: {
                hostname: 'gpu.example.test',
                port: 2222,
                user: 'trainer',
                identity_file: '~/.ssh/trainer',
            },
            ssh_remote_port: 8001,
            ssh_local_port: 8001,
        });
    });

    it('restores a manually configured SSH host when editing a remote trainer', async () => {
        const user = userEvent.setup();
        const fingerprint = 'SHA256:updated-host-key';
        const manualTrainer = {
            ...remoteTrainer,
            connection_mode: 'ssh' as const,
            url: 'http://127.0.0.1:8001',
            ssh_connection: {
                hostname: 'gpu.example.test',
                port: 2222,
                user: null,
                identity_file: '~/.ssh/trainer',
            },
            ssh_remote_port: 8001,
            ssh_local_port: 8001,
        };
        let aliasCreateCount = 0;
        let update: Record<string, unknown> | undefined;
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([manualTrainer])),
            http.post('/api/remote-servers/aliases', () => {
                aliasCreateCount += 1;
                return HttpResponse.json({}, { status: 201 });
            }),
            http.patch(REMOTE_TRAINER_PATH, async ({ request }) => {
                if (request.headers.get('accepted-host-key-fingerprint') !== fingerprint) {
                    return HttpResponse.json<Record<string, unknown>>(
                        {
                            error_code: 'ssh_host_key_confirmation_required',
                            message: 'Confirm the SSH host key fingerprint before connecting.',
                            http_status: 428,
                            fingerprint,
                        },
                        { status: 428 }
                    );
                }
                update = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(manualTrainer);
            })
        );

        render(<RemoteTrainersPage />);

        await user.click(await screen.findByRole('button', { name: `More actions ${manualTrainer.name}` }));
        await user.click(await screen.findByRole('menuitem', { name: 'Edit' }));

        expect(await screen.findByRole('tab', { name: /connection details/i })).toHaveAttribute(
            'aria-selected',
            'true'
        );
        expect(await screen.findByRole('textbox', { name: /^Host/ })).toHaveValue('gpu.example.test');
        expect(screen.getByRole('textbox', { name: /^Port/ })).toHaveValue('2,222');
        expect(screen.getByRole('textbox', { name: /^User/ })).toHaveValue('');
        expect(screen.getByRole('textbox', { name: /key path/i })).toHaveValue('~/.ssh/trainer');

        await user.click(screen.getByRole('button', { name: 'Save changes' }));
        expect(await screen.findByRole('dialog', { name: 'Verify SSH server' })).toHaveTextContent(fingerprint);
        await user.click(screen.getByRole('button', { name: 'Trust host' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(aliasCreateCount).toBe(0);
        expect(update).toMatchObject({ ssh_connection: { user: null } });
    });

    it('creates a remote trainer with an SSH tunnel picked from the SSH config', async () => {
        const user = userEvent.setup();
        let trainers: Record<string, unknown>[] = [];
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json(trainers as (typeof remoteTrainer)[])),
            http.get('/api/remote-servers/aliases', () =>
                HttpResponse.json([{ alias: 'gpu-box', hostname: '10.0.0.5', port: 22, user: 'trainer' }])
            ),
            http.post(REMOTE_TRAINERS_PATH, async ({ request }) => {
                const body = (await request.json()) as Record<string, unknown>;
                trainers = [{ ...body, id: remoteTrainer.id, created_at: remoteTrainer.created_at }];
                return HttpResponse.json(trainers[0], { status: 201 });
            })
        );

        render(<RemoteTrainersPage />);

        expect(await screen.findByText('No remote trainers are configured.')).toBeInTheDocument();
        await user.click(await screen.findByRole('button', { name: /new remote trainer/i }));
        await user.type(screen.getByRole('textbox', { name: /^Name/ }), remoteTrainer.name);
        await user.click(screen.getByRole('tab', { name: /ssh tunnel/i }));
        await user.click(screen.getByRole('tab', { name: /config alias/i }));

        await user.click(await screen.findByRole('button', { name: /select…?/i }));
        await user.click(await screen.findByRole('option', { name: /gpu-box/i }));
        await user.tab();

        await user.click(screen.getByRole('button', { name: 'Add trainer' }));

        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(trainers[0]).toMatchObject({
            connection_mode: 'ssh',
            url: null,
            ssh_host_alias: 'gpu-box',
            ssh_connection: null,
            ssh_remote_port: 8001,
            ssh_local_port: 8001,
        });
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

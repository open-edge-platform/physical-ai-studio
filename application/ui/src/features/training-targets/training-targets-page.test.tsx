import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaRemoteTrainer } from '../../api/openapi-spec';
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
    connection: { connection_mode: 'direct' as const, url: 'https://trainer.example.test/api' },
    created_at: '2026-07-14T12:00:00Z',
};

const sshTrainer = (alias: string) => ({
    ...remoteTrainer,
    connection_mode: 'ssh' as const,
    url: 'http://127.0.0.1:8001',
    connection: {
        connection_mode: 'ssh' as const,
        ssh_host_alias: alias,
        ssh_connection: null,
        ssh_remote_port: 8001,
        ssh_local_port: 8001,
    },
});

const healthyTrainer = {
    remote_trainer_id: remoteTrainer.id,
    status: 'healthy' as const,
    checked_at: '2026-07-16T12:00:00Z',
    latency_ms: 24,
    devices: [{ type: 'cuda' as const, name: 'NVIDIA A100', memory: 85899345920, index: 0 }],
    storage: { total_bytes: 1_000_000_000_000, free_bytes: 600_000_000_000 },
    reason_code: null,
};

const awsTrainer: SchemaRemoteTrainer = {
    ...remoteTrainer,
    id: 'aws-instance',
    name: 'AWS Provider',
    connection_mode: 'aws_batch',
    url: null,
    connection: {
        connection_mode: 'aws_batch',
        schema_version: 1,
        region: 'eu-west-1',
        studio_role_arn: 'arn:aws:iam::123456789012:role/studio',
        configuration_uri: 's3://config-bucket/studio-config.json',
        bucket: 'jobs-bucket',
        targets: { 'g4dn.xlarge': { queue: 'queue-name', job_definition: 'definition-name' } },
    },
};

const providerSchema = [
    {
        id: 'aws' as const,
        title: 'AWS',
        fields: [
            {
                name: 'configuration_uri',
                title: 'S3 configuration file URI',
                pattern: '^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/[^?#]+$',
            },
        ],
    },
];

describe('TrainingTargetsPage', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/remote-trainers/providers', () => HttpResponse.json(providerSchema)),
            http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json(healthyTrainer)),
            http.get('/api/remote-servers/feature-status', () => HttpResponse.json({ network_exposed: false }))
        );
    });

    it('shows configured remote trainers', async () => {
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([remoteTrainer])));

        render(<TrainingTargetsPage />);

        expect(await screen.findByRole('heading', { name: 'Training targets' })).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Add a training target' })).toBeInTheDocument();
        for (const card of screen.getAllByRole('button', { name: /^Add training target:/ })) {
            expect(card).not.toHaveTextContent('Add instance');
        }
        for (const [title, description] of [
            ['Self Managed Remote Trainer', 'Connect to a trainer service you deploy and manage.'],
            ['SSH Remote Trainer', 'Let Studio set up and manage a trainer over SSH.'],
            ['AWS Provider', 'Run training jobs on on-demand GPU compute on AWS.'],
        ]) {
            const card = screen.getByRole('button', { name: `Add training target: ${title}` });
            expect(within(card).getByText(description)).toBeInTheDocument();
            expect(card).toHaveAccessibleDescription(description);
        }
        expect(await screen.findAllByText('managed-trainer')).not.toHaveLength(0);
        expect(await screen.findByRole('button', { name: /show details for managed-trainer/i })).toBeInTheDocument();
    });

    it('opens the SSH card with only SSH configuration', async () => {
        const user = userEvent.setup();
        server.use(http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([])));

        render(<TrainingTargetsPage />);

        const card = await screen.findByRole('button', { name: 'Add training target: SSH Remote Trainer' });
        await waitFor(() => expect(card).toBeEnabled());
        await user.click(card);
        const dialog = await screen.findByRole('dialog', { name: 'Add training target' });
        expect(within(dialog).queryByText('Target type')).not.toBeInTheDocument();
        expect(within(dialog).getByRole('textbox', { name: /Remote port/i })).toBeInTheDocument();
        expect(within(dialog).queryByRole('tab', { name: 'Trainer URL' })).not.toBeInTheDocument();
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
        expect(screen.getByRole('button', { name: 'Add training target: SSH Remote Trainer' })).toBeDisabled();
        await user.click(screen.getByRole('button', { name: 'Add training target: Self Managed Remote Trainer' }));
        const dialog = await screen.findByRole('dialog', { name: 'Add training target' });
        expect(within(dialog).queryByRole('button', { name: /add ssh connection/i })).not.toBeInTheDocument();
        await user.type(within(dialog).getByRole('textbox', { name: /^Name/ }), 'direct-trainer');
        await user.type(within(dialog).getByRole('textbox', { name: /trainer url/i }), 'https://trainer.example.test');
        await user.click(within(dialog).getByRole('button', { name: 'Add training target' }));
        await waitFor(() => expect(created).toBeDefined());
        expect(created).toMatchObject({ connection: { connection_mode: 'direct' } });
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

        await user.click(
            await screen.findByRole('button', { name: 'Add training target: Self Managed Remote Trainer' })
        );
        const dialog = await screen.findByRole('dialog', { name: 'Add training target' });
        expect(within(dialog).getByRole('textbox', { name: /trainer url/i })).toBeInTheDocument();
        expect(within(dialog).queryByRole('tab', { name: 'SSH tunnel' })).not.toBeInTheDocument();
    });

    it.each(['healthy', 'reboot_required'])('hides SSH setup actions when SSH is disabled (%s)', async (reason) => {
        const user = userEvent.setup();
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([sshTrainer('gpu')])),
            http.get(REMOTE_TRAINER_HEALTH_PATH, () =>
                HttpResponse.json({
                    ...healthyTrainer,
                    status: reason === 'healthy' ? 'healthy' : 'degraded',
                    reason_code: reason === 'healthy' ? null : reason,
                })
            ),
            http.get('/api/remote-servers/feature-status', () => HttpResponse.json({ network_exposed: true }))
        );
        render(<TrainingTargetsPage />);

        await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
        expect(screen.queryByRole('menuitem', { name: 'Install prerequisites' })).not.toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Reboot to finish setup' })).not.toBeInTheDocument();
    });

    it('creates a configured remote trainer URL', async () => {
        const user = userEvent.setup();
        let trainers: (typeof remoteTrainer)[] = [];
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json(trainers)),
            http.post(REMOTE_TRAINERS_PATH, async ({ request }) => {
                const body = (await request.json()) as { name: string; connection: { url: string } };
                trainers = [
                    {
                        id: remoteTrainer.id,
                        name: body.name,
                        connection_mode: 'direct',
                        url: body.connection.url,
                        connection: { connection_mode: 'direct', url: body.connection.url },
                        created_at: remoteTrainer.created_at,
                    },
                ];
                return HttpResponse.json(trainers[0], { status: 201 });
            })
        );

        render(<TrainingTargetsPage />);

        await user.click(
            await screen.findByRole('button', { name: 'Add training target: Self Managed Remote Trainer' })
        );
        const dialog = await screen.findByRole('dialog');
        await user.type(within(dialog).getByLabelText(/name/i), remoteTrainer.name);
        await user.type(within(dialog).getByRole('textbox', { name: /trainer url/i }), remoteTrainer.url);
        await user.click(within(dialog).getByRole('button', { name: 'Add training target' }));

        expect(await screen.findByRole('button', { name: /show details for managed-trainer/i })).toBeInTheDocument();
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });

    it('shows the startup phase without shifting the status dot while pulling an image', async () => {
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([sshTrainer('xpu')])),
            http.get(REMOTE_TRAINER_HEALTH_PATH, () =>
                HttpResponse.json({ ...healthyTrainer, status: 'starting', reason_code: 'Pulling trainer image' })
            )
        );

        render(<TrainingTargetsPage />);

        expect(await screen.findAllByText('Starting: Pulling trainer image')).not.toHaveLength(0);
        expect(screen.queryByLabelText('Trainer setup in progress')).not.toBeInTheDocument();
    });

    it('offers installation for an existing SSH target and disables confirmation while submitting', async () => {
        const user = userEvent.setup();
        let installs = 0;
        let finishRequest: () => void = () => {};
        const requestPending = new Promise<void>((resolve) => {
            finishRequest = resolve;
        });
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([sshTrainer('gpu')])),
            http.post('/api/remote-trainers/{remote_trainer_id}/install-prerequisites', async () => {
                installs++;
                await requestPending;
                return new HttpResponse(null, { status: 202 });
            })
        );
        render(<TrainingTargetsPage />);

        await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
        await user.click(await screen.findByRole('menuitem', { name: 'Install prerequisites' }));
        expect(installs).toBe(0);
        const installButton = within(
            await screen.findByRole('alertdialog', { name: 'Install host prerequisites' })
        ).getByRole('button', { name: 'Install' });
        await user.click(installButton);
        await waitFor(() => expect(installs).toBe(1));
        expect(installButton).toBeDisabled();
        finishRequest();
        await waitFor(() =>
            expect(screen.queryByRole('alertdialog', { name: 'Install host prerequisites' })).not.toBeInTheDocument()
        );
    });

    it.each(['reboot_required', 'nvidia_driver_unavailable'])(
        'offers a separate reboot confirmation for %s',
        async (reason) => {
            const user = userEvent.setup();
            let reboots = 0;
            server.use(
                http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([sshTrainer('gpu')])),
                http.get(REMOTE_TRAINER_HEALTH_PATH, () =>
                    HttpResponse.json({ ...healthyTrainer, status: 'degraded', reason_code: reason })
                ),
                http.post('/api/remote-trainers/{remote_trainer_id}/reboot-after-install', () => {
                    reboots++;
                    return new HttpResponse(null, { status: 202 });
                })
            );
            render(<TrainingTargetsPage />);

            await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
            await user.click(await screen.findByRole('menuitem', { name: 'Reboot to finish setup' }));
            expect(reboots).toBe(0);
            await user.click(
                within(await screen.findByRole('alertdialog', { name: 'Reboot SSH host' })).getByRole('button', {
                    name: 'Reboot host',
                })
            );
            await waitFor(() => expect(reboots).toBe(1));
        }
    );

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

        await user.click(await screen.findByRole('button', { name: `Remove ${remoteTrainer.name}` }));
        const confirmation = await screen.findByRole('alertdialog', { name: 'Remove training target' });
        expect(within(confirmation).getByText(`Remove ${remoteTrainer.name}?`)).toBeInTheDocument();
        await user.click(within(confirmation).getByRole('button', { name: 'Remove' }));

        expect(await screen.findByText('No training targets added')).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Training targets' })).toBeInTheDocument();
        expect(screen.getAllByRole('button', { name: /^Add training target:/ })).toHaveLength(3);
    });

    it('shows multiple instances of the same type alongside other types', async () => {
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () =>
                HttpResponse.json([
                    remoteTrainer,
                    { ...remoteTrainer, id: 'second', name: 'second trainer' },
                    awsTrainer,
                ])
            )
        );
        render(<TrainingTargetsPage />);
        expect(await screen.findByRole('heading', { name: 'Training targets' })).toBeInTheDocument();
        expect(screen.getAllByRole('button', { name: /^Edit / })).toHaveLength(3);
        expect(screen.getAllByRole('button', { name: /^Remove / })).toHaveLength(3);
        expect(screen.getAllByRole('button', { name: /^Add training target:/ })).toHaveLength(3);
    });

    it('creates the AWS provider from only an S3 configuration URI', async () => {
        const user = userEvent.setup();
        let trainers: SchemaRemoteTrainer[] = [];
        let resolution: unknown;
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json(trainers)),
            http.post('/api/remote-trainers/providers/aws/configuration', async ({ request }) => {
                resolution = await request.json();
                return HttpResponse.json(
                    awsTrainer.connection as Extract<
                        SchemaRemoteTrainer['connection'],
                        { connection_mode: 'aws_batch' }
                    >
                );
            }),
            http.post(REMOTE_TRAINERS_PATH, async ({ request }) => {
                const body = await request.json();
                trainers = [{ ...awsTrainer, ...body }];
                return HttpResponse.json(trainers[0], { status: 201 });
            })
        );
        render(<TrainingTargetsPage />);
        await user.click(await screen.findByRole('button', { name: 'Add training target: AWS Provider' }));
        const dialog = await screen.findByRole('dialog', { name: 'Add training target' });
        expect(within(dialog).getByRole('button', { name: 'Add training target' })).toBeDisabled();
        expect(within(dialog).queryByRole('textbox', { name: /^Name/ })).not.toBeInTheDocument();
        expect(dialog).not.toHaveTextContent(/instance/i);
        const deployLink = within(dialog).getByRole('link', { name: 'Deploy AWS Batch stack' });
        expect(deployLink).toHaveAttribute('target', '_blank');
        expect(deployLink).toHaveAttribute('rel', 'noopener noreferrer');
        expect(deployLink.getAttribute('href')).toContain('aws-batch-trainer.yaml');
        expect(deployLink.querySelector('svg')).toHaveAttribute('aria-hidden', 'true');
        expect(within(dialog).getAllByRole('separator')).toHaveLength(2);
        await user.type(
            await within(dialog).findByRole('textbox', { name: /^S3 configuration file URI/ }),
            's3://config-bucket/studio-config.json'
        );
        expect(within(dialog).getAllByRole('textbox')).toHaveLength(1);
        expect(within(dialog).queryByRole('textbox', { name: /^Region/ })).not.toBeInTheDocument();
        expect(within(dialog).queryByRole('textbox', { name: /^Studio role ARN/ })).not.toBeInTheDocument();
        await user.click(within(dialog).getByRole('button', { name: 'Add training target' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(resolution).toEqual({
            configuration_uri: 's3://config-bucket/studio-config.json',
        });
        expect(await screen.findByRole('button', { name: 'Edit AWS Provider' })).toBeInTheDocument();
        expect(trainers[0].name).toBe('AWS Provider');
        expect(trainers[0].connection).toEqual(awsTrainer.connection);
    });

    it('prefills AWS edits and refreshes resources when saving', async () => {
        const user = userEvent.setup();
        let updated: unknown;
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([awsTrainer])),
            http.post('/api/remote-trainers/providers/aws/configuration', () =>
                HttpResponse.json({
                    ...(awsTrainer.connection as Extract<
                        SchemaRemoteTrainer['connection'],
                        { connection_mode: 'aws_batch' }
                    >),
                    bucket: 'updated-bucket',
                })
            ),
            http.patch(REMOTE_TRAINER_PATH, async ({ request }) => {
                updated = await request.json();
                return HttpResponse.json(awsTrainer);
            })
        );
        render(<TrainingTargetsPage />);
        await user.click(await screen.findByRole('button', { name: 'Edit AWS Provider' }));
        const dialog = await screen.findByRole('dialog', { name: 'Edit training target' });
        expect(await within(dialog).findByDisplayValue('s3://config-bucket/studio-config.json')).toBeInTheDocument();
        expect(within(dialog).getAllByRole('textbox')).toHaveLength(1);
        await user.click(within(dialog).getByRole('button', { name: 'Save changes' }));
        await waitFor(() => expect(updated).toMatchObject({ connection: { bucket: 'updated-bucket' } }));
    });

    it('shows the backend constraint when adding another AWS provider', async () => {
        const user = userEvent.setup();
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([awsTrainer])),
            http.post('/api/remote-trainers/providers/aws/configuration', () =>
                HttpResponse.json(
                    awsTrainer.connection as Extract<
                        SchemaRemoteTrainer['connection'],
                        { connection_mode: 'aws_batch' }
                    >
                )
            ),
            http.post(
                REMOTE_TRAINERS_PATH,
                () =>
                    new Response(JSON.stringify({ message: 'Only one AWS Provider training target is allowed.' }), {
                        status: 409,
                        headers: { 'Content-Type': 'application/json' },
                    })
            )
        );
        render(<TrainingTargetsPage />);
        await user.click(await screen.findByRole('button', { name: 'Add training target: AWS Provider' }));
        const dialog = await screen.findByRole('dialog', { name: 'Add training target' });
        await user.type(
            await within(dialog).findByRole('textbox', { name: /^S3 configuration file URI/ }),
            's3://config-bucket/studio-config.json'
        );
        await user.click(within(dialog).getByRole('button', { name: 'Add training target' }));
        expect(await within(dialog).findByRole('alert')).toHaveTextContent(
            'Only one AWS Provider training target is allowed.'
        );
    });

    it('keeps the AWS form open when configuration resolution fails', async () => {
        const user = userEvent.setup();
        server.use(
            http.get(REMOTE_TRAINERS_PATH, () => HttpResponse.json([awsTrainer])),
            http.post('/api/remote-trainers/providers/aws/configuration', () =>
                HttpResponse.json({ detail: 'Cannot read the S3 configuration.' }, { status: 400 })
            )
        );
        render(<TrainingTargetsPage />);
        await user.click(await screen.findByRole('button', { name: 'Edit AWS Provider' }));
        const dialog = await screen.findByRole('dialog', { name: 'Edit training target' });
        await within(dialog).findByDisplayValue('s3://config-bucket/studio-config.json');
        await user.click(within(dialog).getByRole('button', { name: 'Save changes' }));
        expect(await within(dialog).findByRole('alert')).toHaveTextContent('Cannot read the S3 configuration.');
        expect(dialog).toBeInTheDocument();
    });
});

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { getMockedRemoteTrainer } from '../../../../test-utils/mocks/mock-remote-trainer';
import { render } from '../../../../test-utils/render';
import { RemoteTrainerForm } from './remote-trainer-form';

const REMOTE_TRAINERS_PATH = '/api/remote-trainers';

const renderForm = (props: Partial<Parameters<typeof RemoteTrainerForm>[0]> = {}) =>
    render(<RemoteTrainerForm close={() => undefined} requestHostKeyConfirmation={() => undefined} {...props} />);

describe('RemoteTrainerForm', () => {
    it('shows the SSH tunnel connection fields', async () => {
        const user = userEvent.setup();

        renderForm();
        await user.click(screen.getByRole('tab', { name: /ssh tunnel/i }));
        await user.click(screen.getByRole('tab', { name: /connection details/i }));

        expect(screen.getByText('Connect through SSH when the trainer is not directly reachable.')).toBeInTheDocument();
        expect(screen.getByRole('tab', { name: /connection details/i })).toHaveAttribute('aria-selected', 'true');
        expect(screen.getByRole('textbox', { name: /^User/ })).toHaveValue('ec2-user');
        expect(screen.getByRole('textbox', { name: /local port/i })).toHaveValue('8001');
        expect(screen.queryByRole('textbox', { name: /trainer url/i })).not.toBeInTheDocument();
    });

    it('creates a remote trainer with an SSH tunnel configured via a new manual host', async () => {
        const user = userEvent.setup();
        const fingerprint = 'SHA256:first-seen-host-key';
        let created: Record<string, unknown> | undefined;
        let requestedConfirmation: { fingerprint: string; onConfirm: () => void } | undefined;

        server.use(
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
                created = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(getMockedRemoteTrainer({ id: 'trainer-1' }), { status: 201 });
            })
        );

        renderForm({
            requestHostKeyConfirmation: (confirmation) => {
                requestedConfirmation = confirmation;
            },
        });

        await user.type(screen.getByRole('textbox', { name: /^Name/ }), 'managed-trainer');
        await user.click(screen.getByRole('tab', { name: /ssh tunnel/i }));
        await user.click(screen.getByRole('tab', { name: /connection details/i }));
        await user.type(screen.getByRole('textbox', { name: /^Host/ }), 'gpu.example.test');
        await user.clear(screen.getByRole('textbox', { name: /^Port/ }));
        await user.type(screen.getByRole('textbox', { name: /^Port/ }), '2222');
        await user.clear(screen.getByRole('textbox', { name: /^User/ }));
        await user.type(screen.getByRole('textbox', { name: /^User/ }), 'trainer');
        await user.type(screen.getByRole('textbox', { name: /key path/i }), '~/.ssh/trainer');

        expect(screen.getByRole('button', { name: 'Add trainer' })).toBeEnabled();
        await user.click(screen.getByRole('button', { name: 'Add trainer' }));

        await waitFor(() => expect(requestedConfirmation).toBeDefined());
        requestedConfirmation?.onConfirm();

        await waitFor(() => expect(created).toBeDefined());
        expect(created).toMatchObject({
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
        const manualTrainer = getMockedRemoteTrainer({
            connection_mode: 'ssh',
            url: undefined,
            ssh_connection: {
                hostname: 'gpu.example.test',
                port: 2222,
                user: null,
                identity_file: '~/.ssh/trainer',
            },
            ssh_remote_port: 8001,
            ssh_local_port: 8001,
        });

        renderForm({ remoteTrainer: manualTrainer });

        expect(await screen.findByRole('tab', { name: /connection details/i })).toHaveAttribute(
            'aria-selected',
            'true'
        );
        expect(await screen.findByRole('textbox', { name: /^Host/ })).toHaveValue('gpu.example.test');
        expect(screen.getByRole('textbox', { name: /^Port/ })).toHaveValue('2,222');
        expect(screen.getByRole('textbox', { name: /^User/ })).toHaveValue('');
        expect(screen.getByRole('textbox', { name: /key path/i })).toHaveValue('~/.ssh/trainer');
    });

    it('does not offer SSH actions for an existing SSH trainer when the feature is unavailable', async () => {
        renderForm({
            sshAvailable: false,
            remoteTrainer: getMockedRemoteTrainer({
                connection_mode: 'ssh',
                ssh_host_alias: 'gpu-box',
                ssh_remote_port: 8001,
                ssh_local_port: 8001,
            }),
        });

        expect(screen.getByRole('tab', { name: 'SSH tunnel' })).toHaveAttribute('aria-disabled', 'true');
        expect(screen.getByText('SSH is unavailable in this environment.')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /add ssh connection/i })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Save changes' })).toBeDisabled();
    });

    it('creates a remote trainer with an SSH tunnel picked from the SSH config', async () => {
        const user = userEvent.setup();
        let created: Record<string, unknown> | undefined;

        server.use(
            http.get('/api/remote-servers/aliases', () =>
                HttpResponse.json([{ alias: 'gpu-box', hostname: '10.0.0.5', port: 22, user: 'trainer' }])
            ),
            http.post(REMOTE_TRAINERS_PATH, async ({ request }) => {
                created = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json(getMockedRemoteTrainer({ id: 'trainer-1' }), { status: 201 });
            })
        );

        renderForm();

        await user.type(screen.getByRole('textbox', { name: /^Name/ }), 'managed-trainer');
        await user.click(screen.getByRole('tab', { name: /ssh tunnel/i }));
        await user.click(screen.getByRole('tab', { name: /config alias/i }));

        await user.click(await screen.findByRole('button', { name: /select…?/i }));
        await user.click(await screen.findByRole('option', { name: /gpu-box/i }));
        await user.tab();

        await user.click(screen.getByRole('button', { name: 'Add trainer' }));

        await waitFor(() => expect(created).toBeDefined());
        expect(created).toMatchObject({
            connection_mode: 'ssh',
            url: null,
            ssh_host_alias: 'gpu-box',
            ssh_connection: null,
            ssh_remote_port: 8001,
            ssh_local_port: 8001,
        });
    });

    it('shows an insecure-URL warning for a non-loopback http:// trainer URL', async () => {
        const user = userEvent.setup();

        renderForm();
        await user.type(screen.getByRole('textbox', { name: /^Name/ }), 'managed-trainer');
        await user.click(screen.getByRole('tab', { name: 'Trainer URL' }));
        await user.type(screen.getByRole('textbox', { name: /trainer url/i }), 'http://trainer.example.test/api');

        expect(
            await screen.findByText(/sends your Hugging Face token to this trainer unencrypted/i)
        ).toBeInTheDocument();
    });
});

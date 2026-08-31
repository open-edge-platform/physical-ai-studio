import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaTrainJob } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { SshSettingsForm } from './ssh-settings-form';

const baseSsh = {
    enabled: false,
    connect_timeout_s: 10,
    command_timeout_s: 15,
    preflight_timeout_s: 30,
    image_pull_timeout_s: 1800,
    readiness_timeout_s: 120,
    gpu_wait_giveup_s: 1800,
    min_free_disk_bytes: 53687091200,
};

const runningJob: SchemaTrainJob = {
    id: 'job-1',
    project_id: 'project-1',
    progress: 42,
    status: 'running',
    message: 'Training',
    start_time: '2026-07-14T10:00:00Z',
    end_time: null,
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

describe('SshSettingsForm', () => {
    beforeEach(() => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));
    });

    it('renders the master switch off by default', () => {
        render(<SshSettingsForm ssh={baseSsh} />);

        expect(screen.getByRole('switch', { name: /enable remote training over ssh/i })).not.toBeChecked();
    });

    it('renders existing values in the timeout fields', () => {
        render(<SshSettingsForm ssh={{ ...baseSsh, connect_timeout_s: 42 }} />);

        expect(screen.getByRole('textbox', { name: /connect timeout/i })).toHaveValue('42');
    });

    it('disables the timeout fields while the master switch is off', () => {
        render(<SshSettingsForm ssh={baseSsh} />);

        expect(screen.getByRole('textbox', { name: /connect timeout/i })).toBeDisabled();
    });

    it('warns about interrupting running jobs when confirming a restart', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([runningJob])));

        const user = userEvent.setup();
        render(<SshSettingsForm ssh={baseSsh} />);

        await user.click(screen.getByRole('switch', { name: /enable remote training over ssh/i }));
        await user.click(screen.getByRole('button', { name: /save/i }));

        expect(await screen.findByText(/1 job is currently running/i)).toBeInTheDocument();
        expect(screen.getByText(/restarting the backend will interrupt it/i)).toBeInTheDocument();
    });

    it('saves the toggled master switch after confirming the restart', async () => {
        let patchedBody: unknown;
        server.use(
            http.patch('/api/settings', async ({ request }) => {
                patchedBody = await request.json();
                return HttpResponse.json({
                    trainer: {
                        request_timeout_s: 30,
                        download_read_timeout_s: 120,
                        stream_reconnect_max_s: 900,
                        stream_reconnect_backoff_max_s: 30,
                    },
                    huggingface: { hf_token: null },
                    ssh: { ...baseSsh, enabled: true },
                });
            })
        );

        const user = userEvent.setup();
        render(<SshSettingsForm ssh={baseSsh} />);

        await user.click(screen.getByRole('switch', { name: /enable remote training over ssh/i }));
        await user.click(screen.getByRole('button', { name: /save/i }));
        await user.click(await screen.findByRole('button', { name: /save and restart/i }));

        await waitFor(() => {
            expect(patchedBody).toMatchObject({ ssh: { enabled: true } });
        });
        expect(await screen.findByText('Saved')).toBeInTheDocument();
    });

    it('saves a changed timeout without touching the master switch', async () => {
        let patchedBody: unknown;
        server.use(
            http.patch('/api/settings', async ({ request }) => {
                patchedBody = await request.json();
                return HttpResponse.json({
                    trainer: {
                        request_timeout_s: 30,
                        download_read_timeout_s: 120,
                        stream_reconnect_max_s: 900,
                        stream_reconnect_backoff_max_s: 30,
                    },
                    huggingface: { hf_token: null },
                    ssh: { ...baseSsh, enabled: true, connect_timeout_s: 42 },
                });
            })
        );

        const user = userEvent.setup();
        render(<SshSettingsForm ssh={{ ...baseSsh, enabled: true }} />);

        const connectTimeoutField = screen.getByRole('textbox', { name: /connect timeout/i });
        await user.clear(connectTimeoutField);
        await user.type(connectTimeoutField, '42');
        await user.click(screen.getByRole('button', { name: /save/i }));
        await user.click(await screen.findByRole('button', { name: /save and restart/i }));

        await waitFor(() => {
            expect(patchedBody).toMatchObject({ ssh: { connect_timeout_s: 42, enabled: true } });
        });
        expect(await screen.findByText('Saved')).toBeInTheDocument();
    });
});

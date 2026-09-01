import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaModel, SchemaRemoteServer } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
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

const remoteServerId = 'c5c3a1a2-2f0d-4c6b-9c7c-9a1a2b3c4d5e';

const healthyRemoteServer: SchemaRemoteServer = {
    id: remoteServerId,
    name: 'lab-gpu-box',
    ssh_host_alias: 'gpu-box',
    device_type: 'cuda',
    last_check_status: 'healthy',
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

const mockProjectWithRemoteTrainer = (options: { remoteServers?: (typeof healthyRemoteServer)[] } = {}) => {
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
        http.get('/api/settings', () =>
            HttpResponse.json({
                trainer: {
                    request_timeout_s: 30,
                    download_read_timeout_s: 120,
                    stream_reconnect_max_s: 900,
                    stream_reconnect_backoff_max_s: 30,
                },
                huggingface: { hf_token: null },
                ssh: {
                    enabled: false,
                    connect_timeout_s: 10,
                    command_timeout_s: 15,
                    preflight_timeout_s: 30,
                    image_pull_timeout_s: 1800,
                    readiness_timeout_s: 120,
                    gpu_wait_giveup_s: 1800,
                    min_free_disk_bytes: 53687091200,
                },
            })
        ),
        http.get('/api/policies/{policy}/huggingface-access', ({ params }) => {
            const policy = params.policy;
            return HttpResponse.json({
                requirements:
                    policy === 'act'
                        ? []
                        : [
                              {
                                  repository: 'google/paligemma-3b-pt-224',
                                  status: 'missing_token',
                                  required: policy === 'pi05',
                                  access_url: 'https://huggingface.co/google/paligemma-3b-pt-224',
                              },
                          ],
            });
        }),
        http.get('/api/remote-servers', () => HttpResponse.json(options.remoteServers ?? [])),
        http.get('/api/remote-servers/{remote_server_id}/status', ({ params }) =>
            HttpResponse.json({
                remote_server_id: params.remote_server_id,
                status: 'healthy',
                device_type: 'cuda',
                waiting_for_gpu: false,
                checks: [],
            })
        )
    );
};

const renderDialog = (props: { baseModel?: SchemaModel } = {}) =>
    render(<TrainModelDialog {...props} close={() => undefined} />, {
        route: `/projects/${projectId}/models`,
        path: '/projects/:project_id/models',
    });

describe('TrainModelDialog', () => {
    it('does not submit a remote job when the final health check fails', async () => {
        const user = userEvent.setup();
        let healthCheckCount = 0;
        let jobSubmitted = false;

        mockProjectWithRemoteTrainer();
        server.use(
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

        renderDialog();

        // A new model starts with no dataset selected, and Train is a no-op without one.
        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));
        await user.click(await screen.findByRole('option', { name: remoteTrainer.name }));
        await screen.findByText('Remote trainer selected');
        await user.click(screen.getByRole('button', { name: 'Train' }));

        await waitFor(() => expect(healthCheckCount).toBeGreaterThan(1));
        expect(jobSubmitted).toBe(false);
    });

    it('offers remote trainers when training a new model', async () => {
        const user = userEvent.setup();
        mockProjectWithRemoteTrainer();

        renderDialog();

        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));

        expect(await screen.findByRole('option', { name: remoteTrainer.name })).toBeInTheDocument();
    });

    it('offers local training only when continuing an existing model', async () => {
        // The trainer protocol can receive a dataset but not a base checkpoint, so
        // the backend rejects a remote resume; don't offer what can't be submitted.
        const user = userEvent.setup();
        mockProjectWithRemoteTrainer();

        renderDialog({ baseModel });

        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));

        expect(await screen.findByRole('option', { name: /this machine \(local\)/i })).toBeInTheDocument();
        expect(screen.queryByRole('option', { name: remoteTrainer.name })).not.toBeInTheDocument();
    });

    it('warns when SmolVLA is selected without a Hugging Face token', async () => {
        const user = userEvent.setup();
        mockProjectWithRemoteTrainer();

        renderDialog();

        await user.click(await screen.findByLabelText('Select SmolVLA policy'));

        expect(
            await screen.findByText(/This policy downloads pretrained assets from Hugging Face/i)
        ).toBeInTheDocument();
        expect(screen.queryByText(/gated base model/i)).not.toBeInTheDocument();
    });

    it('blocks Pi0.5 training without a Hugging Face token', async () => {
        const user = userEvent.setup();
        mockProjectWithRemoteTrainer();

        renderDialog();
        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(screen.getByLabelText('Select Pi0.5 policy'));

        expect(
            await screen.findByText(/This policy downloads pretrained assets from Hugging Face/i)
        ).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Train' })).toBeDisabled();
    });

    it('blocks Pi0.5 training when the token lacks gated-model access', async () => {
        const user = userEvent.setup();
        mockProjectWithRemoteTrainer();
        server.use(
            http.get('/api/policies/{policy}/huggingface-access', () =>
                HttpResponse.json({
                    requirements: [
                        {
                            repository: 'google/paligemma-3b-pt-224',
                            status: 'denied',
                            required: true,
                            access_url: 'https://huggingface.co/google/paligemma-3b-pt-224',
                        },
                    ],
                })
            )
        );

        renderDialog();
        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(screen.getByLabelText('Select Pi0.5 policy'));

        expect(await screen.findByText(/does not have access to this policy/i)).toBeInTheDocument();
    });

    it('blocks Pi0.5 training when the Hugging Face access check itself fails', async () => {
        // Regression test: a failed check (network error, backend 500, ...) must
        // fail closed for a policy with a required Hub dependency, rather than
        // silently letting training through only to fail deep into a remote run.
        const user = userEvent.setup();
        mockProjectWithRemoteTrainer();
        server.use(
            http.get('/api/policies/{policy}/huggingface-access', () =>
                HttpResponse.json({ detail: 'boom' }, { status: 500 })
            )
        );

        renderDialog();
        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(screen.getByLabelText('Select Pi0.5 policy'));

        expect(await screen.findByText(/couldn.t verify Hugging Face access/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Train' })).toBeDisabled();
    });

    it('lists a configured SSH server alongside local and direct-URL trainers in a single control', async () => {
        const user = userEvent.setup();
        mockProjectWithRemoteTrainer({ remoteServers: [healthyRemoteServer] });

        renderDialog();

        // Exactly one "Run on" control: no second remote-server dropdown.
        expect(await screen.findByRole('button', { name: /this machine \(local\)/i })).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: /this machine \(local\)/i }));

        expect(await screen.findByRole('option', { name: remoteTrainer.name })).toBeInTheDocument();
        expect(screen.getByRole('option', { name: healthyRemoteServer.name })).toBeInTheDocument();
        expect(screen.getAllByRole('listbox', { name: 'Run on' })).toHaveLength(1);
    });

    it('submits an SSH job with remote_server_id when a healthy remote server is selected', async () => {
        const user = userEvent.setup();
        let submittedPayload: Record<string, unknown> | null = null;

        mockProjectWithRemoteTrainer({ remoteServers: [healthyRemoteServer] });
        server.use(
            http.post('/api/jobs:train', async ({ request }) => {
                submittedPayload = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json({}, { status: 201 });
            })
        );

        renderDialog();

        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));
        await user.click(await screen.findByRole('option', { name: healthyRemoteServer.name }));
        await user.click(screen.getByRole('button', { name: 'Train' }));

        await waitFor(() => expect(submittedPayload).not.toBeNull());
        expect(submittedPayload).toMatchObject({
            training_target: 'ssh',
            remote_server_id: remoteServerId,
        });
        expect(submittedPayload).not.toHaveProperty('remote_trainer_id');
    });

    it('disables Train and shows a warning for a remote server that is not ready', async () => {
        const user = userEvent.setup();
        mockProjectWithRemoteTrainer({
            remoteServers: [{ ...healthyRemoteServer, last_check_status: 'unreachable' }],
        });

        renderDialog();

        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));
        await user.click(await screen.findByRole('option', { name: healthyRemoteServer.name }));

        expect(
            await screen.findByText(
                (_, element) =>
                    element?.children.length === 0 && (element?.textContent ?? '').includes("isn't ready for training")
            )
        ).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Train' })).toBeDisabled();
    });

    it('allows submitting a job against a never-verified SSH server, letting the backend verify it', async () => {
        // "unknown" last_check_status just means nobody has clicked "Pull & verify
        // image" yet - not a confirmed failure. The job endpoint runs the same
        // Tier-2 verification automatically, so the dialog must not force a trip
        // to the training targets page first.
        const user = userEvent.setup();
        let submittedPayload: Record<string, unknown> | null = null;

        mockProjectWithRemoteTrainer({
            remoteServers: [{ ...healthyRemoteServer, last_check_status: 'unknown' }],
        });
        server.use(
            http.post('/api/jobs:train', async ({ request }) => {
                submittedPayload = (await request.json()) as Record<string, unknown>;
                return HttpResponse.json({}, { status: 201 });
            })
        );

        renderDialog();

        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));
        await user.click(await screen.findByRole('option', { name: healthyRemoteServer.name }));

        expect(await screen.findByText(/hasn't been verified yet/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Train' })).not.toBeDisabled();

        await user.click(screen.getByRole('button', { name: 'Train' }));

        await waitFor(() => expect(submittedPayload).not.toBeNull());
        expect(submittedPayload).toMatchObject({
            training_target: 'ssh',
            remote_server_id: remoteServerId,
        });
    });

    it('shows the backend error when Tier-2 verification fails during submission', async () => {
        const user = userEvent.setup();

        mockProjectWithRemoteTrainer({
            remoteServers: [{ ...healthyRemoteServer, last_check_status: 'unknown' }],
        });
        server.use(
            http.post('/api/jobs:train', () =>
                HttpResponse.json(
                    {
                        error_code: 'remote_server_not_ready',
                        message: "Remote server 'lab-gpu-box' is not ready for training.",
                        http_status: 409,
                    } as never,
                    { status: 409 }
                )
            )
        );

        renderDialog();

        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));
        await user.click(await screen.findByRole('option', { name: healthyRemoteServer.name }));
        await user.click(screen.getByRole('button', { name: 'Train' }));

        expect(await screen.findByText(/is not ready for training/i)).toBeInTheDocument();
    });

    it('does not submit an SSH job when the live status is unhealthy despite a persisted healthy check', async () => {
        // A server can be verified (last_check_status === "healthy") yet go
        // unreachable before the next explicit verification — the live Tier-1
        // poll must still block submission.
        const user = userEvent.setup();
        let jobSubmitted = false;

        mockProjectWithRemoteTrainer({ remoteServers: [healthyRemoteServer] });
        server.use(
            http.get('/api/remote-servers/{remote_server_id}/status', ({ params }) =>
                HttpResponse.json({
                    remote_server_id: params.remote_server_id,
                    status: 'unreachable',
                    device_type: 'cuda',
                    waiting_for_gpu: false,
                    checks: [],
                })
            ),
            http.post('/api/jobs:train', () => {
                jobSubmitted = true;
                return HttpResponse.json({}, { status: 201 });
            })
        );

        renderDialog();

        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));
        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));
        await user.click(await screen.findByRole('option', { name: healthyRemoteServer.name }));

        expect(
            await screen.findByText(
                (_, element) =>
                    element?.children.length === 0 && (element?.textContent ?? '').includes("isn't ready for training")
            )
        ).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Train' })).toBeDisabled();
        expect(jobSubmitted).toBe(false);
    });

    it('shows a status indicator for each run target so its health is clear at a glance', async () => {
        const user = userEvent.setup();
        mockProjectWithRemoteTrainer({ remoteServers: [healthyRemoteServer] });
        server.use(
            http.get('/api/remote-trainers/{remote_trainer_id}/health', () => HttpResponse.json(healthyRemoteTrainer)),
            http.get('/api/remote-servers/{remote_server_id}/status', () =>
                HttpResponse.json({
                    remote_server_id: remoteServerId,
                    status: 'healthy',
                    device_type: 'cuda',
                    waiting_for_gpu: false,
                    checks: [],
                })
            )
        );

        renderDialog();

        await user.click(await screen.findByRole('button', { name: /this machine \(local\)/i }));

        // Local always has a status (its device state), remote targets reflect live health.
        const localOption = await screen.findByRole('option', { name: /this machine \(local\)/i });
        expect(within(localOption).getByText('CPU only')).toBeInTheDocument();

        const trainerOption = screen.getByRole('option', { name: new RegExp(remoteTrainer.name) });
        await waitFor(() => expect(within(trainerOption).getByText('Healthy')).toBeInTheDocument());

        const sshOption = screen.getByRole('option', { name: new RegExp(healthyRemoteServer.name) });
        await waitFor(() => expect(within(sshOption).getByText('Healthy')).toBeInTheDocument());
    });

    it('submits only one job when Train is double-clicked before the request resolves', async () => {
        // The dialog stays open for a moment after the request settles (the parent
        // closes it once `close()` runs), so a fast double-click could previously
        // fire `save()` twice before the button became disabled, submitting two
        // identical jobs.
        const user = userEvent.setup();
        let submitCount = 0;

        mockProjectWithRemoteTrainer();
        server.use(
            http.post('/api/jobs:train', async () => {
                submitCount += 1;
                // Simulate real network/server latency so both clicks land while the
                // first request is still in flight.
                await new Promise((resolve) => setTimeout(resolve, 50));
                return HttpResponse.json({}, { status: 201 });
            })
        );

        renderDialog();

        await user.click(await screen.findByRole('button', { name: /select…/i }));
        await user.click(await screen.findByRole('option', { name: 'Test dataset' }));

        const trainButton = screen.getByRole('button', { name: 'Train' });

        // Fire both clicks back-to-back, the way a real double-click would,
        // instead of awaiting user-event's full interaction between them.
        void user.click(trainButton);
        void user.click(trainButton);

        await waitFor(() => expect(trainButton).toBeDisabled());
        await new Promise((resolve) => setTimeout(resolve, 100));

        expect(submitCount).toBe(1);
    });
});

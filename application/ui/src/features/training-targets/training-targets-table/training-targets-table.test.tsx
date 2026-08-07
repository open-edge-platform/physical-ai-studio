import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { TrainingTargetRow } from './training-target-row';
import { TrainingTargetsTable } from './training-targets-table';

const REMOTE_TRAINER_HEALTH_PATH = '/api/remote-trainers/{remote_trainer_id}/health';
const REMOTE_SERVER_STATUS_PATH = '/api/remote-servers/{remote_server_id}/status';
const REMOTE_SERVER_CHECK_PATH = '/api/remote-servers/{remote_server_id}/check';

const remoteTrainer = {
    id: 'b8b28d4f-e78f-48ad-afb8-03d060178a3c',
    name: 'managed-trainer',
    url: 'https://trainer.example.test/api',
    created_at: '2026-07-14T12:00:00Z',
};

const secondRemoteTrainer = {
    ...remoteTrainer,
    id: 'b5a0da22-7066-426d-b0ae-6dfae2d983dc',
    name: 'unavailable-trainer',
};

const remoteServer = {
    id: 'f1a2b3c4-d5e6-47a8-99b0-1234567890ab',
    name: 'lambda-a100',
    ssh_host_alias: 'gpu-01',
    device_type: 'cuda' as const,
    last_check_status: 'unknown' as const,
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

const healthyServerStatus = {
    remote_server_id: remoteServer.id,
    status: 'healthy' as const,
    device_type: 'cuda',
    checks: [
        {
            key: 'alias_resolved' as const,
            tier: 1 as const,
            outcome: 'passed' as const,
            blocking: true,
            checked_at: '2026-08-07T12:00:00Z',
        },
        {
            key: 'gpu_free' as const,
            tier: 1 as const,
            outcome: 'passed' as const,
            blocking: false,
            checked_at: '2026-08-07T12:00:00Z',
        },
    ],
    checked_at: '2026-08-07T12:00:00Z',
    waiting_for_gpu: false,
};

const busyServerStatus = {
    ...healthyServerStatus,
    checks: [
        healthyServerStatus.checks[0],
        {
            key: 'gpu_free' as const,
            tier: 1 as const,
            outcome: 'warning' as const,
            blocking: false,
            checked_at: '2026-08-07T12:00:00Z',
            detail: 'GPU is occupied by another process',
        },
    ],
};

const misconfiguredServerStatus = {
    ...healthyServerStatus,
    status: 'unreachable' as const,
    checks: [
        {
            key: 'alias_resolved' as const,
            tier: 1 as const,
            outcome: 'failed' as const,
            blocking: true,
            checked_at: '2026-08-07T12:00:00Z',
            reason_code: 'alias_not_found',
            detail: "SSH host alias 'gpu-01' was not found in your SSH config.",
        },
    ],
    reason_code: 'alias_not_found',
};

const directUrlRow = (trainer: typeof remoteTrainer): TrainingTargetRow => ({ kind: 'direct-url', trainer });
const sshRow = (server_: typeof remoteServer): TrainingTargetRow => ({ kind: 'ssh', server: server_ });

describe('TrainingTargetsTable', () => {
    beforeEach(() => {
        server.use(
            http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json(healthyTrainer)),
            http.get(REMOTE_SERVER_STATUS_PATH, () => HttpResponse.json(healthyServerStatus))
        );
    });

    describe('direct-URL trainer rows', () => {
        it('lists every configured remote trainer with its status and compute badge', async () => {
            render(
                <TrainingTargetsTable
                    rows={[directUrlRow(remoteTrainer), directUrlRow(secondRemoteTrainer)]}
                    onEdit={vi.fn()}
                    onDelete={vi.fn()}
                />
            );

            expect(await screen.findByText(remoteTrainer.name)).toBeInTheDocument();
            expect(screen.getByText(secondRemoteTrainer.name)).toBeInTheDocument();
            expect(await screen.findAllByText('Healthy')).not.toHaveLength(0);
            expect(screen.getAllByText('CUDA')).not.toHaveLength(0);
        });

        it('shows connection details for the expanded remote trainer', async () => {
            render(<TrainingTargetsTable rows={[directUrlRow(remoteTrainer)]} onEdit={vi.fn()} onDelete={vi.fn()} />);

            expect(await screen.findAllByText('Healthy')).not.toHaveLength(0);
            expect(screen.getByText('Trainer health endpoint')).toBeInTheDocument();
            expect(screen.getByText('Compute capability')).toBeInTheDocument();
            expect(screen.getAllByText(/NVIDIA A100/)).not.toHaveLength(0);
            expect(screen.getByText('Storage capacity')).toBeInTheDocument();
            expect(screen.getAllByText(/558\.8 GB free of 931\.3 GB/)).not.toHaveLength(0);
            expect(screen.getAllByText(remoteTrainer.url)).not.toHaveLength(0);
        });

        it('distinguishes a failed health request from an unreachable trainer', async () => {
            server.use(http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json({ detail: [] }, { status: 422 })));

            render(<TrainingTargetsTable rows={[directUrlRow(remoteTrainer)]} onEdit={vi.fn()} onDelete={vi.fn()} />);

            expect(await screen.findAllByText('Check failed')).not.toHaveLength(0);
            expect(screen.getAllByText('Studio could not complete the health check. Try again.')).not.toHaveLength(0);
        });

        it('expands the first row by default and only one row at a time', async () => {
            const user = userEvent.setup();

            render(
                <TrainingTargetsTable
                    rows={[directUrlRow(remoteTrainer), directUrlRow(secondRemoteTrainer)]}
                    onEdit={vi.fn()}
                    onDelete={vi.fn()}
                />
            );

            const firstToggle = await screen.findByRole('button', { name: /show details for managed-trainer/i });
            const secondToggle = await screen.findByRole('button', { name: /show details for unavailable-trainer/i });

            expect(firstToggle).toHaveAttribute('aria-expanded', 'true');
            expect(secondToggle).toHaveAttribute('aria-expanded', 'false');

            await user.click(secondToggle);

            expect(firstToggle).toHaveAttribute('aria-expanded', 'false');
            expect(secondToggle).toHaveAttribute('aria-expanded', 'true');

            await user.click(secondToggle);

            expect(secondToggle).toHaveAttribute('aria-expanded', 'false');
        });

        it('calls onEdit with the selected row', async () => {
            const user = userEvent.setup();
            const onEdit = vi.fn();

            render(<TrainingTargetsTable rows={[directUrlRow(remoteTrainer)]} onEdit={onEdit} onDelete={vi.fn()} />);

            await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
            await user.click(await screen.findByRole('menuitem', { name: 'Edit' }));

            expect(onEdit).toHaveBeenCalledWith(directUrlRow(remoteTrainer));
        });

        it('calls onDelete with the selected row', async () => {
            const user = userEvent.setup();
            const onDelete = vi.fn();

            render(<TrainingTargetsTable rows={[directUrlRow(remoteTrainer)]} onEdit={vi.fn()} onDelete={onDelete} />);

            await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
            await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));

            expect(onDelete).toHaveBeenCalledWith(directUrlRow(remoteTrainer));
        });

        it('triggers a health re-check without expanding or collapsing the row', async () => {
            const user = userEvent.setup();

            render(<TrainingTargetsTable rows={[directUrlRow(remoteTrainer)]} onEdit={vi.fn()} onDelete={vi.fn()} />);

            const toggle = await screen.findByRole('button', { name: /show details for managed-trainer/i });
            expect(toggle).toHaveAttribute('aria-expanded', 'true');

            await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
            await user.click(await screen.findByRole('menuitem', { name: 'Check status' }));

            expect(toggle).toHaveAttribute('aria-expanded', 'true');
            expect(await screen.findAllByText('Healthy')).not.toHaveLength(0);
        });

        it('shows a failed health check for a row without swallowing other rows', async () => {
            server.use(
                http.get(REMOTE_TRAINER_HEALTH_PATH, ({ params }) =>
                    params.remote_trainer_id === secondRemoteTrainer.id
                        ? HttpResponse.json({ detail: [] }, { status: 503 })
                        : HttpResponse.json(healthyTrainer)
                )
            );

            render(
                <TrainingTargetsTable
                    rows={[directUrlRow(remoteTrainer), directUrlRow(secondRemoteTrainer)]}
                    onEdit={vi.fn()}
                    onDelete={vi.fn()}
                />
            );

            const unavailableRow = await screen.findByTestId(`training-target-row-${secondRemoteTrainer.id}`);
            expect(await within(unavailableRow).findByText('Check failed')).toBeInTheDocument();

            const healthyRow = await screen.findByTestId(`training-target-row-${remoteTrainer.id}`);
            expect(await within(healthyRow).findAllByText('Healthy')).not.toHaveLength(0);
        });
    });

    describe('SSH server rows', () => {
        it('shows an SSH server with a type badge distinct from direct-URL rows', async () => {
            render(
                <TrainingTargetsTable
                    rows={[directUrlRow(remoteTrainer), sshRow(remoteServer)]}
                    onEdit={vi.fn()}
                    onDelete={vi.fn()}
                />
            );

            expect(await screen.findByText(remoteServer.name)).toBeInTheDocument();
            expect(screen.getByText('SSH')).toBeInTheDocument();
            expect(screen.getByText('Direct URL')).toBeInTheDocument();
        });

        it('shows a Healthy status badge for a passing SSH target', async () => {
            render(<TrainingTargetsTable rows={[sshRow(remoteServer)]} onEdit={vi.fn()} onDelete={vi.fn()} />);

            expect(await screen.findAllByText('Healthy')).not.toHaveLength(0);
        });

        it('shows a Busy status badge, not a failure, when the GPU is occupied', async () => {
            server.use(http.get(REMOTE_SERVER_STATUS_PATH, () => HttpResponse.json(busyServerStatus)));

            render(<TrainingTargetsTable rows={[sshRow(remoteServer)]} onEdit={vi.fn()} onDelete={vi.fn()} />);

            expect(await screen.findAllByText('Busy')).not.toHaveLength(0);
            expect(screen.queryByText('Unreachable')).not.toBeInTheDocument();
        });

        it('shows a Misconfigured/Unreachable status badge with the actionable reason for a missing alias', async () => {
            server.use(http.get(REMOTE_SERVER_STATUS_PATH, () => HttpResponse.json(misconfiguredServerStatus)));

            render(<TrainingTargetsTable rows={[sshRow(remoteServer)]} onEdit={vi.fn()} onDelete={vi.fn()} />);

            expect(await screen.findAllByText('Unreachable')).not.toHaveLength(0);
            expect(
                await screen.findByText("SSH host alias 'gpu-01' was not found in your SSH config.")
            ).toBeInTheDocument();
        });

        it('never fires the Tier 2 check request merely from mounting or expanding a row', async () => {
            const checkSpy = vi.fn();
            server.use(http.post(REMOTE_SERVER_CHECK_PATH, checkSpy));

            const user = userEvent.setup();
            render(<TrainingTargetsTable rows={[sshRow(remoteServer)]} onEdit={vi.fn()} onDelete={vi.fn()} />);

            expect(await screen.findAllByText('Healthy')).not.toHaveLength(0);
            expect(checkSpy).not.toHaveBeenCalled();

            const toggle = await screen.findByRole('button', { name: /show details for lambda-a100/i });
            await user.click(toggle);
            await user.click(toggle);

            expect(checkSpy).not.toHaveBeenCalled();
        });

        it('runs the Tier 2 check only from the explicit Test connection action', async () => {
            let callCount = 0;
            server.use(
                http.post(REMOTE_SERVER_CHECK_PATH, () => {
                    callCount += 1;
                    return HttpResponse.json({
                        remote_server_id: remoteServer.id,
                        tiers_run: [2 as const],
                        checks: [],
                        checked_at: '2026-08-07T12:05:00Z',
                    });
                })
            );

            const user = userEvent.setup();
            render(<TrainingTargetsTable rows={[sshRow(remoteServer)]} onEdit={vi.fn()} onDelete={vi.fn()} />);

            await user.click(await screen.findByRole('button', { name: 'Test connection' }));

            expect(callCount).toBe(1);
        });

        it('calls onEdit and onDelete with the selected SSH row', async () => {
            const user = userEvent.setup();
            const onEdit = vi.fn();
            const onDelete = vi.fn();

            render(<TrainingTargetsTable rows={[sshRow(remoteServer)]} onEdit={onEdit} onDelete={onDelete} />);

            await user.click(await screen.findByRole('button', { name: `More actions ${remoteServer.name}` }));
            await user.click(await screen.findByRole('menuitem', { name: 'Edit' }));
            expect(onEdit).toHaveBeenCalledWith(sshRow(remoteServer));

            await user.click(await screen.findByRole('button', { name: `More actions ${remoteServer.name}` }));
            await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));
            expect(onDelete).toHaveBeenCalledWith(sshRow(remoteServer));
        });
    });
});

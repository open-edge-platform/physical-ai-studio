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

const remoteTrainer = {
    id: 'b8b28d4f-e78f-48ad-afb8-03d060178a3c',
    name: 'managed-trainer',
    connection_mode: 'direct' as const,
    url: 'https://trainer.example.test/api',
    ssh_remote_port: null,
    ssh_local_port: null,
    created_at: '2026-07-14T12:00:00Z',
};

const secondRemoteTrainer = {
    ...remoteTrainer,
    id: 'b5a0da22-7066-426d-b0ae-6dfae2d983dc',
    name: 'unavailable-trainer',
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

const directUrlRow = (trainer: typeof remoteTrainer): TrainingTargetRow => ({ kind: 'direct-url', trainer });

describe('TrainingTargetsTable', () => {
    beforeEach(() => {
        server.use(http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json(healthyTrainer)));
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
});

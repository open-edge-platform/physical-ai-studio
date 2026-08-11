import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { RemoteTrainersTable } from './remote-trainers-table';

const REMOTE_TRAINER_HEALTH_PATH = '/api/remote-trainers/{remote_trainer_id}/health';

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

const healthyTrainer = {
    remote_trainer_id: remoteTrainer.id,
    status: 'healthy' as const,
    checked_at: '2026-07-16T12:00:00Z',
    latency_ms: 24,
    devices: [{ type: 'cuda' as const, name: 'NVIDIA A100', memory: 85899345920, index: 0 }],
    storage: { total_bytes: 1_000_000_000_000, free_bytes: 600_000_000_000 },
    reason_code: null,
};

describe('RemoteTrainersTable', () => {
    beforeEach(() => {
        server.use(http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json(healthyTrainer)));
    });

    it('lists every configured remote trainer with its status and compute badge', async () => {
        render(
            <RemoteTrainersTable
                remoteTrainers={[remoteTrainer, secondRemoteTrainer]}
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
        render(<RemoteTrainersTable remoteTrainers={[remoteTrainer]} onEdit={vi.fn()} onDelete={vi.fn()} />);

        expect(await screen.findAllByText('Healthy')).not.toHaveLength(0);
        expect(screen.getByText('Trainer health endpoint')).toBeInTheDocument();
        expect(screen.getByText('Compute capability')).toBeInTheDocument();
        expect(screen.getAllByText(/NVIDIA A100/)).not.toHaveLength(0);
        expect(screen.getByText('Storage capacity')).toBeInTheDocument();
        expect(screen.getAllByText(/558\.8 GB free of 931\.3 GB/)).not.toHaveLength(0);
        expect(screen.getAllByText(remoteTrainer.url)).not.toHaveLength(0);
    });

    it('attributes an invalid device report to compute capability', async () => {
        server.use(
            http.get(REMOTE_TRAINER_HEALTH_PATH, () =>
                HttpResponse.json({
                    ...healthyTrainer,
                    status: 'degraded',
                    devices: [],
                    reason_code: 'invalid_devices_response',
                })
            )
        );

        render(<RemoteTrainersTable remoteTrainers={[remoteTrainer]} onEdit={vi.fn()} onDelete={vi.fn()} />);

        expect(await screen.findAllByText('Healthy')).not.toHaveLength(0);
        const trainerHealthRow = (await screen.findByText('Trainer health endpoint')).closest('div');
        const computeRow = screen.getByText('Compute capability').closest('div');

        if (trainerHealthRow === null || computeRow === null) {
            throw new Error('Expected trainer health and compute capability rows to be rendered.');
        }

        expect(within(trainerHealthRow).getByText('Healthy')).toBeInTheDocument();
        expect(within(computeRow).getByText('Unknown')).toBeInTheDocument();
    });

    it('distinguishes a failed health request from an unreachable trainer', async () => {
        server.use(http.get(REMOTE_TRAINER_HEALTH_PATH, () => HttpResponse.json({ detail: [] }, { status: 422 })));

        render(<RemoteTrainersTable remoteTrainers={[remoteTrainer]} onEdit={vi.fn()} onDelete={vi.fn()} />);

        expect(await screen.findAllByText('Check failed')).not.toHaveLength(0);
        expect(screen.getAllByText('Studio could not complete the health check. Try again.')).not.toHaveLength(0);
    });

    it('expands the first row by default and only one row at a time', async () => {
        const user = userEvent.setup();

        render(
            <RemoteTrainersTable
                remoteTrainers={[remoteTrainer, secondRemoteTrainer]}
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

    it('calls onEdit with the selected remote trainer', async () => {
        const user = userEvent.setup();
        const onEdit = vi.fn();

        render(<RemoteTrainersTable remoteTrainers={[remoteTrainer]} onEdit={onEdit} onDelete={vi.fn()} />);

        await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
        await user.click(await screen.findByRole('menuitem', { name: 'Edit' }));

        expect(onEdit).toHaveBeenCalledWith(remoteTrainer);
    });

    it('calls onDelete with the selected remote trainer', async () => {
        const user = userEvent.setup();
        const onDelete = vi.fn();

        render(<RemoteTrainersTable remoteTrainers={[remoteTrainer]} onEdit={vi.fn()} onDelete={onDelete} />);

        await user.click(await screen.findByRole('button', { name: `More actions ${remoteTrainer.name}` }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));

        expect(onDelete).toHaveBeenCalledWith(remoteTrainer);
    });

    it('triggers a health re-check without expanding or collapsing the row', async () => {
        const user = userEvent.setup();

        render(<RemoteTrainersTable remoteTrainers={[remoteTrainer]} onEdit={vi.fn()} onDelete={vi.fn()} />);

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
            <RemoteTrainersTable
                remoteTrainers={[remoteTrainer, secondRemoteTrainer]}
                onEdit={vi.fn()}
                onDelete={vi.fn()}
            />
        );

        const unavailableRow = await screen.findByTestId(`remote-trainer-row-${secondRemoteTrainer.id}`);
        expect(await within(unavailableRow).findByText('Check failed')).toBeInTheDocument();

        const healthyRow = await screen.findByTestId(`remote-trainer-row-${remoteTrainer.id}`);
        expect(await within(healthyRow).findAllByText('Healthy')).not.toHaveLength(0);
    });
});

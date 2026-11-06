// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, rs } from '@rstest/core';
import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { $api } from '../../api/client';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { createQueryClient } from '../../query-client/query-client';
import { render } from '../../test-utils/render';
import { RobotSimulationLaunch } from './robot-simulation';
import { SimulationDialog } from './simulation-dialog';
import { SimulationNotRunning } from './simulation-not-running';
import { Simulation } from './use-simulation';

const PROJECT_ID = 'project-1';
const robot = { id: 'robot-1', name: 'Sim arm', type: 'SO101_Follower' };

const launch: RobotSimulationLaunch = {
    scenes: [
        { id: 'single_pick_place', display_name: 'Single Pick & Place', description: 'One block and a target disc' },
        { id: 'yahtzee', display_name: 'Yahtzee', description: 'Pick up 6 dice' },
    ],
    default_scene: 'single_pick_place',
    max_seed: 4294967295,
    labels: ['1 arm', 'experimental'],
};

const idle: Simulation = { status: 'idle', cameras: [] };

const mockSimulation = (simulation: Simulation) => {
    server.use(
        http.get('/api/simulation', () => HttpResponse.json(simulation)),
        http.get('/api/robots/catalog', () =>
            HttpResponse.json([
                {
                    type: 'SO101_Follower',
                    display_name: 'SO101 Follower (experimental)',
                    role: 'follower',
                    urdf_path: '',
                    package_map: {},
                    joint_map: {},
                    zero_calibration: null,
                    simulation: launch,
                },
            ])
        )
    );
};

const renderDialog = (close = rs.fn(), onRunning = rs.fn()) => {
    render(
        <SimulationDialog robot={robot} project_id={PROJECT_ID} launch={launch} close={close} onRunning={onRunning} />,
        { route: `/projects/${PROJECT_ID}/robots`, path: '/projects/:project_id/robots' }
    );

    return { close, onRunning };
};

describe('SimulationDialog', () => {
    afterEach(() => {
        window.localStorage.clear();
        rs.restoreAllMocks();
    });

    it('shows the robot labels and starts the chosen scene with a random seed', async () => {
        mockSimulation(idle);
        const started: unknown[] = [];
        server.use(
            http.post('/api/simulation', async ({ request }) => {
                started.push(await request.json());
                return HttpResponse.json(
                    { ...idle, status: 'starting', robot_id: robot.id, scene: 'single_pick_place' },
                    { status: 202 }
                );
            })
        );
        const user = userEvent.setup();
        renderDialog();

        expect(await screen.findByText('Sim arm')).toBeInTheDocument();
        expect(screen.getByText('SO101 Follower (experimental)')).toBeInTheDocument();
        expect(screen.queryByText(/1 arm/)).not.toBeInTheDocument();
        expect(screen.getByText('Seed')).not.toBeVisible();
        await user.click(screen.getByRole('button', { name: 'Start simulation' }));

        await waitFor(() =>
            expect(started).toEqual([
                { project_id: PROJECT_ID, robot_id: robot.id, scene: 'single_pick_place', seed: null, replace: false },
            ])
        );
        expect(await screen.findByRole('heading', { name: 'Starting simulation' })).toBeInTheDocument();
    });

    it('offers to replace the simulation another robot runs', async () => {
        mockSimulation({ ...idle, status: 'running', robot_id: 'other', robot_type: 'MuJoCo_Koch', scene: 'yahtzee' });
        const started: { replace?: boolean }[] = [];
        server.use(
            http.post('/api/simulation', async ({ request }) => {
                started.push((await request.json()) as { replace?: boolean });
                return HttpResponse.json({ ...idle, status: 'starting', robot_id: robot.id }, { status: 202 });
            })
        );
        const user = userEvent.setup();
        renderDialog();

        expect(await screen.findByText('Another simulation is running')).toBeInTheDocument();
        expect(screen.getByText(/MuJoCo Koch · yahtzee is running/)).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Stop it and start this one' }));

        await waitFor(() => expect(started[0]?.replace).toBe(true));
    });

    it('shows why a start was refused', async () => {
        mockSimulation(idle);
        server.use(
            http.post('/api/simulation', () =>
                HttpResponse.json(
                    {
                        error_code: 'simulation_name_in_use',
                        message: "A simulation named 'sim' is already running outside Studio.",
                        http_status: 409,
                    } as never,
                    { status: 409 }
                )
            )
        );
        const user = userEvent.setup();
        renderDialog();

        await user.click(await screen.findByRole('button', { name: 'Start simulation' }));

        expect(await screen.findByText(/already running outside Studio/)).toBeInTheDocument();
    });

    it('keeps the seed under Advanced', async () => {
        mockSimulation(idle);
        const user = userEvent.setup();
        renderDialog();

        await user.click(await screen.findByRole('button', { name: 'Advanced' }));

        expect(await screen.findByRole('radio', { name: 'Fixed' })).toBeInTheDocument();
    });

    it('shows the download progress of a starting simulation', async () => {
        mockSimulation({
            ...idle,
            status: 'starting',
            robot_id: robot.id,
            robot_type: 'SO101_Follower',
            scene: 'yahtzee',
            phase: 'fetch',
            download: { bytes: 2_500_000, total: 10_000_000 },
            owner_name: 'sim-a',
        });
        renderDialog();

        expect(await screen.findByRole('heading', { name: 'Starting simulation' })).toBeInTheDocument();
        expect(screen.getByText('downloading 2.5 MB of 10.0 MB')).toBeInTheDocument();
        expect(screen.getByText('as sim-a')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    });

    it('shows the error of a failed start and goes back to the form', async () => {
        mockSimulation({
            ...idle,
            status: 'failed',
            robot_id: robot.id,
            phase: null,
            error: { message: 'Menagerie download failed', phase: 'fetch', exit_code: 1 },
        });
        server.use(http.delete('/api/simulation', () => HttpResponse.json(idle)));
        const user = userEvent.setup();
        renderDialog();

        expect(await screen.findByText('Menagerie download failed')).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Back' }));

        expect(await screen.findByRole('heading', { name: 'Start simulation' })).toBeInTheDocument();
    });

    it('replaces its own simulation when starting again after a failed start', async () => {
        mockSimulation({
            ...idle,
            status: 'failed',
            robot_id: robot.id,
            error: { message: 'boom', phase: 'load', exit_code: null },
        });
        const started: { replace?: boolean }[] = [];
        server.use(
            // The failed child is still exiting: Back leaves it stopping.
            http.delete('/api/simulation', () =>
                HttpResponse.json({ ...idle, status: 'stopping', robot_id: robot.id })
            ),
            http.post('/api/simulation', async ({ request }) => {
                started.push((await request.json()) as { replace?: boolean });
                return HttpResponse.json({ ...idle, status: 'starting', robot_id: robot.id }, { status: 202 });
            })
        );
        const user = userEvent.setup();
        renderDialog();

        await user.click(await screen.findByRole('button', { name: 'Back' }));
        await user.click(await screen.findByRole('button', { name: 'Start simulation' }));

        await waitFor(() => expect(started[0]?.replace).toBe(true));
        expect(await screen.findByRole('heading', { name: 'Starting simulation' })).toBeInTheDocument();
    });

    it('reports when the simulation is running', async () => {
        mockSimulation({ ...idle, status: 'running', robot_id: robot.id, cameras: ['wrist'] });
        const { onRunning } = renderDialog();

        expect(await screen.findByRole('heading', { name: 'Simulation running' })).toBeInTheDocument();
        // It was already running when the dialog opened, so nothing changed.
        expect(onRunning).not.toHaveBeenCalled();
    });
});

describe('SimulationDialog running transition', () => {
    it('reports once when the simulation it follows starts running', async () => {
        const starting: Simulation = { ...idle, status: 'starting', robot_id: robot.id, phase: 'load' };
        mockSimulation(starting);
        const queryClient = createQueryClient();
        const onRunning = rs.fn();
        render(
            <SimulationDialog
                robot={robot}
                project_id={PROJECT_ID}
                launch={launch}
                close={rs.fn()}
                onRunning={onRunning}
            />,
            { route: `/projects/${PROJECT_ID}/robots`, path: '/projects/:project_id/robots', queryClient }
        );
        expect(await screen.findByRole('heading', { name: 'Starting simulation' })).toBeInTheDocument();

        act(() => {
            queryClient.setQueryData($api.queryOptions('get', '/api/simulation').queryKey, {
                ...starting,
                status: 'running',
                phase: null,
                cameras: ['wrist'],
            });
        });

        expect(await screen.findByRole('heading', { name: 'Simulation running' })).toBeInTheDocument();
        expect(onRunning).toHaveBeenCalledTimes(1);
    });
});

describe('SimulationNotRunning', () => {
    it('offers to start and to reconnect after a start outside Studio', async () => {
        const onStart = rs.fn();
        const onReconnect = rs.fn();
        const user = userEvent.setup();
        render(<SimulationNotRunning robot={robot} onStart={onStart} onReconnect={onReconnect} />);

        expect(await screen.findByRole('heading', { name: 'Simulation not running' })).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Start simulation…' }));
        expect(onStart).toHaveBeenCalled();
        await user.click(screen.getByRole('button', { name: 'Reconnect' }));
        expect(onReconnect).toHaveBeenCalled();
    });

    it('offers the progress of a simulation that is starting', async () => {
        render(<SimulationNotRunning robot={robot} isStarting onStart={rs.fn()} />);

        expect(await screen.findByRole('heading', { name: 'Simulation starting' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Show progress…' })).toBeInTheDocument();
    });

    it('has no start button for a robot type without a simulation', async () => {
        render(<SimulationNotRunning robot={robot} />);

        expect(await screen.findByRole('heading', { name: 'Simulation not running' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Start simulation…' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Reconnect' })).not.toBeInTheDocument();
    });
});

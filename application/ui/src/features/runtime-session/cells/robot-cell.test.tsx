// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { act, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { $api } from '../../../api/client';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { createQueryClient } from '../../../query-client/query-client';
import { render } from '../../../test-utils/render';
import { resetRecoveredSimulationRuns, RobotCell } from './robot-cell';

const retry = rs.fn();
const send = rs.fn(async () => ({ type: 'ack', request_id: '', ok: true }));
const session = {
    environment_id: null as string | null,
    environmentState: 'failed',
};

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: {},
    failureOf: (statuses: Record<string, { state: string; error?: string }>, component: string) =>
        statuses[component]?.state === 'failed' ? (statuses[component].error ?? 'failed') : null,
    useSessionJoints: () => undefined,
    useRuntimeSession: () => ({
        state: {
            environment: { environment_id: session.environment_id, robot_id: ROBOT_ID, leader_id: null, cameras: [] },
        },
        error: null,
        isLoading: false,
        statuses: {
            environment: {
                type: 'status',
                component: 'environment',
                state: session.environmentState,
                error: 'follower: not running',
            },
        },
        send,
        retry,
    }),
}));

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';
const ENVIRONMENT_ID = 'environment-1';

const mockSimulatedFollower = () =>
    server.use(
        http.get('/api/robots/catalog', () =>
            HttpResponse.json([
                {
                    type: 'SO101_Follower',
                    display_name: 'SO101 Follower',
                    role: 'follower',
                    urdf_path: '',
                    package_map: {},
                    joint_map: {},
                    zero_calibration: null,
                    simulation: {
                        scenes: [{ id: 'single_pick_place', display_name: 'Pick & place', description: '' }],
                        default_scene: 'single_pick_place',
                        max_seed: null,
                        labels: ['1 arm'],
                        noun: 'simulation',
                        cameras: true,
                    },
                },
            ])
        ),
        // Cast the whole response: the robot union has more than 25 members, past what TypeScript relates member by member.
        http.get(
            '/api/projects/{project_id}/robots/{robot_id}',
            () =>
                HttpResponse.json({
                    id: ROBOT_ID,
                    name: 'Sim arm',
                    type: 'SO101_Follower',
                    payload: { connection_string: '', serial_number: '' },
                }) as never
        ),
        http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
            HttpResponse.json({ running: false } as never)
        ),
        http.get('/api/simulation', () => HttpResponse.json({ status: 'idle', cameras: [] })),
        http.get('/api/projects/{project_id}/environments', () =>
            HttpResponse.json([
                {
                    id: ENVIRONMENT_ID,
                    name: 'Sim bench',
                    robots: [{ robot_id: ROBOT_ID, tele_operator: { type: 'none' } }],
                    cameras: [],
                },
            ] as never)
        )
    );

const renderCell = (queryClient = createQueryClient()) => {
    const { unmount } = render(<RobotCell follower_id={ROBOT_ID} />, {
        route: `/projects/${PROJECT_ID}/environments`,
        path: '/projects/:project_id/environments',
        queryClient,
    });
    const simulationKey = $api.queryOptions('get', '/api/projects/{project_id}/robots/{robot_id}/simulation', {
        params: { path: { project_id: PROJECT_ID, robot_id: ROBOT_ID } },
    }).queryKey;
    const setRunning = (running: boolean) =>
        act(() => {
            queryClient.setQueryData(
                simulationKey,
                (running ? { running, started_by: 'studio' } : { running }) as never
            );
        });

    return { setRunning, unmount, queryClient };
};

describe('RobotCell of a simulated follower', () => {
    beforeEach(() => {
        retry.mockClear();
        resetRecoveredSimulationRuns();
        send.mockClear();
        session.environment_id = null;
        session.environmentState = 'failed';
    });

    it('offers to start the simulation, and loads the page environment again once it runs', async () => {
        mockSimulatedFollower();
        const { setRunning } = renderCell();

        expect(await screen.findByRole('heading', { name: 'Simulation not running' })).toBeInTheDocument();
        expect(retry).not.toHaveBeenCalled();

        setRunning(true);

        await waitFor(() => expect(retry).toHaveBeenCalledTimes(1));
        expect(send).not.toHaveBeenCalled();
    });

    it('loads the saved environment that is loaded now, not the page one', async () => {
        mockSimulatedFollower();
        session.environment_id = ENVIRONMENT_ID;
        const { setRunning } = renderCell();
        await screen.findByRole('heading', { name: 'Simulation not running' });

        setRunning(true);

        await waitFor(() =>
            expect(send).toHaveBeenCalledWith({
                command: 'load_environment',
                environment: {
                    id: ENVIRONMENT_ID,
                    name: 'Sim bench',
                    robots: [{ robot_id: ROBOT_ID, tele_operator: { type: 'none' } }],
                    cameras: [],
                },
                unload_incompatible: false,
            })
        );
        expect(retry).not.toHaveBeenCalled();
    });

    it('retries once when a running simulation sees the load fail, and again only once it runs again', async () => {
        mockSimulatedFollower();
        session.environmentState = 'connecting';
        const { setRunning } = renderCell();
        await screen.findByRole('heading', { name: 'Simulation not running' });

        // The simulation is up before the load in flight fails.
        setRunning(true);
        await screen.findByText('Simulation running');
        expect(retry).not.toHaveBeenCalled();
        session.environmentState = 'failed';
        setRunning(true);
        await waitFor(() => expect(retry).toHaveBeenCalledTimes(1));

        // Still failed: no loop.
        setRunning(true);
        await new Promise((resolve) => setTimeout(resolve, 50));
        expect(retry).toHaveBeenCalledTimes(1);

        setRunning(false);
        await screen.findByRole('heading', { name: 'Simulation not running' });
        setRunning(true);
        await waitFor(() => expect(retry).toHaveBeenCalledTimes(2));
    });

    it('does not retry again when the cell mounts again in the same run', async () => {
        mockSimulatedFollower();
        const first = renderCell();
        await screen.findByRole('heading', { name: 'Simulation not running' });
        first.setRunning(true);
        await waitFor(() => expect(retry).toHaveBeenCalledTimes(1));

        // A load in progress unmounts the dock; the cell comes back while the simulation still runs.
        first.unmount();
        const second = renderCell(first.queryClient);
        second.setRunning(true);
        await screen.findByText('Simulation running');
        await new Promise((resolve) => setTimeout(resolve, 50));

        expect(retry).toHaveBeenCalledTimes(1);
    });

    it('sends nothing when another environment is loaded while it fetches the saved one', async () => {
        mockSimulatedFollower();
        let release: () => void = () => {};
        const released = new Promise<void>((resolve) => (release = resolve));
        server.use(
            http.get('/api/projects/{project_id}/environments', async () => {
                await released;
                return HttpResponse.json([{ id: ENVIRONMENT_ID, name: 'Sim bench', robots: [], cameras: [] }] as never);
            })
        );
        session.environment_id = ENVIRONMENT_ID;
        const { setRunning } = renderCell();
        await screen.findByRole('heading', { name: 'Simulation not running' });
        setRunning(true);
        await screen.findByText('Simulation running');

        // The workspace swaps to another environment before the fetch returns.
        session.environment_id = 'environment-2';
        setRunning(true);
        await new Promise((resolve) => setTimeout(resolve, 50));
        release();
        await new Promise((resolve) => setTimeout(resolve, 50));

        expect(send).not.toHaveBeenCalled();
    });

    it('does not retry again when the load again connects and fails once more', async () => {
        mockSimulatedFollower();
        const { setRunning } = renderCell();
        await screen.findByRole('heading', { name: 'Simulation not running' });
        setRunning(true);
        await waitFor(() => expect(retry).toHaveBeenCalledTimes(1));

        session.environmentState = 'connecting';
        setRunning(true);
        await new Promise((resolve) => setTimeout(resolve, 50));
        session.environmentState = 'failed';
        setRunning(true);
        await new Promise((resolve) => setTimeout(resolve, 50));

        expect(retry).toHaveBeenCalledTimes(1);
    });
});

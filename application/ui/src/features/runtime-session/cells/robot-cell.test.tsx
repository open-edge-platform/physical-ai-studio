// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, rs } from '@rstest/core';
import { act, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { $api } from '../../../api/client';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { createQueryClient } from '../../../query-client/query-client';
import { render } from '../../../test-utils/render';
import { RobotCell } from './robot-cell';

const retry = rs.fn();

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: {},
    failureOf: (statuses: Record<string, { state: string; error?: string }>, component: string) =>
        statuses[component]?.state === 'failed' ? (statuses[component].error ?? 'failed') : null,
    useSessionJoints: () => undefined,
    useRuntimeSession: () => ({
        state: { environment: { robot_id: ROBOT_ID, leader_id: null, cameras: [] } },
        error: null,
        isLoading: false,
        statuses: {
            environment: { type: 'status', component: 'environment', state: 'failed', error: 'follower: not running' },
        },
        send: rs.fn(),
        retry,
    }),
}));

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';

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
        http.get('/api/simulation', () => HttpResponse.json({ status: 'idle', cameras: [] }))
    );

describe('RobotCell of a simulated follower', () => {
    it('offers to start the simulation, and loads the environment again once it runs', async () => {
        mockSimulatedFollower();
        const queryClient = createQueryClient();
        render(<RobotCell follower_id={ROBOT_ID} />, {
            route: `/projects/${PROJECT_ID}/environments`,
            path: '/projects/:project_id/environments',
            queryClient,
        });

        expect(await screen.findByRole('heading', { name: 'Simulation not running' })).toBeInTheDocument();
        expect(retry).not.toHaveBeenCalled();

        const simulationKey = $api.queryOptions('get', '/api/projects/{project_id}/robots/{robot_id}/simulation', {
            params: { path: { project_id: PROJECT_ID, robot_id: ROBOT_ID } },
        }).queryKey;
        act(() => {
            queryClient.setQueryData(simulationKey, { running: true, started_by: 'studio' } as never);
        });

        await waitFor(() => expect(retry).toHaveBeenCalledTimes(1));
    });
});

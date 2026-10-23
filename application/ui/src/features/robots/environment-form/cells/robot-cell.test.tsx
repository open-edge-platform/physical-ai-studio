// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, rs } from '@rstest/core';
import { act, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { $api } from '../../../../api/client';
import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { createQueryClient } from '../../../../query-client/query-client';
import { render } from '../../../../test-utils/render';
import { RobotCell } from './robot-cell';

const restart = rs.fn();

rs.mock('../../use-joint-state', () => ({
    useJointState: () => ({
        joints: [],
        state: { connected: false, has_leader: false, follower_source: 'hold' },
        error: "No MuJoCo simulation named 'sim' is running. Start it with: uv run physicalai-mujoco start",
        errorCode: 'simulation_not_running',
        warning: null,
        setFollowerSource: rs.fn(),
        restart,
    }),
    useSynchronizeModelJoints: rs.fn(),
}));

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';

describe('RobotCell of a simulated follower', () => {
    it('shows the not-running panel and reconnects once Studio runs the simulation', async () => {
        server.use(
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
            http.get('/api/simulation', () => HttpResponse.json({ status: 'idle', cameras: [] }))
        );
        const queryClient = createQueryClient();
        render(<RobotCell follower_id={ROBOT_ID} camera_ids={[]} />, {
            route: `/projects/${PROJECT_ID}/environments`,
            path: '/projects/:project_id/environments',
            queryClient,
        });

        expect(await screen.findByRole('heading', { name: 'Simulation not running' })).toBeInTheDocument();
        const simulationKey = $api.queryOptions('get', '/api/simulation').queryKey;
        await waitFor(() => expect(queryClient.getQueryData(simulationKey)).toBeDefined());
        expect(restart).not.toHaveBeenCalled();

        act(() => {
            queryClient.setQueryData(simulationKey, {
                status: 'running',
                robot_id: ROBOT_ID,
                cameras: [],
            });
        });

        await waitFor(() => expect(restart).toHaveBeenCalledTimes(1));
    });
});

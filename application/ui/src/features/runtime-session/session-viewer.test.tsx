// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { SessionViewer } from './session-viewer';

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';

rs.mock('dockview-react', () => ({ DockviewReact: () => <div>Dock</div> }));

rs.mock('./workspace/part-controls', () => ({ usePartControls: () => ({ pendingPart: null, act: rs.fn() }) }));

rs.mock('./runtime-session-provider', () => ({
    failureOf: (statuses: Record<string, { state: string; error?: string }>, component: string) =>
        statuses[component]?.state === 'failed' ? (statuses[component].error ?? 'failed') : null,
    useRuntimeSession: () => ({
        state: { environment: { environment_id: null, robot_id: ROBOT_ID, leader_id: null, cameras: [] } },
        error: null,
        loading: null,
        statuses: {
            environment: { type: 'status', component: 'environment', state: 'failed', error: 'follower: not running' },
        },
        retry: rs.fn(),
    }),
}));

const mockFollower = (simulated: boolean) =>
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
                    simulation: simulated
                        ? {
                              scenes: [{ id: 'single_pick_place', display_name: 'Pick & place', description: '' }],
                              default_scene: 'single_pick_place',
                              max_seed: null,
                              labels: ['1 arm'],
                              noun: 'simulation',
                              cameras: true,
                          }
                        : null,
                },
            ])
        ),
        http.get(
            '/api/projects/{project_id}/robots/{robot_id}',
            () =>
                HttpResponse.json({
                    id: ROBOT_ID,
                    name: 'Arm',
                    type: 'SO101_Follower',
                    payload: { connection_string: '', serial_number: '' },
                }) as never
        )
    );

const renderViewer = () =>
    render(<SessionViewer cancelHref='/' />, {
        route: `/projects/${PROJECT_ID}/record`,
        path: '/projects/:project_id/record',
    });

describe('SessionViewer when the first load fails', () => {
    it('shows the dock for a simulated follower, whose cell offers to start the simulation', async () => {
        mockFollower(true);
        renderViewer();

        expect(await screen.findByText('Dock')).toBeInTheDocument();
    });

    it('keeps showing the failure for any other follower', async () => {
        mockFollower(false);
        renderViewer();

        expect(await screen.findByText('follower: not running')).toBeInTheDocument();
        expect(screen.queryByText('Dock')).not.toBeInTheDocument();
    });
});

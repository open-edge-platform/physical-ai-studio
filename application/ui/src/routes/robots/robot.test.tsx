// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from '@rstest/core';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { Robot } from './robot';

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';
const robot = {
    id: ROBOT_ID,
    name: 'Sim arm',
    type: 'SO101_Follower' as const,
    payload: { connection_string: '', serial_number: '' },
};

const mockSimulatedRobot = (connection_status: 'online' | 'offline', simulation: object) => {
    server.use(
        http.get('/api/projects/{project_id}/robots/online', () =>
            HttpResponse.json([{ ...robot, connection_status }])
        ),
        http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
            HttpResponse.json({
                running: false,
                started_by: null,
                owner_name: 'sim',
                http_url: null,
                viewer_url: null,
                scene: null,
                cameras: [],
            })
        ),
        // After /online: `{robot_id}` would match it too.
        // Cast the whole response: the robot union has more than 25 members, past what TypeScript relates member by member.
        http.get('/api/projects/{project_id}/robots/{robot_id}', () => HttpResponse.json(robot) as never),
        http.get('/api/simulation', () => HttpResponse.json({ cameras: [], ...simulation } as never)),
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
                        labels: [],
                    },
                },
            ])
        )
    );
};

const renderRobotPage = () =>
    render(<Robot />, {
        route: `/projects/${PROJECT_ID}/robots/${ROBOT_ID}`,
        path: '/projects/:project_id/robots/:robot_id',
    });

describe('Robot page of a simulated robot', () => {
    it('offers to start the simulation when it is not running', async () => {
        mockSimulatedRobot('offline', { status: 'idle' });
        renderRobotPage();

        expect(await screen.findByRole('heading', { name: 'Simulation not running' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Start simulation…' })).toBeInTheDocument();
        // It notices a simulation started elsewhere by itself: no Retry.
        expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
    });

    it('offers the progress while Studio starts the simulation', async () => {
        mockSimulatedRobot('offline', { status: 'starting', robot_id: ROBOT_ID });
        renderRobotPage();

        expect(await screen.findByRole('heading', { name: 'Simulation starting' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Show progress…' })).toBeInTheDocument();
    });

    it('shows the viewer of a simulation started outside Studio', async () => {
        mockSimulatedRobot('online', { status: 'idle' });
        server.use(
            http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
                HttpResponse.json({
                    running: true,
                    started_by: 'external',
                    owner_name: 'sim',
                    http_url: 'http://127.0.0.1:8080',
                    viewer_url: 'http://127.0.0.1:8081',
                    viewer_proxy_url: '/api/projects/project-1/robots/robot-1/simulation/viewer/',
                    scene: 'yahtzee',
                    cameras: [],
                })
            )
        );
        renderRobotPage();

        // Through Studio's own address, not the simulation's port: the browser may not reach that (Docker).
        expect(await screen.findByTitle('Sim arm simulation viewer')).toHaveAttribute(
            'src',
            '/api/projects/project-1/robots/robot-1/simulation/viewer/'
        );
        expect(screen.getByRole('link', { name: 'Open viewer ↗' })).toHaveAttribute(
            'href',
            '/api/projects/project-1/robots/robot-1/simulation/viewer/'
        );
        expect(screen.getByText('started outside Studio')).toBeInTheDocument();
        expect(screen.queryByText(/real time/)).not.toBeInTheDocument();
        expect(screen.queryByText(':8080 / :8081')).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Stop simulation' })).not.toBeInTheDocument();
        expect(screen.queryByRole('heading', { name: 'Simulation not running' })).not.toBeInTheDocument();
    });

    it('lets Studio stop the simulation it started', async () => {
        mockSimulatedRobot('online', { status: 'running', robot_id: ROBOT_ID });
        server.use(
            http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
                HttpResponse.json({
                    running: true,
                    started_by: 'studio',
                    owner_name: 'sim',
                    http_url: 'http://127.0.0.1:56209',
                    viewer_url: 'http://127.0.0.1:56210',
                    scene: 'yahtzee',
                    cameras: [],
                })
            )
        );
        renderRobotPage();

        expect(await screen.findByRole('button', { name: 'Stop simulation' })).toBeInTheDocument();
        expect(screen.queryByText(/started by/)).not.toBeInTheDocument();
    });
});

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { SchemaEnvironmentWithRelations, SchemaSimulationCameraInfo } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { SimulationCamerasPrompt } from './simulation-cameras-prompt';

const PROJECT_ID = 'project-1';
const SIM = 'http://127.0.0.1:56209';
const simulationCamera = (id: string, robot_id: string, camera: string) => ({
    id,
    name: `Sim arm ${camera}`,
    driver: 'simulation',
    fingerprint: { robot_id, camera },
    hardware_name: camera,
    payload: { robot_id, camera },
});

const robot = { id: 'robot-1', name: 'Sim arm', type: 'SO101_Follower', payload: { connection_string: '' } };

const environment = {
    id: 'env-1',
    name: 'Sim env',
    robots: [{ robot, tele_operator: { type: 'none' } }],
    cameras: [simulationCamera('cam-1', 'robot-1', 'wrist')],
} as unknown as SchemaEnvironmentWithRelations;

const camera = (name: string, source: SchemaSimulationCameraInfo['source']) => ({
    name,
    width: 640,
    height: 480,
    fps: 30,
    source,
    stream_url: `${SIM}/cameras/${name}/mjpeg`,
    snapshot_url: `${SIM}/cameras/${name}/frame.jpg`,
    snapshot_proxy_url: `/api/projects/${PROJECT_ID}/robots/robot-1/simulation/http/cameras/${name}/frame.jpg`,
});

const mockSimulation = (running: boolean) =>
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
                    simulation: { scenes: [], default_scene: 'x', max_seed: null, labels: [] },
                },
            ])
        ),
        http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
            HttpResponse.json({
                running,
                started_by: running ? 'studio' : null,
                owner_name: 'sim',
                http_url: running ? SIM : null,
                viewer_url: null,
                scene: null,
                cameras: running ? [camera('wrist', 'override'), camera('overview', 'scene')] : [],
            })
        )
    );

const renderPrompt = (shown: SchemaEnvironmentWithRelations = environment) =>
    render(<SimulationCamerasPrompt environment={shown} />, {
        route: `/projects/${PROJECT_ID}/environments/env-1`,
        path: '/projects/:project_id/environments/:environment_id',
    });

describe('SimulationCamerasPrompt', () => {
    afterEach(() => window.sessionStorage.clear());

    it('offers the cameras the environment does not have and adds the chosen ones', async () => {
        mockSimulation(true);
        const added: unknown[] = [];
        server.use(
            http.post(
                '/api/projects/{project_id}/environments/{environment_id}/simulation-cameras',
                async ({ request }) => {
                    added.push(await request.json());
                    return HttpResponse.json([], { status: 201 });
                }
            )
        );
        const user = userEvent.setup();
        renderPrompt();

        expect(await screen.findByRole('heading', { name: 'Add simulated cameras?' })).toBeInTheDocument();
        expect(screen.getByRole('checkbox', { name: 'overview' })).toBeChecked();
        expect(screen.queryByRole('checkbox', { name: 'wrist' })).not.toBeInTheDocument();
        expect(screen.getByText('scene camera · 640×480 · 30 fps')).toBeInTheDocument();
        // The preview comes through Studio's own address, not the simulation's port.
        expect(screen.getByRole('img', { name: 'overview snapshot' })).toHaveAttribute(
            'src',
            `/api/projects/${PROJECT_ID}/robots/robot-1/simulation/http/cameras/overview/frame.jpg`
        );
        await user.click(screen.getByRole('button', { name: 'Add 1 camera' }));

        await waitFor(() => expect(added).toEqual([{ robot_id: robot.id, cameras: ['overview'], names: {} }]));
    });

    it("offers a camera when the environment has another robot's camera of the same name", async () => {
        mockSimulation(true);
        renderPrompt({
            ...environment,
            cameras: [simulationCamera('cam-0', 'robot-0', 'wrist')],
        } as unknown as SchemaEnvironmentWithRelations);

        expect(await screen.findByRole('checkbox', { name: 'wrist' })).toBeChecked();
        expect(screen.getByRole('checkbox', { name: 'overview' })).toBeChecked();
    });

    it("offers a camera when the environment has only an IP camera on the simulation's stream", async () => {
        mockSimulation(true);
        const onTheStream = {
            id: 'cam-0',
            name: 'Wrist stream',
            driver: 'ipcam',
            fingerprint: { url: `${SIM}/cameras/wrist/mjpeg` },
            hardware_name: null,
            payload: { url: `${SIM}/cameras/wrist/mjpeg`, fps: 30 },
        };
        renderPrompt({ ...environment, cameras: [onTheStream] } as unknown as SchemaEnvironmentWithRelations);

        expect(await screen.findByRole('checkbox', { name: 'wrist' })).toBeChecked();
    });

    it("counts the robot's simulation cameras the environment has as present", async () => {
        mockSimulation(true);
        renderPrompt({
            ...environment,
            cameras: [simulationCamera('cam-1', robot.id, 'wrist'), simulationCamera('cam-2', robot.id, 'overview')],
        } as unknown as SchemaEnvironmentWithRelations);

        await new Promise((resolve) => setTimeout(resolve, 50));
        expect(screen.queryByRole('heading', { name: 'Add simulated cameras?' })).not.toBeInTheDocument();
    });

    it('adds a camera under the name the user gives it', async () => {
        mockSimulation(true);
        const added: unknown[] = [];
        server.use(
            http.post(
                '/api/projects/{project_id}/environments/{environment_id}/simulation-cameras',
                async ({ request }) => {
                    added.push(await request.json());
                    return HttpResponse.json([], { status: 201 });
                }
            )
        );
        const user = userEvent.setup();
        renderPrompt();

        const name = await screen.findByRole('textbox', { name: 'Name for overview' });
        expect(name).toHaveValue(`${robot.name} overview`);
        await user.clear(name);
        await user.type(name, 'Top view');
        await user.click(screen.getByRole('button', { name: 'Add 1 camera' }));

        await waitFor(() =>
            expect(added).toEqual([{ robot_id: robot.id, cameras: ['overview'], names: { overview: 'Top view' } }])
        );
    });

    it('stays hidden for the browser session after Not now', async () => {
        mockSimulation(true);
        const user = userEvent.setup();
        const { unmount } = renderPrompt();

        await user.click(await screen.findByRole('button', { name: 'Not now' }));
        expect(screen.queryByRole('heading', { name: 'Add simulated cameras?' })).not.toBeInTheDocument();
        unmount();

        renderPrompt();
        await new Promise((resolve) => setTimeout(resolve, 50));
        expect(screen.queryByRole('heading', { name: 'Add simulated cameras?' })).not.toBeInTheDocument();
    });

    it('says nothing while the simulation does not run', async () => {
        mockSimulation(false);
        renderPrompt();

        await new Promise((resolve) => setTimeout(resolve, 50));
        expect(screen.queryByRole('heading', { name: 'Add simulated cameras?' })).not.toBeInTheDocument();
    });
});

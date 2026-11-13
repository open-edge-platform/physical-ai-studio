// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from '@rstest/core';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { SchemaProjectCamera } from '../../api/types';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { CameraFeed } from './camera-feed';
import { RETRY_STREAM_MS } from './simulation-camera-stream';

const PROJECT_ID = 'project-1';

const simulationCamera = {
    id: 'cam-1',
    name: 'Sim arm wrist',
    driver: 'simulation',
    fingerprint: { robot_id: 'robot-1', camera: 'wrist' },
    hardware_name: 'wrist',
    payload: { robot_id: 'robot-1', camera: 'wrist' },
} satisfies SchemaProjectCamera;

const mockSimulation = (simulation: { running: boolean; http_url?: string }) =>
    server.use(
        http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
            HttpResponse.json({ owner_name: 'sim', cameras: [], ...simulation })
        )
    );

const renderFeed = () =>
    render(<CameraFeed camera={simulationCamera} />, {
        route: `/projects/${PROJECT_ID}/cameras/cam-1`,
        path: '/projects/:project_id/cameras/:camera_id',
    });

describe('CameraFeed of a simulation camera', () => {
    it("streams through Studio's own address, without a resolution of its own", async () => {
        mockSimulation({ running: true, http_url: 'http://127.0.0.1:50001' });
        renderFeed();

        expect(await screen.findByRole('img', { name: 'Camera: Sim arm wrist' })).toHaveAttribute(
            'src',
            `/api/projects/${PROJECT_ID}/robots/robot-1/simulation/http/cameras/wrist/mjpeg`
        );
        expect(screen.getByText('Simulation camera wrist')).toBeInTheDocument();
        expect(screen.queryByText(/ x .* @ /)).not.toBeInTheDocument();
    });

    it('asks to start the simulation while the stream does not load, and tries it again', async () => {
        mockSimulation({ running: true, http_url: 'http://127.0.0.1:50001' });
        renderFeed();

        fireEvent.error(await screen.findByRole('img', { name: 'Camera: Sim arm wrist' }));

        expect(await screen.findByText('Start the simulation to see this camera.')).toBeInTheDocument();
        await waitFor(
            () =>
                expect(screen.getByRole('img', { name: 'Camera: Sim arm wrist' })).toHaveAttribute(
                    'src',
                    `/api/projects/${PROJECT_ID}/robots/robot-1/simulation/http/cameras/wrist/mjpeg?attempt=1`
                ),
            { timeout: RETRY_STREAM_MS + 2_000 }
        );
        expect(screen.queryByText('Start the simulation to see this camera.')).not.toBeInTheDocument();
    });

    it('asks to start the simulation while it does not run', async () => {
        mockSimulation({ running: false });
        renderFeed();

        expect(await screen.findByText('Start the simulation to see this camera.')).toBeInTheDocument();
        expect(screen.queryByRole('img', { name: 'Camera: Sim arm wrist' })).not.toBeInTheDocument();
    });
});

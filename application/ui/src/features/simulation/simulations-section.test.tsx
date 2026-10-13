// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { SimulationsSection } from './simulations-section';

const PROJECT_ID = 'project-1';

const projectSimulation = (robot_id: string, robot_name: string, started_by: 'studio' | 'external', port: number) => ({
    running: true,
    started_by,
    owner_name: robot_name,
    http_url: `http://127.0.0.1:${port}`,
    viewer_url: `http://127.0.0.1:${port + 1}`,
    scene: 'yahtzee',
    cameras: [],
    robot_id,
    robot_name,
    robot_type: 'MuJoCo_SO101_Follower',
});

const renderSection = () =>
    render(<SimulationsSection />, { route: `/projects/${PROJECT_ID}`, path: '/projects/:project_id' });

describe('SimulationsSection', () => {
    it("lists Studio's simulation with Stop and an outside one with Open", async () => {
        server.use(
            http.get('/api/simulation', () =>
                HttpResponse.json({
                    status: 'running',
                    robot_id: 'robot-a',
                    robot_type: 'MuJoCo_SO101_Follower',
                    cameras: [],
                })
            ),
            http.get('/api/projects/{project_id}/simulations', () =>
                HttpResponse.json([
                    projectSimulation('robot-a', 'Sim arm', 'studio', 8181),
                    projectSimulation('robot-b', 'CLI arm', 'external', 8080),
                ])
            )
        );
        let stopped = false;
        server.use(
            http.delete('/api/simulation', () => {
                stopped = true;
                return HttpResponse.json({ status: 'stopping', cameras: [] });
            })
        );
        const user = userEvent.setup();
        renderSection();

        expect(await screen.findByText('Sim arm · yahtzee')).toBeInTheDocument();
        expect(screen.getByText('Started by Studio')).toBeInTheDocument();
        expect(screen.getByText('Started outside Studio (CLI)')).toBeInTheDocument();
        expect(screen.getByRole('link', { name: 'Open ↗' })).toHaveAttribute('href', 'http://127.0.0.1:8081');
        await user.click(screen.getByRole('button', { name: 'Stop' }));
        await waitFor(() => expect(stopped).toBe(true));
    });

    it("lists Studio's simulation from another project by its type", async () => {
        server.use(
            http.get('/api/simulation', () =>
                HttpResponse.json({
                    status: 'running',
                    robot_id: 'elsewhere',
                    robot_type: 'MuJoCo_SO101_Follower',
                    scene: 'garment_fold',
                    cameras: [],
                })
            ),
            http.get('/api/projects/{project_id}/simulations', () => HttpResponse.json([]))
        );
        renderSection();

        expect(await screen.findByText('MuJoCo SO101 Follower · garment_fold')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument();
    });

    it('shows nothing without simulations', async () => {
        server.use(
            http.get('/api/simulation', () => HttpResponse.json({ status: 'idle', cameras: [] })),
            http.get('/api/projects/{project_id}/simulations', () => HttpResponse.json([]))
        );
        renderSection();

        await new Promise((resolve) => setTimeout(resolve, 50));
        expect(screen.queryByRole('heading', { name: 'Simulations' })).not.toBeInTheDocument();
    });
});

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { $api } from '../../api/client';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { createQueryClient } from '../../query-client/query-client';
import { render } from '../../test-utils/render';
import { SessionViewer } from './session-viewer';

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';

rs.mock('dockview-react', () => ({ DockviewReact: () => <div>Dock</div> }));

const loadEnvironmentDraft = rs.fn();
rs.mock('./workspace/part-controls', () => ({
    usePartControls: () => ({ pendingPart: null, act: rs.fn(), loadEnvironmentDraft }),
}));

const retry = rs.fn();
const send = rs.fn(async () => ({ type: 'ack', request_id: '', ok: true }));
const PAGE_DRAFT = { name: 'Bench', robots: [{ robot_id: ROBOT_ID, tele_operator: { type: 'none' } }], cameras: [] };
/** What the mocked session holds: by default an environment that loaded, failed. */
const session: {
    loaded: boolean;
    error: { message: string; error_code: string } | null;
    /** What the last environment load asked for, the page's draft unless a test says otherwise. */
    lastDraft: object;
} = {
    loaded: true,
    error: null,
    lastDraft: PAGE_DRAFT,
};

rs.mock('./runtime-session-provider', () => ({
    failureOf: (statuses: Record<string, { state: string; error?: string }>, component: string) =>
        statuses[component]?.state === 'failed' ? (statuses[component].error ?? 'failed') : null,
    useRuntimeSession: () => ({
        draft: PAGE_DRAFT,
        environmentLoad: session.error ? { draft: session.lastDraft, error: session.error } : null,
        state: {
            environment: session.loaded
                ? { environment_id: null, robot_id: ROBOT_ID, leader_id: null, cameras: [] }
                : null,
        },
        error: session.error,
        loading: null,
        statuses: session.loaded
            ? {
                  environment: {
                      type: 'status',
                      component: 'environment',
                      state: 'failed',
                      error: 'follower: not running',
                  },
              }
            : {},
        retry,
        send,
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

const renderViewer = (queryClient = createQueryClient()) =>
    render(<SessionViewer cancelHref='/' />, {
        route: `/projects/${PROJECT_ID}/record`,
        path: '/projects/:project_id/record',
        queryClient,
    });

const setRunning = (queryClient: ReturnType<typeof createQueryClient>) =>
    act(() => {
        queryClient.setQueryData(
            $api.queryOptions('get', '/api/projects/{project_id}/robots/{robot_id}/simulation', {
                params: { path: { project_id: PROJECT_ID, robot_id: ROBOT_ID } },
            }).queryKey,
            { running: true, started_by: 'studio', owner_name: 'arm', cameras: [] } as never
        );
    });

beforeEach(() => {
    retry.mockClear();
    send.mockClear();
    loadEnvironmentDraft.mockClear();
    session.lastDraft = PAGE_DRAFT;
    session.loaded = true;
    session.error = null;
});

describe('SessionViewer when the simulation of its cameras does not run', () => {
    const refuseLoading = () => {
        session.loaded = false;
        session.error = {
            message: "Start the simulation of 'Arm' to use its camera 'Arm wrist'.",
            error_code: 'simulation_not_running',
        };
        server.use(
            http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
                HttpResponse.json({ running: false, owner_name: 'arm' } as never)
            ),
            http.get('/api/simulation', () => HttpResponse.json({ status: 'idle', cameras: [] } as never))
        );
    };

    it('offers to start the simulation of the page follower', async () => {
        mockFollower(true);
        refuseLoading();
        renderViewer();

        expect(await screen.findByRole('heading', { name: 'Simulation not running' })).toBeInTheDocument();
        expect(screen.getByText('Start the simulation of Arm to load Bench.')).toBeInTheDocument();
        // Only how to start it: the refusal and a plain retry say nothing more.
        expect(screen.queryByText(/to use its camera/)).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Try again' })).not.toBeInTheDocument();
        expect(screen.getByText('Cancel')).toBeInTheDocument();
        await userEvent.setup().click(screen.getByRole('button', { name: 'Start simulation…' }));
        expect(await screen.findByRole('dialog')).toBeInTheDocument();
    });

    it('loads the environment again once the simulation runs', async () => {
        mockFollower(true);
        refuseLoading();
        const queryClient = createQueryClient();
        renderViewer(queryClient);
        await screen.findByRole('heading', { name: 'Simulation not running' });
        expect(retry).not.toHaveBeenCalled();

        setRunning(queryClient);

        await waitFor(() => expect(retry).toHaveBeenCalledTimes(1));
    });

    it('loads an environment the workspace loaded again once the simulation runs', async () => {
        mockFollower(true);
        refuseLoading();
        const swapped = { id: 'environment-2', name: 'Other bench', robots: PAGE_DRAFT.robots, cameras: [] };
        session.lastDraft = swapped;
        const queryClient = createQueryClient();
        renderViewer(queryClient);
        await screen.findByRole('heading', { name: 'Simulation not running' });

        setRunning(queryClient);

        await waitFor(() => expect(loadEnvironmentDraft).toHaveBeenCalledWith(swapped));
        expect(retry).not.toHaveBeenCalled();
    });

    it('offers, over the loaded environment, to start the simulation of one the workspace could not load', async () => {
        mockFollower(true);
        refuseLoading();
        session.loaded = true; // another environment stays loaded
        const swapped = { id: 'environment-2', name: 'Other bench', robots: PAGE_DRAFT.robots, cameras: [] };
        session.lastDraft = swapped;
        const queryClient = createQueryClient();
        renderViewer(queryClient);

        expect(await screen.findByText('Dock')).toBeInTheDocument();
        expect(await screen.findByText(/Other bench could not load/)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Start simulation…' })).toBeInTheDocument();

        // Started elsewhere: it asks before replacing the loaded environment.
        setRunning(queryClient);
        await userEvent.setup().click(await screen.findByRole('button', { name: 'Load Other bench' }));
        expect(loadEnvironmentDraft).toHaveBeenCalledTimes(1);
        expect(loadEnvironmentDraft).toHaveBeenCalledWith(swapped);
        expect(retry).not.toHaveBeenCalled();
    });

    it('offers nothing for a follower without a simulation', async () => {
        mockFollower(false);
        refuseLoading();
        renderViewer();

        expect(await screen.findByText(/Start the simulation of 'Arm'/)).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Start simulation…' })).not.toBeInTheDocument();
    });
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

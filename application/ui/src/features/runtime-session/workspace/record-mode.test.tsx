// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { describeAutopilot } from '../../simulation/autopilot-status';
import * as actualProvider from '../runtime-session-provider' with { rstest: 'importActual' };
import { RecordMode } from './record-mode';

const PROJECT_ID = 'project-1';
const ROBOT_ID = 'robot-1';
const ENVIRONMENT_ID = 'environment-1';

const send = rs.fn(async () => ({ type: 'ack', request_id: '', ok: true }));
const DATASET = {
    dataset_id: 'dataset-1',
    name: 'Auto',
    fps: 30,
    fps_fixed: false,
    is_recording: false,
    episodes_recorded: 0,
};
const session: {
    autopilot: boolean;
    dataset: typeof DATASET | null;
    autopilot_recording: Record<string, unknown> | null;
} = {
    autopilot: false,
    dataset: null,
    autopilot_recording: null,
};

rs.mock('../runtime-session-provider', () => ({
    ERROR_TOAST: {},
    failureOf: () => null,
    savedEnvironmentDraft: actualProvider.savedEnvironmentDraft,
    useRuntimeSession: () => ({
        state: {
            environment: {
                environment_id: ENVIRONMENT_ID,
                name: 'Sim bench',
                robot_id: ROBOT_ID,
                leader_id: session.autopilot ? 'autopilot-leader' : null,
                autopilot: session.autopilot,
                joint_names: [],
                cameras: [],
            },
            control: null,
            dataset: session.dataset,
            autopilot_recording: session.autopilot_recording,
        },
        statuses: {},
        send,
    }),
}));

const SUPPORTED = { supported: true, reason: null, options: ['max_attempts', 'recovery_fraction'] };
let support: { supported: boolean; reason: string | null; options: string[] } = SUPPORTED;

beforeEach(() => {
    send.mockClear();
    session.autopilot = false;
    session.dataset = null;
    session.autopilot_recording = null;
    support = SUPPORTED;
    server.use(
        http.get('/api/settings', () => HttpResponse.json({ hotkeys: { bindings: {} } } as never)),
        http.get('/api/dataset/{dataset_id}', () => HttpResponse.json({ default_task: 'Sort' } as never)),
        http.get('/api/robots/catalog', () =>
            HttpResponse.json([
                {
                    type: 'MuJoCo_SO101_Follower',
                    display_name: 'MuJoCo SO101 Follower',
                    role: 'follower',
                    urdf_path: '',
                    package_map: {},
                    joint_map: {},
                    simulation: {
                        scenes: [{ id: 'conveyor', display_name: 'Conveyor', description: '' }],
                        default_scene: 'conveyor',
                    },
                },
            ] as never)
        ),
        http.get('/api/projects/{project_id}/robots', () =>
            HttpResponse.json([{ id: ROBOT_ID, name: 'Sim arm', type: 'MuJoCo_SO101_Follower', payload: {} }] as never)
        ),
        http.get('/api/projects/{project_id}/environments', () =>
            HttpResponse.json([
                {
                    id: ENVIRONMENT_ID,
                    name: 'Sim bench',
                    robots: [{ robot_id: ROBOT_ID, tele_operator: { type: 'none' } }],
                    cameras: [{ camera_id: 'camera-1' }],
                },
            ] as never)
        ),
        http.get('/api/projects/{project_id}/robots/{robot_id}/simulation', () =>
            HttpResponse.json({
                running: true,
                started_by: 'studio',
                owner_name: 'arm',
                cameras: [],
                autopilot_support: support,
            } as never)
        )
    );
});

const renderMode = () =>
    render(<RecordMode />, { route: `/projects/${PROJECT_ID}/record`, path: '/projects/:project_id/record' });

describe('RecordMode with a simulated follower', () => {
    it('lets the autopilot lead this session without editing the saved environment', async () => {
        renderMode();
        const user = userEvent.setup();

        await user.click(await screen.findByRole('button', { name: /Teleoperator/ }));
        await user.click(await screen.findByRole('option', { name: 'Autopilot (simulation)' }));

        await waitFor(() =>
            expect(send).toHaveBeenCalledWith({
                command: 'load_environment',
                environment: {
                    id: ENVIRONMENT_ID,
                    name: 'Sim bench',
                    robots: [{ robot_id: ROBOT_ID, tele_operator: { type: 'autopilot' } }],
                    cameras: [{ camera_id: 'camera-1' }],
                },
                unload_incompatible: false,
            })
        );
    });

    it('shows what the autopilot does while it leads, from the session', async () => {
        session.autopilot = true;
        session.dataset = { ...DATASET, is_recording: true };
        session.autopilot_recording = {
            phase: 'recording',
            message: '',
            task: 'Put the red cube in the blue bucket.',
            saved: 0,
            discarded: 0,
            discards: {},
            max_episodes: 0,
        };
        renderMode();

        expect(await screen.findByText('Autopilot: leading: Put the red cube in the blue bucket.')).toBeInTheDocument();
        // The simulation records the episodes: no manual ones, and teleoperation goes with the recording.
        expect(screen.queryByRole('switch', { name: 'Teleoperate' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Start episode/ })).not.toBeInTheDocument();
    });

    it('shows no autopilot state for a leader arm', async () => {
        renderMode();

        expect(await screen.findByRole('button', { name: /Teleoperator/ })).toBeInTheDocument();
        expect(screen.queryByText(/Autopilot:/)).not.toBeInTheDocument();
    });
});

describe('RecordMode recording automatically', () => {
    it('starts and stops automatic recording on the session, with no relay', async () => {
        session.autopilot = true;
        session.dataset = DATASET;
        renderMode();
        const user = userEvent.setup();

        await user.click(await screen.findByRole('button', { name: 'Record automatically' }));

        await waitFor(() =>
            expect(send).toHaveBeenCalledWith({
                command: 'start_autopilot_recording',
                dataset_id: 'dataset-1',
                options: { max_episodes: 0, max_attempts: null, recovery_fraction: null },
            })
        );
        expect(send).not.toHaveBeenCalledWith(expect.objectContaining({ command: 'set_control' }));
    });

    it('shows the run, its discards and how to stop it while it runs', async () => {
        session.autopilot = true;
        session.dataset = { ...DATASET, is_recording: true };
        session.autopilot_recording = {
            phase: 'recording',
            message: '',
            task: 'Put the red cube in the blue bucket.',
            saved: 2,
            discarded: 3,
            discards: { failure: 2, stale_stream: 1 },
            max_episodes: 5,
        };
        renderMode();
        const user = userEvent.setup();

        expect(await screen.findByText('Task: Put the red cube in the blue bucket.')).toBeInTheDocument();
        expect(
            screen.getByText('Saved 2 of 5, discarded 3 (2 failed, 1 a camera stopped sending)')
        ).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Start episode/ })).not.toBeInTheDocument();
        expect(await screen.findByRole('button', { name: /Teleoperator/ })).toBeDisabled();

        await user.click(screen.getByRole('button', { name: 'Stop recording' }));
        await waitFor(() => expect(send).toHaveBeenCalledWith({ command: 'stop_autopilot_recording' }));
    });

    it('says why a run failed', async () => {
        session.autopilot = true;
        session.dataset = DATASET;
        session.autopilot_recording = {
            phase: 'failed',
            message: 'No new samples from leader (0.61 s): the follower holds where it is.',
            saved: 1,
            discarded: 1,
            discards: { aborted: 1 },
            max_episodes: 0,
        };
        renderMode();

        expect(await screen.findByText('Failed')).toBeInTheDocument();
        expect(
            screen.getByText('No new samples from leader (0.61 s): the follower holds where it is.')
        ).toBeInTheDocument();
        expect(screen.getByText('Autopilot: off, it leads while you record automatically')).toBeInTheDocument();
        // Enabled once the simulation reports it can record.
        await waitFor(() => expect(screen.getByRole('button', { name: 'Record automatically' })).toBeEnabled());
    });

    it('does not offer to start where the simulation reports it cannot record', async () => {
        session.autopilot = true;
        session.dataset = DATASET;
        support = {
            supported: false,
            reason: 'The native MuJoCo viewer is open: use the web viewer to record automatically.',
            options: [],
        };
        renderMode();

        expect(await screen.findByText(/native MuJoCo viewer is open/)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Record automatically' })).toBeDisabled();
    });

    it('offers no automatic recording for a leader arm', async () => {
        renderMode();

        expect(await screen.findByRole('button', { name: /Start episode/ })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Record automatically' })).not.toBeInTheDocument();
    });
});

describe('describeAutopilot', () => {
    const recording = { message: '', saved: 0, discarded: 0, discards: {}, max_episodes: 0 };

    it('says the autopilot is off while nothing records', () => {
        expect(describeAutopilot(null).variant).toBe('neutral');
        expect(describeAutopilot({ ...recording, phase: 'stopped' }).variant).toBe('neutral');
    });

    it('follows the episode phase of the recording', () => {
        expect(describeAutopilot({ ...recording, phase: 'preparing' }).text).toBe('laying out the next episode');
        expect(describeAutopilot({ ...recording, phase: 'saving' }).variant).toBe('positive');
    });
});

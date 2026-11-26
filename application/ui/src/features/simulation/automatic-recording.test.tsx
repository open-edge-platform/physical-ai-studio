// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { SchemaAutopilotRecordingState, SchemaSessionState } from '../../api/openapi-spec';
import { render } from '../../test-utils/render';
import { AutomaticRecording, describeDiscards, isAutopilotRecording } from './automatic-recording';

const send = rs.fn(async (_command: unknown) => ({ type: 'ack' as const, request_id: '', ok: true }));

const DATASET = {
    dataset_id: 'dataset-1',
    name: 'Auto',
    fps: 30,
    fps_fixed: false,
    is_recording: false,
    episodes_recorded: 0,
};

const RUN: SchemaAutopilotRecordingState = {
    phase: 'recording',
    message: '',
    scene: 'bucket_sort',
    task: 'Put the red cube in the blue bucket.',
    saved: 0,
    discarded: 0,
    discards: {},
    max_episodes: 3,
    last: null,
};

const sessionState = (
    autopilot_recording: SchemaAutopilotRecordingState | null,
    dataset = DATASET
): SchemaSessionState =>
    ({
        session_id: 's',
        environment: null,
        control: null,
        dataset,
        model: null,
        autopilot_recording,
    }) as SchemaSessionState;

beforeEach(() => {
    send.mockClear();
    send.mockImplementation(async () => ({ type: 'ack' as const, request_id: '', ok: true }));
});

describe('AutomaticRecording', () => {
    it('starts on the session with the episodes and the options the user set', async () => {
        render(
            <AutomaticRecording
                state={sessionState(null)}
                canRecord
                send={send}
                options={['max_attempts', 'recovery_fraction']}
            />
        );
        const user = userEvent.setup();

        const episodes = screen.getByRole('textbox', { name: 'Episodes' });
        await user.clear(episodes);
        await user.type(episodes, '3');
        await user.click(screen.getByRole('button', { name: 'More' }));
        const attempts = screen.getByRole('textbox', { name: 'Attempts per episode' });
        await user.type(attempts, '2');
        await user.click(screen.getByRole('button', { name: 'Record automatically' }));

        await waitFor(() =>
            expect(send).toHaveBeenCalledWith({
                command: 'start_autopilot_recording',
                dataset_id: 'dataset-1',
                options: { max_episodes: 3, max_attempts: 2, recovery_fraction: null },
            })
        );
    });

    it('shows only the options the running scene takes', async () => {
        const user = userEvent.setup();
        const { unmount } = render(
            <AutomaticRecording state={sessionState(null)} canRecord send={send} options={[]} />
        );
        expect(screen.queryByRole('button', { name: 'More' })).not.toBeInTheDocument(); // the conveyor takes none
        unmount();

        render(<AutomaticRecording state={sessionState(null)} canRecord send={send} options={['max_attempts']} />);
        await user.click(screen.getByRole('button', { name: 'More' }));
        expect(screen.getByRole('textbox', { name: 'Attempts per episode' })).toBeInTheDocument();
        expect(screen.queryByRole('textbox', { name: 'Stuck starts' })).not.toBeInTheDocument();
    });

    it('sends no option the scene does not take', async () => {
        render(<AutomaticRecording state={sessionState(null)} canRecord send={send} options={[]} />);
        const user = userEvent.setup();

        await user.click(screen.getByRole('button', { name: 'Record automatically' }));

        await waitFor(() =>
            expect(send).toHaveBeenCalledWith({
                command: 'start_autopilot_recording',
                dataset_id: 'dataset-1',
                options: { max_episodes: 0, max_attempts: null, recovery_fraction: null },
            })
        );
    });

    it('cannot start where the simulation says it cannot record, and says why', () => {
        const reason = 'The native MuJoCo viewer is open: its mouse forces move objects without going through the run.';
        render(<AutomaticRecording state={sessionState(null)} canRecord send={send} unavailable={reason} />);

        expect(screen.getByText(reason)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Record automatically' })).toBeDisabled();
    });

    it('waits for the simulation to be checked before it offers to start', () => {
        render(<AutomaticRecording state={sessionState(null)} canRecord send={send} unavailable='' />);

        expect(screen.getByRole('button', { name: 'Record automatically' })).toBeDisabled();
    });

    it('has no keep rule and no task: the simulation names each episode, and only successes are kept', () => {
        render(<AutomaticRecording state={sessionState(null)} canRecord send={send} />);

        expect(screen.queryByRole('radiogroup', { name: 'Keep' })).not.toBeInTheDocument();
        expect(screen.queryByRole('textbox', { name: 'Task' })).not.toBeInTheDocument();
    });

    it('cannot start without a dataset or a working environment', () => {
        render(<AutomaticRecording state={sessionState(null)} canRecord={false} send={send} />);

        expect(screen.getByRole('button', { name: 'Record automatically' })).toBeDisabled();
    });

    it('stays ready to start again when the session refuses', async () => {
        send.mockImplementation(async () => ({
            type: 'ack' as const,
            request_id: '',
            ok: false,
            error: { message: 'Start the simulation from Studio first', error_code: 'simulation_not_running' },
        }));
        render(<AutomaticRecording state={sessionState(null)} canRecord send={send} />);
        const user = userEvent.setup();

        await user.click(screen.getByRole('button', { name: 'Record automatically' }));

        await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Record automatically' })).toBeEnabled());
    });

    it('shows the run and stops it', async () => {
        const state = sessionState({ ...RUN, saved: 1, discarded: 2, discards: { failure: 1, too_short: 1 } });
        render(<AutomaticRecording state={state} canRecord send={send} />);
        const user = userEvent.setup();

        expect(screen.getByText('Recording')).toBeInTheDocument();
        expect(screen.getByText('Task: Put the red cube in the blue bucket.')).toBeInTheDocument();
        expect(screen.getByText('Saved 1 of 3, discarded 2 (1 failed, 1 too short)')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Record automatically' })).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Stop recording' }));
        await waitFor(() => expect(send).toHaveBeenCalledWith({ command: 'stop_autopilot_recording' }));
    });

    it('says why the last episode was discarded', () => {
        const last = {
            id: 'e-0002',
            kept: false,
            outcome: 'aborted',
            reason: 'No new samples from top (0.72 s)',
            discard_reason: 'stale_stream' as const,
        };
        render(<AutomaticRecording state={sessionState({ ...RUN, discarded: 1, last })} canRecord send={send} />);

        expect(screen.getByText('Last discarded: No new samples from top (0.72 s)')).toBeInTheDocument();
        expect(screen.getByText('Saved 0 of 3, discarded 1')).toBeInTheDocument();
    });

    it('says why it stopped once, also when the stop discarded the episode for the same reason', () => {
        const last = {
            id: 'e-0003',
            kept: false,
            outcome: 'aborted',
            reason: 'Stopped.',
            discard_reason: 'aborted' as const,
        };
        const stopped = {
            ...RUN,
            phase: 'stopped' as const,
            message: 'Stopped.',
            discarded: 1,
            discards: { aborted: 1 },
            last,
        };
        render(<AutomaticRecording state={sessionState(stopped)} canRecord send={send} />);

        expect(screen.getAllByText(/Stopped\./)).toHaveLength(1); // the phase reads "Stopped", the message once
        expect(screen.queryByText('Last discarded: Stopped.')).not.toBeInTheDocument();
    });
});

describe('isAutopilotRecording and describeDiscards', () => {
    it('counts a recording as running until it is done, stopped or failed', () => {
        expect(isAutopilotRecording(null)).toBe(false);
        for (const phase of ['starting', 'preparing', 'recording', 'saving', 'stopping'] as const) {
            expect(isAutopilotRecording({ ...RUN, phase })).toBe(true);
        }
        for (const phase of ['done', 'stopped', 'failed'] as const) {
            expect(isAutopilotRecording({ ...RUN, phase })).toBe(false);
        }
    });

    it('lists every reason with its count', () => {
        expect(describeDiscards({ stalled: 2, aborted: 0, dataset: 1 })).toBe('2 stalled, 1 the dataset discarded it');
    });
});

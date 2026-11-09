import { describe, expect, it, rs } from '@rstest/core';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { SchemaErrorMessage, SchemaSessionState, SchemaStatusMessage } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedDataset } from '../../../test-utils/mocks/mock-dataset';
import { getMockedSettings } from '../../../test-utils/mocks/mock-settings';
import { render } from '../../../test-utils/render';
import * as actualProvider from '../runtime-session-provider' with { rstest: 'importActual' };
import { RecordMode } from './record-mode';
import { RecordingCommandProvider } from './recording-command';

const { session, send } = rs.hoisted(() => ({
    session: { current: {} as Record<string, unknown> },
    send: rs.fn(),
}));

rs.mock('../runtime-session-provider', () => ({
    ...actualProvider,
    useRuntimeSession: () => session.current,
}));

const DATASET_ID = '610d64d2-7133-45e9-b979-6a4ef14a589c';

const state = (overrides: Partial<SchemaSessionState> = {}): SchemaSessionState => ({
    session_id: 'session-1',
    environment: {
        environment_id: 'environment-1',
        name: 'Bench',
        robot_id: 'robot-1',
        leader_id: null,
        joint_names: [],
        cameras: [],
    },
    dataset: {
        dataset_id: DATASET_ID,
        name: 'dice',
        fps: 30,
        fps_fixed: false,
        is_recording: false,
        episodes_recorded: 0,
    },
    ...overrides,
});

const failed = (component: string): Record<string, SchemaStatusMessage> => ({
    [component]: { type: 'status', component, state: 'failed', error: 'Unplugged' },
});

const sessionError: SchemaErrorMessage = {
    type: 'error',
    component: 'environment',
    message: 'Could not load',
    error_code: 'load_failed',
    session_alive: true,
};

const setSession = (overrides: Record<string, unknown> = {}) => {
    session.current = {
        state: state(),
        statuses: {},
        error: null,
        isLoading: false,
        recordingStartedAt: null,
        send,
        ...overrides,
    };
};

const renderRecordMode = () => {
    server.use(
        http.get('/api/settings', () => HttpResponse.json(getMockedSettings())),
        http.get('/api/dataset/{dataset_id}', () =>
            HttpResponse.json(getMockedDataset({ id: DATASET_ID, default_task: 'pick the dice' }))
        )
    );

    return render(
        <RecordingCommandProvider>
            <RecordMode />
        </RecordingCommandProvider>
    );
};

describe('RecordMode', () => {
    beforeEach(() => {
        send.mockReset();
        send.mockResolvedValue({ ok: true });
    });

    it('starts an episode with the dataset task when the session is healthy', async () => {
        setSession();
        renderRecordMode();

        const start = await screen.findByRole('button', { name: 'Start episode' });
        await waitFor(() => expect(start).toBeEnabled());

        fireEvent.click(start);

        await waitFor(() => expect(send).toHaveBeenCalledWith({ command: 'start_episode', task: 'pick the dice' }));
    });

    it('starts an episode on the start hotkey when the session is healthy', async () => {
        setSession();
        renderRecordMode();
        await waitFor(() => expect(screen.getByRole('button', { name: 'Start episode' })).toBeEnabled());

        fireEvent.keyDown(window, { key: ' ' });

        await waitFor(() => expect(send).toHaveBeenCalledWith({ command: 'start_episode', task: 'pick the dice' }));
    });

    it.each([
        ['the session has an error', { error: sessionError }],
        ['the environment failed', { statuses: failed('environment') }],
        ['the dataset failed', { statuses: failed('dataset') }],
        ['the session is loading', { isLoading: true }],
        ['there is no environment', { state: state({ environment: null }) }],
    ])('cannot start an episode when %s', async (_name, overrides) => {
        setSession(overrides);
        renderRecordMode();

        const start = await screen.findByRole('button', { name: 'Start episode' });
        // Let the default task load, so only the session can be what disables it.
        await waitFor(() => expect(screen.getByRole('textbox', { name: /task/i })).toHaveValue('pick the dice'));

        expect(start).toBeDisabled();

        fireEvent.keyDown(window, { key: ' ' });

        fireEvent.click(start);

        expect(send).not.toHaveBeenCalled();
    });

    it('cannot start an episode without a task', async () => {
        setSession();
        renderRecordMode();
        await waitFor(() => expect(screen.getByRole('textbox', { name: /task/i })).toHaveValue('pick the dice'));

        fireEvent.change(screen.getByRole('textbox', { name: /task/i }), { target: { value: '' } });

        expect(screen.getByRole('button', { name: 'Start episode' })).toBeDisabled();
    });

    it('still saves and discards a recording after a session error', async () => {
        setSession({
            error: sessionError,
            state: state({
                dataset: {
                    dataset_id: DATASET_ID,
                    name: 'dice',
                    fps: 30,
                    fps_fixed: false,
                    is_recording: true,
                    episodes_recorded: 0,
                },
            }),
        });
        renderRecordMode();

        const save = await screen.findByRole('button', { name: 'Save episode' });
        expect(save).toBeEnabled();
        expect(screen.getByRole('button', { name: 'Discard' })).toBeEnabled();

        fireEvent.keyDown(window, { key: 'Enter' });

        await waitFor(() => expect(send).toHaveBeenCalledWith({ command: 'save_episode' }));
    });
});

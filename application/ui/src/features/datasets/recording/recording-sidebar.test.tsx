// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { rs } from '@rstest/core';
import { act, fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedDataset } from '../../../test-utils/mocks/mock-dataset';
import { getMockedEpisodes } from '../../../test-utils/mocks/mock-episode';
import {
    getMockedRuntimeMutation,
    getMockedRuntimeSession,
    getMockedRuntimeSessionState,
} from '../../../test-utils/mocks/mock-runtime-session';
import { getMockedSettings } from '../../../test-utils/mocks/mock-settings';
import { render } from '../../../test-utils/render';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { RecordingProvider } from './recording-provider';
import { RecordingSidebar } from './recording-sidebar';

rs.mock('../../robots/runtime-session-provider', () => ({
    useRuntimeSession: rs.fn(),
}));

const PROJECT_ID = 'project-1';
const DATASET_ID = 'dataset-1';

type SessionOverrides = Parameters<typeof getMockedRuntimeSession>[0];

interface Session {
    isRecording?: boolean;
    episodesRecorded?: number;
    startPending?: boolean;
    savePending?: boolean;
    defaultTask?: string;
}

const startEpisode = getMockedRuntimeMutation();
const saveEpisode = getMockedRuntimeMutation();
const discardEpisode = getMockedRuntimeMutation();

const buildSession = ({
    isRecording = false,
    episodesRecorded = 0,
    startPending = false,
    savePending = false,
    defaultTask = 'pick the cube',
}: Session): SessionOverrides => ({
    dataset: getMockedDataset({ id: DATASET_ID, project_id: PROJECT_ID, default_task: defaultTask }),
    state: getMockedRuntimeSessionState({ is_recording: isRecording, episodes_recorded: episodesRecorded }),
    startEpisode: { ...startEpisode, isPending: startPending },
    saveEpisode: { ...saveEpisode, isPending: savePending },
    discardEpisode,
});

const applySession = (session: Session) => {
    rs.mocked(useRuntimeSession).mockReturnValue(getMockedRuntimeSession(buildSession(session)));
};

const Harness = ({ initial }: { initial: Session }) => {
    const [, setSession] = useState(initial);

    const changeSession = (session: Session) => {
        applySession(session);
        setSession(session);
    };

    return (
        <>
            <button onClick={() => changeSession({ isRecording: true })}>simulate recording started</button>
            <button onClick={() => changeSession({ isRecording: false, episodesRecorded: 1 })}>
                simulate recording stopped
            </button>
            <RecordingProvider>
                <RecordingSidebar />
            </RecordingProvider>
        </>
    );
};

const renderSidebar = (
    initial: Session = {},
    {
        episodes = getMockedEpisodes([]),
        bindings = {},
    }: { episodes?: ReturnType<typeof getMockedEpisodes>; bindings?: Record<string, string> } = {}
) => {
    server.use(
        http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json(episodes)),
        http.get('/api/settings', () => HttpResponse.json(getMockedSettings({ hotkeys: { bindings } }))),
        http.get('/api/runtime/sessions', () => HttpResponse.json([]))
    );

    applySession(initial);

    return render(<Harness initial={initial} />, {
        route: `/projects/${PROJECT_ID}/datasets/${DATASET_ID}/record`,
        path: '/projects/:project_id/datasets/:dataset_id/record',
    });
};

describe('RecordingSidebar', () => {
    beforeEach(() => {
        rs.clearAllMocks();
        // Spectrum renders a tray-style combobox at <= 700px. jsdom's screen is 0.
        Object.defineProperty(window.screen, 'width', { configurable: true, value: 1280 });
    });

    afterEach(() => {
        rs.useRealTimers();
        Object.defineProperty(window.screen, 'width', { configurable: true, value: 0 });
    });

    describe('episode section', () => {
        it('shows the next episode number after the episodes already in the dataset and this session', async () => {
            renderSidebar(
                { episodesRecorded: 2 },
                { episodes: getMockedEpisodes([{ episode_index: 0 }, { episode_index: 1 }]) }
            );

            expect(await screen.findByText('E5')).toBeInTheDocument();
            expect(screen.getByText('Next episode')).toBeInTheDocument();
        });

        it('starts at E1 for an empty dataset', async () => {
            renderSidebar();

            expect(await screen.findByText('E1')).toBeInTheDocument();
        });

        it('labels the episode as recording while recording', async () => {
            renderSidebar({ isRecording: true });

            expect(await screen.findByText('Recording')).toBeInTheDocument();
            expect(screen.queryByText('Next episode')).not.toBeInTheDocument();
        });

        it('labels the episode as saving while the save request is pending', async () => {
            renderSidebar({ isRecording: true, savePending: true });

            expect(await screen.findByText('Saving…')).toBeInTheDocument();
        });
    });

    describe('task picker', () => {
        it('starts with the dataset default task', async () => {
            renderSidebar({ defaultTask: 'stack the blocks' });

            expect(await screen.findByRole('combobox', { name: /task/i })).toHaveValue('stack the blocks');
        });

        it('offers the default task and the unique tasks of recorded episodes', async () => {
            const user = userEvent.setup();
            renderSidebar(
                { defaultTask: 'pick the cube' },
                {
                    episodes: getMockedEpisodes([
                        { episode_index: 0, tasks: ['pick the cube', 'stack the blocks'] },
                        { episode_index: 1, tasks: ['stack the blocks', 'open the drawer'] },
                    ]),
                }
            );

            await user.click(await screen.findByRole('button', { name: /show suggestions/i }));

            const options = await screen.findAllByRole('option');
            expect(options.map((option) => option.textContent)).toEqual([
                'pick the cube',
                'stack the blocks',
                'open the drawer',
            ]);
        });

        it('cannot be edited while recording', async () => {
            renderSidebar({ isRecording: true });

            expect(await screen.findByRole('combobox', { name: /task/i })).toHaveAttribute('readonly');
        });

        it('blocks starting an episode and explains why when the task is empty', async () => {
            const user = userEvent.setup();
            renderSidebar();

            await user.clear(await screen.findByRole('combobox', { name: /task/i }));
            await user.keyboard('{Escape}');

            expect(screen.getByRole('button', { name: /start episode/i })).toBeDisabled();
            expect(await screen.findByText('A task is required in order to record.')).toBeInTheDocument();
        });
    });

    describe('idle controls', () => {
        it('starts an episode with the current task', async () => {
            const user = userEvent.setup();
            renderSidebar({ defaultTask: 'pick the cube' });

            const task = await screen.findByRole('combobox', { name: /task/i });
            await user.clear(task);
            await user.type(task, 'wipe the table');
            await user.keyboard('{Escape}');
            await user.click(screen.getByRole('button', { name: /start episode/i }));

            expect(startEpisode.mutate).toHaveBeenCalledTimes(1);
            expect(startEpisode.mutate).toHaveBeenCalledWith('wipe the table', expect.anything());
        });

        it('offers only start, not save or discard', async () => {
            renderSidebar();

            expect(await screen.findByRole('button', { name: /start episode/i })).toBeEnabled();
            expect(screen.queryByRole('button', { name: /save episode/i })).not.toBeInTheDocument();
            expect(screen.queryByRole('button', { name: /discard/i })).not.toBeInTheDocument();
        });

        it('shows the effective hotkeys in the helper text and next to the timer', async () => {
            renderSidebar();

            expect(
                await screen.findByText('After the episode: Save (Enter) or Discard (Backspace).')
            ).toBeInTheDocument();
            expect(screen.getByText('Space')).toBeInTheDocument();
        });

        it('reflects custom hotkey bindings', async () => {
            renderSidebar(
                {},
                {
                    bindings: {
                        'recording.start_episode': 'S',
                        'recording.accept_episode': 'Shift+ArrowRight',
                        'recording.discard_episode': 'D',
                    },
                }
            );

            expect(await screen.findByText('After the episode: Save (Shift+→) or Discard (D).')).toBeInTheDocument();
            expect(screen.getByText('S')).toBeInTheDocument();
        });
    });

    describe('recording controls', () => {
        it('offers save and discard instead of start', async () => {
            renderSidebar({ isRecording: true });

            expect(await screen.findByRole('button', { name: /save episode/i })).toBeEnabled();
            expect(screen.getByRole('button', { name: /discard/i })).toBeEnabled();
            expect(screen.queryByRole('button', { name: /start episode/i })).not.toBeInTheDocument();
        });

        it('saves the episode', async () => {
            const user = userEvent.setup();
            renderSidebar({ isRecording: true });

            await user.click(await screen.findByRole('button', { name: /save episode/i }));

            expect(saveEpisode.mutate).toHaveBeenCalledTimes(1);
            expect(discardEpisode.mutate).not.toHaveBeenCalled();
        });

        it('discards the episode', async () => {
            const user = userEvent.setup();
            renderSidebar({ isRecording: true });

            await user.click(await screen.findByRole('button', { name: /discard/i }));

            expect(discardEpisode.mutate).toHaveBeenCalledTimes(1);
            expect(saveEpisode.mutate).not.toHaveBeenCalled();
        });

        it('disables discard while saving', async () => {
            renderSidebar({ isRecording: true, savePending: true });

            expect(await screen.findByRole('button', { name: /discard/i })).toBeDisabled();
        });
    });

    describe('hotkeys', () => {
        it('starts an episode with Space', async () => {
            renderSidebar({ defaultTask: 'pick the cube' });
            await screen.findByRole('button', { name: /start episode/i });

            fireEvent.keyDown(document.body, { key: ' ' });

            expect(startEpisode.mutate).toHaveBeenCalledWith('pick the cube', expect.anything());
        });

        it('does not start with Space when the task is empty', async () => {
            const user = userEvent.setup();
            renderSidebar();
            await user.clear(await screen.findByRole('combobox', { name: /task/i }));
            await user.tab();

            fireEvent.keyDown(document.body, { key: ' ' });

            expect(startEpisode.mutate).not.toHaveBeenCalled();
        });

        it('does not start with Space while a start request is pending', async () => {
            renderSidebar({ startPending: true });
            await screen.findByRole('combobox', { name: /task/i });

            fireEvent.keyDown(document.body, { key: ' ' });

            expect(startEpisode.mutate).not.toHaveBeenCalled();
        });

        it('saves with Enter while recording', async () => {
            renderSidebar({ isRecording: true });
            await screen.findByRole('button', { name: /save episode/i });

            fireEvent.keyDown(document.body, { key: 'Enter' });

            expect(saveEpisode.mutate).toHaveBeenCalledTimes(1);
            expect(discardEpisode.mutate).not.toHaveBeenCalled();
        });

        it('discards with Backspace while recording', async () => {
            renderSidebar({ isRecording: true });
            await screen.findByRole('button', { name: /save episode/i });

            fireEvent.keyDown(document.body, { key: 'Backspace' });

            expect(discardEpisode.mutate).toHaveBeenCalledTimes(1);
            expect(saveEpisode.mutate).not.toHaveBeenCalled();
        });

        it('does not save or discard when not recording', async () => {
            renderSidebar();
            await screen.findByRole('button', { name: /start episode/i });

            fireEvent.keyDown(document.body, { key: 'Enter' });
            fireEvent.keyDown(document.body, { key: 'Backspace' });

            expect(saveEpisode.mutate).not.toHaveBeenCalled();
            expect(discardEpisode.mutate).not.toHaveBeenCalled();
        });

        it('does not save or discard while the save request is pending', async () => {
            renderSidebar({ isRecording: true, savePending: true });
            await screen.findByText('E1');

            fireEvent.keyDown(document.body, { key: 'Enter' });
            fireEvent.keyDown(document.body, { key: 'Backspace' });

            expect(saveEpisode.mutate).not.toHaveBeenCalled();
            expect(discardEpisode.mutate).not.toHaveBeenCalled();
        });

        it('edits the task text instead of discarding when Backspace is pressed in the task field', async () => {
            const user = userEvent.setup();
            renderSidebar({ defaultTask: 'pick the cube' });

            const task = await screen.findByRole('combobox', { name: /task/i });
            await user.click(task);
            await user.keyboard('{Backspace}');

            expect(task).toHaveValue('pick the cub');
            expect(discardEpisode.mutate).not.toHaveBeenCalled();
        });

        it('does not start when Space is pressed on a focused button', async () => {
            renderSidebar();
            const robotsOrOtherButton = await screen.findByRole('button', { name: /start episode/i });

            fireEvent.keyDown(robotsOrOtherButton, { key: ' ' });

            expect(startEpisode.mutate).not.toHaveBeenCalled();
        });

        it('keeps using the overridden bindings', async () => {
            renderSidebar({ isRecording: true }, { bindings: { 'recording.accept_episode': 'S' } });
            await screen.findByRole('button', { name: /save episode/i });

            fireEvent.keyDown(document.body, { key: 'Enter' });
            expect(saveEpisode.mutate).not.toHaveBeenCalled();

            fireEvent.keyDown(document.body, { key: 's' });
            expect(saveEpisode.mutate).toHaveBeenCalledTimes(1);
        });
    });

    describe('timer', () => {
        it('shows 00:00 when idle', async () => {
            renderSidebar();

            expect(await screen.findByText('00:00')).toBeInTheDocument();
        });

        it('counts the elapsed time while recording', async () => {
            renderSidebar();
            await screen.findByText('00:00');
            rs.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] });

            fireEvent.click(screen.getByRole('button', { name: 'simulate recording started' }));
            await act(async () => {
                rs.advanceTimersByTime(3_000);
            });

            expect(screen.getByText('00:03')).toBeInTheDocument();
        });

        it('keeps counting when the recording was already running when the page opened', async () => {
            rs.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] });
            renderSidebar({ isRecording: true });
            await screen.findByRole('button', { name: /save episode/i });

            await act(async () => {
                rs.advanceTimersByTime(5_000);
            });

            expect(screen.getByText('00:05')).toBeInTheDocument();
        });

        it('resets to 00:00 once the recording stops', async () => {
            renderSidebar();
            await screen.findByText('00:00');
            rs.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] });
            fireEvent.click(screen.getByRole('button', { name: 'simulate recording started' }));
            await act(async () => {
                rs.advanceTimersByTime(4_000);
            });
            expect(screen.getByText('00:04')).toBeInTheDocument();

            fireEvent.click(screen.getByRole('button', { name: 'simulate recording stopped' }));

            expect(screen.getByText('00:00')).toBeInTheDocument();
        });
    });
});

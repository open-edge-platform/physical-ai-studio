// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useLocation } from 'react-router';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedDataset } from '../../../test-utils/mocks/mock-dataset';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import { getMockedEpisodes } from '../../../test-utils/mocks/mock-episode';
import { getMockedRuntimeSession, getMockedRuntimeSessionState } from '../../../test-utils/mocks/mock-runtime-session';
import { getMockedSettings } from '../../../test-utils/mocks/mock-settings';
import { render } from '../../../test-utils/render';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { RecordingProvider } from './recording-provider';
import { RecordingTopBar } from './recording-top-bar';
import { TaskPicker } from './task-picker';

rs.mock('../../robots/runtime-session-provider', () => ({
    useRuntimeSession: rs.fn(),
}));

const PROJECT_ID = 'project-1';
const DATASET_ID = 'dataset-1';
const DATASET_URL = `/projects/${PROJECT_ID}/datasets/${DATASET_ID}`;

const LocationDisplay = () => <output aria-label='location'>{useLocation().pathname}</output>;

interface Options {
    isRecording?: boolean;
    episodesInDataset?: number;
    episodesRecorded?: number;
}

const renderTopBar = ({ isRecording = false, episodesInDataset = 0, episodesRecorded = 0 }: Options = {}) => {
    rs.mocked(useRuntimeSession).mockReturnValue(
        getMockedRuntimeSession({
            dataset: getMockedDataset({
                id: DATASET_ID,
                project_id: PROJECT_ID,
                name: 'Kitchen dataset',
                default_task: 'pick the cube',
            }),
            environment: getMockedEnvironment({ name: 'Kitchen setup' }),
            state: getMockedRuntimeSessionState({ is_recording: isRecording, episodes_recorded: episodesRecorded }),
        })
    );
    server.use(
        http.get('/api/dataset/{dataset_id}/episodes', () =>
            HttpResponse.json(
                getMockedEpisodes(Array.from({ length: episodesInDataset }, (_, i) => ({ episode_index: i })))
            )
        ),
        http.get('/api/settings', () => HttpResponse.json(getMockedSettings()))
    );

    return render(
        <RecordingProvider>
            <RecordingTopBar />
            <TaskPicker />
            <LocationDisplay />
        </RecordingProvider>,
        { route: `${DATASET_URL}/record`, path: '*' }
    );
};

describe('RecordingTopBar', () => {
    beforeEach(() => {
        rs.clearAllMocks();
        // Spectrum renders a tray-style combobox at <= 700px. jsdom's screen is 0.
        Object.defineProperty(window.screen, 'width', { configurable: true, value: 1280 });
    });

    afterEach(() => {
        Object.defineProperty(window.screen, 'width', { configurable: true, value: 0 });
    });

    it('shows the title with the dataset, task and environment in the subtitle', async () => {
        renderTopBar();

        const link = await screen.findByRole('link');
        expect(link).toHaveTextContent('Recording episode');
        expect(link).toHaveTextContent('Kitchen dataset');
        expect(link).toHaveTextContent('pick the cube');
        expect(link).toHaveTextContent('Kitchen setup');
    });

    it('updates the subtitle live as the task is edited', async () => {
        const user = userEvent.setup();
        renderTopBar();

        const task = await screen.findByRole('combobox', { name: /task/i });
        await user.clear(task);
        await user.type(task, 'wipe the table');
        await user.keyboard('{Escape}');

        expect(screen.getByRole('link')).toHaveTextContent('wipe the table');
        expect(screen.getByRole('link')).not.toHaveTextContent('pick the cube');
    });

    describe('episode counts', () => {
        it('hides the counts for an empty dataset', async () => {
            renderTopBar();
            await screen.findByRole('link');

            expect(screen.queryByText(/in dataset/)).not.toBeInTheDocument();
        });

        it('shows the dataset total without a session part when nothing was recorded yet', async () => {
            renderTopBar({ episodesInDataset: 12 });

            expect(await screen.findByText('12 episodes in dataset')).toBeInTheDocument();
        });

        it('adds this session recordings to the dataset total', async () => {
            renderTopBar({ episodesInDataset: 12, episodesRecorded: 3 });

            expect(await screen.findByText('15 episodes in dataset · 3 this session')).toBeInTheDocument();
        });
    });

    describe('leaving the page', () => {
        it('links back to the dataset page', async () => {
            renderTopBar();

            expect(await screen.findByRole('link')).toHaveAttribute('href', DATASET_URL);
        });

        it('navigates to the dataset page when Done is pressed', async () => {
            const user = userEvent.setup();
            renderTopBar();

            await user.click(await screen.findByRole('button', { name: 'Done' }));

            expect(screen.getByLabelText('location')).toHaveTextContent(DATASET_URL);
        });

        it('disables Done and the back link while recording', async () => {
            renderTopBar({ isRecording: true });

            expect(await screen.findByRole('button', { name: 'Done' })).toBeDisabled();
            expect(screen.queryByRole('link')).not.toBeInTheDocument();
        });

        it('does not navigate when the back area is pressed while recording', async () => {
            const user = userEvent.setup();
            renderTopBar({ isRecording: true });

            await user.click(await screen.findByText('Recording episode'));

            expect(screen.getByLabelText('location')).toHaveTextContent(`${DATASET_URL}/record`);
        });
    });
});

import { describe, expect, it } from '@rstest/core';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { SchemaSessionDataset } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedEpisodes } from '../../../test-utils/mocks/mock-episode';
import { render } from '../../../test-utils/render';
import { getEpisodeTagLabel, RecordingEpisodeTag } from './recording-episode-tag';

const DATASET_ID = '610d64d2-7133-45e9-b979-6a4ef14a589c';

const getDataset = (overrides: Partial<SchemaSessionDataset> = {}): SchemaSessionDataset => ({
    dataset_id: DATASET_ID,
    name: 'dice',
    fps: 30,
    fps_fixed: false,
    is_recording: false,
    episodes_recorded: 1,
    ...overrides,
});

const renderTag = (dataset: SchemaSessionDataset, pendingCommand: 'save_episode' | 'discard_episode' | null = null) => {
    server.use(
        http.get('/api/dataset/{dataset_id}/episodes', () =>
            HttpResponse.json(getMockedEpisodes([{ episode_index: 0 }, { episode_index: 1 }, { episode_index: 2 }]))
        )
    );

    return render(<RecordingEpisodeTag dataset={dataset} pendingCommand={pendingCommand} />);
};

describe('getEpisodeTagLabel', () => {
    it('says next when idle', () => {
        expect(getEpisodeTagLabel(null, false)).toBe('next');
    });

    it('says recording while recording', () => {
        expect(getEpisodeTagLabel(null, true)).toBe('recording');
    });

    it('says saving while a save is pending, even while recording', () => {
        expect(getEpisodeTagLabel('save_episode', true)).toBe('saving');
    });

    it('says discarding while a discard is pending', () => {
        expect(getEpisodeTagLabel('discard_episode', true)).toBe('discarding');
    });
});

describe('RecordingEpisodeTag', () => {
    it('numbers the next episode after the saved and the recorded ones', async () => {
        renderTag(getDataset());

        expect(await screen.findByText('E5')).toBeInTheDocument();
        expect(screen.getByText('Next episode')).toBeInTheDocument();
    });

    it('shows Recording while recording', async () => {
        renderTag(getDataset({ is_recording: true }));

        expect(await screen.findByText('Recording')).toBeInTheDocument();
        expect(screen.queryByText('Next episode')).not.toBeInTheDocument();
    });

    it('shows Saving… and Discarding… while those commands are pending', async () => {
        const { unmount } = renderTag(getDataset({ is_recording: true }), 'save_episode');
        expect(await screen.findByText('Saving…')).toBeInTheDocument();
        unmount();

        renderTag(getDataset({ is_recording: true }), 'discard_episode');
        expect(await screen.findByText('Discarding…')).toBeInTheDocument();
    });
});

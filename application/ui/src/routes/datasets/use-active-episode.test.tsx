// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { getMockedDataset } from '../../test-utils/mocks/mock-dataset';
import { getMockedEpisodes } from '../../test-utils/mocks/mock-episode';
import { renderHook } from '../../test-utils/render';
import { DatasetProvider } from './dataset-provider';
import { useActiveEpisode } from './use-active-episode';

const PROJECT_ID = 'project-1';
const DATASET_ID = 'dataset-1';

const EPISODES = getMockedEpisodes([{ episode_index: 5 }, { episode_index: 7 }]);

const wrapper = ({ children }: { children: ReactNode }) => (
    <DatasetProvider dataset_id={DATASET_ID}>{children}</DatasetProvider>
);

const setupMocks = (episodes = EPISODES) => {
    server.use(
        http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset({ id: DATASET_ID }))),
        http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json(episodes))
    );
};

describe('useActiveEpisode', () => {
    it('redirects to the first episode when no episode index is in the URL', async () => {
        setupMocks();

        const { result } = renderHook(() => useActiveEpisode(), {
            route: `/projects/${PROJECT_ID}/datasets/${DATASET_ID}`,
            path: '/projects/:project_id/datasets/:dataset_id/episodes?/:episode_index?',
            wrapper,
        });

        await waitFor(() => expect(result.current[0]).toBe(5));
    });

    it('redirects to the first episode when the URL has an invalid episode index', async () => {
        setupMocks();

        const { result } = renderHook(() => useActiveEpisode(), {
            route: `/projects/${PROJECT_ID}/datasets/${DATASET_ID}/episodes/999`,
            path: '/projects/:project_id/datasets/:dataset_id/episodes?/:episode_index?',
            wrapper,
        });

        await waitFor(() => expect(result.current[0]).toBe(5));
    });

    it('keeps a valid episode index from the URL', async () => {
        setupMocks();

        const { result } = renderHook(() => useActiveEpisode(), {
            route: `/projects/${PROJECT_ID}/datasets/${DATASET_ID}/episodes/7`,
            path: '/projects/:project_id/datasets/:dataset_id/episodes?/:episode_index?',
            wrapper,
        });

        await waitFor(() => expect(result.current[0]).toBe(7));
    });

    it('returns null and does not redirect when the dataset has no episodes', async () => {
        setupMocks(getMockedEpisodes([]));

        const { result } = renderHook(() => useActiveEpisode(), {
            route: `/projects/${PROJECT_ID}/datasets/${DATASET_ID}`,
            path: '/projects/:project_id/datasets/:dataset_id/episodes?/:episode_index?',
            wrapper,
        });

        await waitFor(() => expect(result.current[0]).toBeNull());
    });
});

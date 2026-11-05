// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { getEpisodesQueryKey } from '../../features/datasets/episodes/episodes-query-key';
import { setPendingRecordedEpisodes } from '../../features/datasets/pending-recorded-episodes';
import { server } from '../../msw-node-setup';
import { createQueryClient } from '../../query-client/query-client';
import { getMockedDataset } from '../../test-utils/mocks/mock-dataset';
import { getMockedEpisodes } from '../../test-utils/mocks/mock-episode';
import { render } from '../../test-utils/render';
import { DatasetProvider, useDataset } from './dataset-provider';

const DATASET_ID = 'dataset-1';
const OTHER_DATASET_ID = 'dataset-2';
const EPISODES_QUERY_KEY = getEpisodesQueryKey(DATASET_ID);

const SelectedEpisodes = () => {
    const { selectedEpisodes, setSelectedEpisodes } = useDataset();

    return (
        <>
            <button onClick={() => setSelectedEpisodes([1])}>Select episode</button>
            <output data-testid='selected-episodes'>{selectedEpisodes.join(',')}</output>
        </>
    );
};

const DatasetProviderHarness = () => {
    const [datasetId, setDatasetId] = useState(DATASET_ID);

    return (
        <>
            <button onClick={() => setDatasetId(OTHER_DATASET_ID)}>Change dataset</button>
            <DatasetProvider dataset_id={datasetId}>
                <SelectedEpisodes />
            </DatasetProvider>
        </>
    );
};

describe('DatasetProvider', () => {
    afterEach(() => window.sessionStorage.clear());

    it('revalidates cached episodes when it mounts', async () => {
        let episodesRequests = 0;
        server.use(
            http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset({ id: DATASET_ID }))),
            http.get('/api/dataset/{dataset_id}/episodes', () => {
                episodesRequests += 1;
                return HttpResponse.json(getMockedEpisodes());
            })
        );
        const queryClient = createQueryClient();
        queryClient.setQueryData(EPISODES_QUERY_KEY, getMockedEpisodes());

        render(
            <DatasetProvider dataset_id={DATASET_ID}>
                <div>Dataset content</div>
            </DatasetProvider>,
            { queryClient }
        );

        await waitFor(() => expect(episodesRequests).toBe(1));
    });

    it('polls until the expected recorded episode count is available', async () => {
        let episodesRequests = 0;
        setPendingRecordedEpisodes(DATASET_ID, 3);
        server.use(
            http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset({ id: DATASET_ID }))),
            http.get('/api/dataset/{dataset_id}/episodes', () => {
                episodesRequests += 1;
                return HttpResponse.json(
                    getMockedEpisodes(
                        episodesRequests === 1
                            ? [{ episode_index: 0 }]
                            : [{ episode_index: 0 }, { episode_index: 1 }, { episode_index: 2 }]
                    )
                );
            })
        );

        render(
            <DatasetProvider dataset_id={DATASET_ID}>
                <div>Dataset content</div>
            </DatasetProvider>
        );

        await waitFor(() => expect(episodesRequests).toBe(2), { timeout: 2000 });
        await new Promise((resolve) => window.setTimeout(resolve, 1100));
        expect(episodesRequests).toBe(2);
    });

    it('clears selected episodes when the dataset changes', async () => {
        server.use(
            http.get('/api/dataset/{dataset_id}', () => HttpResponse.json(getMockedDataset({ id: DATASET_ID }))),
            http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json(getMockedEpisodes()))
        );

        render(<DatasetProviderHarness />);

        fireEvent.click(await screen.findByRole('button', { name: 'Select episode' }));
        expect(screen.getByTestId('selected-episodes')).toHaveTextContent('1');

        fireEvent.click(screen.getByRole('button', { name: 'Change dataset' }));
        await waitFor(() => expect(screen.getByTestId('selected-episodes')).toBeEmptyDOMElement());
    });
});

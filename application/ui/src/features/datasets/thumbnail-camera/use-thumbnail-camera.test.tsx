// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedEpisodes } from '../../../test-utils/mocks/mock-episode';
import { getMockedEpisodeDetail } from '../../../test-utils/mocks/mock-episode-detail';
import { renderHook } from '../../../test-utils/render';
import { useThumbnailCamera } from './use-thumbnail-camera';

const TOP = 'observation.images.top';
const WRIST = 'observation.images.wrist';
const STORAGE_PREFIX = 'physicalai:thumbnailCamera:';

const mockDatasetCameras = (datasetId: string, videoKeys: string[]) => {
    server.use(
        http.get('/api/dataset/{dataset_id}/episodes', ({ params }) =>
            params.dataset_id === datasetId ? HttpResponse.json(getMockedEpisodes()) : undefined
        ),
        http.get('/api/dataset/{dataset_id}/episodes/{episode_index}', ({ params }) =>
            params.dataset_id === datasetId
                ? HttpResponse.json(
                      getMockedEpisodeDetail({
                          videos: Object.fromEntries(
                              videoKeys.map((key) => [key, { start: 0, end: 3, path: `${key}.mp4` }])
                          ),
                      })
                  )
                : undefined
        )
    );
};

const mockEmptyDataset = () => {
    server.use(http.get('/api/dataset/{dataset_id}/episodes', () => HttpResponse.json([])));
};

describe('useThumbnailCamera', () => {
    afterEach(() => {
        localStorage.clear();
    });

    it('returns the stored key on first render, before cameras have loaded', () => {
        localStorage.setItem(`${STORAGE_PREFIX}dataset-1`, WRIST);
        mockDatasetCameras('dataset-1', [TOP, WRIST]);

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        expect(result.current.cameras).toEqual([]);
        expect(result.current.thumbnailCamera).toBe(WRIST);
        expect(result.current.selectedCameraKey).toBe(WRIST);
    });

    it('leaves thumbnailCamera undefined and selects the first camera when nothing is stored', async () => {
        mockDatasetCameras('dataset-1', [TOP, WRIST]);

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        await waitFor(() => expect(result.current.cameras).toHaveLength(2));

        expect(result.current.thumbnailCamera).toBeUndefined();
        expect(result.current.selectedCameraKey).toBe(TOP);
    });

    it('keeps a stored key that is among the loaded cameras', async () => {
        localStorage.setItem(`${STORAGE_PREFIX}dataset-1`, WRIST);
        mockDatasetCameras('dataset-1', [TOP, WRIST]);

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        await waitFor(() => expect(result.current.cameras).toHaveLength(2));

        expect(result.current.thumbnailCamera).toBe(WRIST);
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBe(WRIST);
    });

    it('clears a stored key that is not among the loaded cameras', async () => {
        localStorage.setItem(`${STORAGE_PREFIX}dataset-1`, 'observation.images.gone');
        mockDatasetCameras('dataset-1', [TOP, WRIST]);

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        await waitFor(() => expect(result.current.cameras).toHaveLength(2));

        expect(result.current.thumbnailCamera).toBeUndefined();
        expect(result.current.selectedCameraKey).toBe(TOP);
        await waitFor(() => expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBeNull());
    });

    it('does not clear a stored key while cameras are still loading', async () => {
        localStorage.setItem(`${STORAGE_PREFIX}dataset-1`, 'observation.images.gone');
        let resolveEpisodes: () => void = () => {};
        const episodesGate = new Promise<void>((resolve) => {
            resolveEpisodes = resolve;
        });
        server.use(
            http.get('/api/dataset/{dataset_id}/episodes', async () => {
                await episodesGate;
                return HttpResponse.json(getMockedEpisodes());
            }),
            http.get('/api/dataset/{dataset_id}/episodes/{episode_index}', () =>
                HttpResponse.json(getMockedEpisodeDetail({ videos: {} }))
            )
        );

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        // Give the effects a chance to run while the request is pending.
        await act(async () => {});

        expect(result.current.thumbnailCamera).toBe('observation.images.gone');
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBe('observation.images.gone');

        resolveEpisodes();
    });

    it('does not clear a stored key when the dataset has no episodes', async () => {
        localStorage.setItem(`${STORAGE_PREFIX}dataset-1`, WRIST);
        mockEmptyDataset();

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        await waitFor(() => expect(result.current.cameras).toEqual([]));
        await act(async () => {});

        expect(result.current.thumbnailCamera).toBe(WRIST);
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBe(WRIST);
    });

    it('persists the key and updates thumbnailCamera when one is selected', async () => {
        mockDatasetCameras('dataset-1', [TOP, WRIST]);

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));
        await waitFor(() => expect(result.current.cameras).toHaveLength(2));

        act(() => {
            result.current.changeThumbnailCamera(WRIST);
        });

        expect(result.current.thumbnailCamera).toBe(WRIST);
        expect(result.current.selectedCameraKey).toBe(WRIST);
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBe(WRIST);
    });

    it('re-reads storage for the new dataset when datasetId changes without remounting', async () => {
        localStorage.setItem(`${STORAGE_PREFIX}dataset-1`, TOP);
        localStorage.setItem(`${STORAGE_PREFIX}dataset-2`, WRIST);
        mockDatasetCameras('dataset-1', [TOP, WRIST]);
        mockDatasetCameras('dataset-2', [TOP, WRIST]);

        let datasetId = 'dataset-1';
        const { result, rerender } = renderHook(() => useThumbnailCamera(datasetId));

        expect(result.current.thumbnailCamera).toBe(TOP);

        datasetId = 'dataset-2';
        rerender();

        expect(result.current.thumbnailCamera).toBe(WRIST);

        datasetId = 'dataset-3';
        rerender();

        expect(result.current.thumbnailCamera).toBeUndefined();

        datasetId = 'dataset-1';
        rerender();

        expect(result.current.thumbnailCamera).toBe(TOP);
    });

    it('writes a selection to the current dataset after datasetId changes', async () => {
        mockDatasetCameras('dataset-1', [TOP, WRIST]);
        mockDatasetCameras('dataset-2', [TOP, WRIST]);

        let datasetId = 'dataset-1';
        const { result, rerender } = renderHook(() => useThumbnailCamera(datasetId));

        datasetId = 'dataset-2';
        rerender();

        act(() => {
            result.current.changeThumbnailCamera(WRIST);
        });

        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-2`)).toBe(WRIST);
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBeNull();
        expect(result.current.thumbnailCamera).toBe(WRIST);
    });
});

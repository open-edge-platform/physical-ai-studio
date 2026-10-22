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

    it('uses a stored key before cameras have loaded and keeps it once they have', async () => {
        localStorage.setItem(`${STORAGE_PREFIX}dataset-1`, WRIST);
        mockDatasetCameras('dataset-1', [TOP, WRIST]);

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        expect(result.current.cameras).toEqual([]);
        expect(result.current.thumbnailCamera).toBe(WRIST);

        await waitFor(() => expect(result.current.cameras).toHaveLength(2));

        expect(result.current.thumbnailCamera).toBe(WRIST);
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBe(WRIST);
    });

    it('falls back to the first camera when nothing is stored and persists a selection', async () => {
        mockDatasetCameras('dataset-1', [TOP, WRIST]);

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        expect(result.current.isLoading).toBe(true);
        expect(result.current.thumbnailCamera).toBeUndefined();

        await waitFor(() => expect(result.current.cameras).toHaveLength(2));

        expect(result.current.isLoading).toBe(false);
        expect(result.current.thumbnailCamera).toBe(TOP);

        act(() => {
            result.current.changeThumbnailCamera(WRIST);
        });

        expect(result.current.thumbnailCamera).toBe(WRIST);
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBe(WRIST);
    });

    it('stays undefined when the dataset has no cameras', async () => {
        mockDatasetCameras('dataset-1', []);

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        await waitFor(() => expect(result.current.isLoading).toBe(false));

        expect(result.current.cameras).toEqual([]);
        expect(result.current.thumbnailCamera).toBeUndefined();
    });

    it('keeps a stale key while cameras load, then clears it and falls back to the first camera', async () => {
        localStorage.setItem(`${STORAGE_PREFIX}dataset-1`, 'observation.images.gone');
        const { promise: episodesGate, resolve: resolveEpisodes } = Promise.withResolvers();

        mockDatasetCameras('dataset-1', [TOP, WRIST]);
        server.use(
            http.get('/api/dataset/{dataset_id}/episodes', async () => {
                await episodesGate;
                return HttpResponse.json(getMockedEpisodes());
            })
        );

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        expect(result.current.thumbnailCamera).toBe('observation.images.gone');
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBe('observation.images.gone');

        resolveEpisodes(0);

        await waitFor(() => expect(result.current.cameras).toHaveLength(2));

        expect(result.current.thumbnailCamera).toBe(TOP);
        await waitFor(() => expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBeNull());
    });

    it('does not clear a stored key when the dataset has no episodes', async () => {
        localStorage.setItem(`${STORAGE_PREFIX}dataset-1`, WRIST);
        mockEmptyDataset();

        const { result } = renderHook(() => useThumbnailCamera('dataset-1'));

        await waitFor(() => expect(result.current.cameras).toEqual([]));

        expect(result.current.thumbnailCamera).toBe(WRIST);
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBe(WRIST);
    });

    it('re-reads storage when datasetId changes without remounting and writes selections to the new dataset', () => {
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

        act(() => {
            result.current.changeThumbnailCamera(WRIST);
        });

        expect(result.current.thumbnailCamera).toBe(WRIST);
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-3`)).toBe(WRIST);
        expect(localStorage.getItem(`${STORAGE_PREFIX}dataset-1`)).toBe(TOP);

        datasetId = 'dataset-1';
        rerender();

        expect(result.current.thumbnailCamera).toBe(TOP);
    });
});

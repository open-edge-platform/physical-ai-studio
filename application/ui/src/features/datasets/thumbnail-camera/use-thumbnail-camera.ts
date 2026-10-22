// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from 'react';

import { useDatasetCameras, type DatasetCamera } from '../use-dataset-cameras';
import { clearThumbnailCamera, getThumbnailCamera, setThumbnailCamera } from './thumbnail-camera-storage';

interface ThumbnailCameraState {
    datasetId: string;
    cameraKey: string | undefined;
}

interface UseThumbnailCameraResult {
    cameras: DatasetCamera[];
    /** The stored camera key, or `undefined` to let the backend pick its default. Used for thumbnail URLs. */
    thumbnailCamera: string | undefined;
    /** The camera shown as selected in the menu: the stored key, falling back to the first camera. */
    selectedCameraKey: string | undefined;
    changeThumbnailCamera: (cameraKey: string) => void;
}

export const useThumbnailCamera = (datasetId: string): UseThumbnailCameraResult => {
    const { cameras, isLoading, isEmpty } = useDatasetCameras(datasetId);

    const [state, setState] = useState<ThumbnailCameraState>(() => ({
        datasetId,
        cameraKey: getThumbnailCamera(datasetId),
    }));

    if (state.datasetId !== datasetId) {
        setState({ datasetId, cameraKey: getThumbnailCamera(datasetId) });
    }

    const camerasLoaded = !isLoading && !isEmpty && cameras.length > 0;
    const isStale =
        camerasLoaded && state.cameraKey !== undefined && !cameras.some(({ key }) => key === state.cameraKey);

    useEffect(() => {
        if (!isStale) {
            return;
        }

        clearThumbnailCamera(datasetId);
        setState({ datasetId, cameraKey: undefined });
    }, [isStale, datasetId]);

    const thumbnailCamera = isStale ? undefined : state.cameraKey;
    console.log({ thumbnailCamera });

    const changeThumbnailCamera = useCallback(
        (cameraKey: string) => {
            setThumbnailCamera(datasetId, cameraKey);
            setState({ datasetId, cameraKey });
        },
        [datasetId]
    );

    return {
        cameras,
        thumbnailCamera,
        selectedCameraKey: thumbnailCamera ?? cameras.at(0)?.key,
        changeThumbnailCamera,
    };
};

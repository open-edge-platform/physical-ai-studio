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
    thumbnailCamera: string | undefined;
    isLoading: boolean;
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

    const storedCamera = isStale ? undefined : state.cameraKey;
    const thumbnailCamera = storedCamera ?? cameras.at(0)?.key;

    const changeThumbnailCamera = useCallback(
        (cameraKey: string) => {
            setThumbnailCamera(datasetId, cameraKey);
            setState({ datasetId, cameraKey });
        },
        [datasetId]
    );

    return {
        cameras,
        isLoading,
        thumbnailCamera,
        changeThumbnailCamera,
    };
};

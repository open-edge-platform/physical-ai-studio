// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const STORAGE_KEY_PREFIX = 'physicalai:thumbnailCamera:';

const getStorageKey = (datasetId: string): string => `${STORAGE_KEY_PREFIX}${datasetId}`;

export const getThumbnailCamera = (datasetId: string): string | undefined => {
    if (typeof window === 'undefined') {
        return undefined;
    }

    try {
        const cameraKey = window.localStorage.getItem(getStorageKey(datasetId));
        return cameraKey || undefined;
    } catch {
        return undefined;
    }
};

export const setThumbnailCamera = (datasetId: string, cameraKey: string): void => {
    if (typeof window === 'undefined') {
        return;
    }

    try {
        window.localStorage.setItem(getStorageKey(datasetId), cameraKey);
    } catch {}
};

export const clearThumbnailCamera = (datasetId: string): void => {
    if (typeof window === 'undefined') {
        return;
    }

    try {
        window.localStorage.removeItem(getStorageKey(datasetId));
    } catch {}
};

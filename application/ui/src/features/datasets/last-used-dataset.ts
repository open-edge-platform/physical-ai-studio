// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const STORAGE_KEY_PREFIX = 'physicalai:lastDataset:';

const getStorageKey = (projectId: string): string => `${STORAGE_KEY_PREFIX}${projectId}`;

export const getLastUsedDatasetId = (projectId: string): string | undefined => {
    if (typeof window === 'undefined') {
        return undefined;
    }

    try {
        const datasetId = window.localStorage.getItem(getStorageKey(projectId));
        return datasetId || undefined;
    } catch {
        return undefined;
    }
};

export const setLastUsedDatasetId = (projectId: string, datasetId: string): void => {
    if (typeof window === 'undefined') {
        return;
    }

    try {
        window.localStorage.setItem(getStorageKey(projectId), datasetId);
    } catch {}
};

export const clearLastUsedDatasetId = (projectId: string): void => {
    if (typeof window === 'undefined') {
        return;
    }

    try {
        window.localStorage.removeItem(getStorageKey(projectId));
    } catch {}
};

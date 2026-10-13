// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from '../../test-utils/render';
import {
    clearLastUsedDatasetId,
    getLastUsedDatasetId,
    setLastUsedDatasetId,
    useRecordLastUsedDataset,
} from './last-used-dataset';

describe('last-used-dataset storage', () => {
    afterEach(() => {
        vi.restoreAllMocks();
        localStorage.clear();
    });

    it('returns stored id after setting it', () => {
        setLastUsedDatasetId('project-1', 'dataset-1');

        expect(getLastUsedDatasetId('project-1')).toBe('dataset-1');
    });

    it('returns undefined when project has no stored id', () => {
        expect(getLastUsedDatasetId('project-1')).toBeUndefined();
    });

    it('isolates stored ids per project', () => {
        setLastUsedDatasetId('project-1', 'dataset-1');

        expect(getLastUsedDatasetId('project-1')).toBe('dataset-1');
        expect(getLastUsedDatasetId('project-2')).toBeUndefined();
    });

    it('removes stored id when cleared', () => {
        setLastUsedDatasetId('project-1', 'dataset-1');

        clearLastUsedDatasetId('project-1');

        expect(getLastUsedDatasetId('project-1')).toBeUndefined();
    });

    it('returns undefined for an empty stored id', () => {
        localStorage.setItem('physicalai:lastDataset:project-1', '');

        expect(getLastUsedDatasetId('project-1')).toBeUndefined();
    });

    it('does not throw when localStorage access throws', () => {
        vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
            throw new Error('Storage unavailable');
        });
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
            throw new Error('Storage unavailable');
        });
        vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {
            throw new Error('Storage unavailable');
        });

        expect(getLastUsedDatasetId('project-1')).toBeUndefined();
        expect(() => setLastUsedDatasetId('project-1', 'dataset-1')).not.toThrow();
        expect(() => clearLastUsedDatasetId('project-1')).not.toThrow();
    });
});

describe('useRecordLastUsedDataset', () => {
    afterEach(() => {
        vi.restoreAllMocks();
        localStorage.clear();
    });

    it('records id on mount and when dataset id changes', () => {
        let datasetId: string | undefined = 'dataset-1';
        const { rerender } = renderHook(() => useRecordLastUsedDataset('project-1', datasetId));

        expect(getLastUsedDatasetId('project-1')).toBe('dataset-1');

        datasetId = 'dataset-2';
        rerender();

        expect(getLastUsedDatasetId('project-1')).toBe('dataset-2');
    });

    it('does not write when dataset id is undefined', () => {
        let datasetId: string | undefined = 'dataset-1';
        const { rerender } = renderHook(() => useRecordLastUsedDataset('project-1', datasetId));

        datasetId = undefined;
        rerender();

        expect(getLastUsedDatasetId('project-1')).toBe('dataset-1');
    });
});

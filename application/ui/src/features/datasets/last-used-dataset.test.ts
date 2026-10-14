// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rs } from '@rstest/core';

import { clearLastUsedDatasetId, getLastUsedDatasetId, setLastUsedDatasetId } from './last-used-dataset';

describe('last-used-dataset storage', () => {
    afterEach(() => {
        rs.restoreAllMocks();
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
        rs.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
            throw new Error('Storage unavailable');
        });
        rs.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
            throw new Error('Storage unavailable');
        });
        rs.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {
            throw new Error('Storage unavailable');
        });

        expect(getLastUsedDatasetId('project-1')).toBeUndefined();
        expect(() => setLastUsedDatasetId('project-1', 'dataset-1')).not.toThrow();
        expect(() => clearLastUsedDatasetId('project-1')).not.toThrow();
    });
});

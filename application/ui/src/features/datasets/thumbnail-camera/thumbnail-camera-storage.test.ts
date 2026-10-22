// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rs } from '@rstest/core';

import { clearThumbnailCamera, getThumbnailCamera, setThumbnailCamera } from './thumbnail-camera-storage';

describe('thumbnail-camera storage', () => {
    afterEach(() => {
        rs.restoreAllMocks();
        localStorage.clear();
    });

    it('returns stored key after setting it', () => {
        setThumbnailCamera('dataset-1', 'observation.images.top');

        expect(getThumbnailCamera('dataset-1')).toBe('observation.images.top');
        expect(localStorage.getItem('physicalai:thumbnailCamera:dataset-1')).toBe('observation.images.top');
    });

    it('returns undefined when dataset has no stored key', () => {
        expect(getThumbnailCamera('dataset-1')).toBeUndefined();
    });

    it('isolates stored keys per dataset', () => {
        setThumbnailCamera('dataset-1', 'observation.images.top');

        expect(getThumbnailCamera('dataset-1')).toBe('observation.images.top');
        expect(getThumbnailCamera('dataset-2')).toBeUndefined();
    });

    it('removes stored key when cleared', () => {
        setThumbnailCamera('dataset-1', 'observation.images.top');

        clearThumbnailCamera('dataset-1');

        expect(getThumbnailCamera('dataset-1')).toBeUndefined();
    });

    it('returns undefined for an empty stored key', () => {
        localStorage.setItem('physicalai:thumbnailCamera:dataset-1', '');

        expect(getThumbnailCamera('dataset-1')).toBeUndefined();
    });
});

// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    clearPendingRecordedEpisodes,
    getPendingRecordedEpisodes,
    setPendingRecordedEpisodes,
} from './pending-recorded-episodes';

const DATASET_ID = 'dataset-1';

describe('pending recorded episodes', () => {
    afterEach(() => window.sessionStorage.clear());

    it('returns the expected episode count stored for a dataset', () => {
        setPendingRecordedEpisodes(DATASET_ID, 3);

        expect(getPendingRecordedEpisodes(DATASET_ID)).toMatchObject({ expectedEpisodeCount: 3 });
    });

    it('clears the expected episode count once it is reached', () => {
        setPendingRecordedEpisodes(DATASET_ID, 3);
        clearPendingRecordedEpisodes(DATASET_ID);

        expect(getPendingRecordedEpisodes(DATASET_ID)).toBeUndefined();
    });
});

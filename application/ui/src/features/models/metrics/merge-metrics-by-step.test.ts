// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { foldMetricsBatch, sortMetricsByStep } from './merge-metrics-by-step';
import type { MetricsEntry } from './types';

const entry = (overrides: Partial<MetricsEntry>): MetricsEntry => ({
    step: 0,
    epoch: null,
    train_loss: null,
    val_loss: null,
    'lr-AdamW': null,
    ...overrides,
});

// Convenience for tests that only care about the end result of merging a flat
// list of raw entries in one go (as opposed to across multiple batches).
const mergeAllAtOnce = (entries: MetricsEntry[]): MetricsEntry[] =>
    sortMetricsByStep(foldMetricsBatch(new Map(), entries));

describe('foldMetricsBatch + sortMetricsByStep (merging a full list at once)', () => {
    it('returns an empty array for empty input', () => {
        expect(mergeAllAtOnce([])).toEqual([]);
    });

    it('returns a single entry unchanged', () => {
        const single = entry({ step: 1, train_loss: 0.5 });

        expect(mergeAllAtOnce([single])).toEqual([single]);
    });

    it('sorts entries by step ascending', () => {
        const result = mergeAllAtOnce([entry({ step: 3 }), entry({ step: 1 }), entry({ step: 2 })]);

        expect(result.map((e) => e.step)).toEqual([1, 2, 3]);
    });

    it('merges multiple entries for the same step into one', () => {
        const result = mergeAllAtOnce([entry({ step: 1, train_loss: 0.5 }), entry({ step: 1, val_loss: 0.3 })]);

        expect(result).toHaveLength(1);
    });

    it('fills a null field on an earlier entry with a later non-null value for the same step', () => {
        const result = mergeAllAtOnce([
            entry({ step: 1, train_loss: 0.5, val_loss: null }),
            entry({ step: 1, train_loss: null, val_loss: 0.3 }),
        ]);

        expect(result[0]).toMatchObject({ train_loss: 0.5, val_loss: 0.3 });
    });

    it('lets a later non-null value overwrite an earlier value for the same field', () => {
        const result = mergeAllAtOnce([entry({ step: 1, train_loss: 0.5 }), entry({ step: 1, train_loss: 0.9 })]);

        expect(result[0].train_loss).toBe(0.9);
    });

    it('keeps the earlier value when the later entry has null/undefined for that field', () => {
        const result = mergeAllAtOnce([entry({ step: 1, train_loss: 0.5 }), entry({ step: 1, train_loss: undefined })]);

        expect(result[0].train_loss).toBe(0.5);
    });

    it('normalizes a missing field to null when never provided across merged entries', () => {
        const result = mergeAllAtOnce([
            entry({ step: 1, epoch: null, train_loss: null, val_loss: null, 'lr-AdamW': null }),
        ]);

        expect(result[0]).toEqual({ step: 1, epoch: null, train_loss: null, val_loss: null, 'lr-AdamW': null });
    });

    it('keeps distinct steps as separate entries', () => {
        const result = mergeAllAtOnce([entry({ step: 1, train_loss: 0.5 }), entry({ step: 2, train_loss: 0.4 })]);

        expect(result).toHaveLength(2);
        expect(result.map((e) => e.step)).toEqual([1, 2]);
    });
});

describe('foldMetricsBatch', () => {
    it('does not mutate the map passed in', () => {
        const initial = new Map<number, MetricsEntry>();

        foldMetricsBatch(initial, [entry({ step: 1, train_loss: 0.5 })]);

        expect(initial.size).toBe(0);
    });

    it('folds a new batch into an existing map without reprocessing prior entries', () => {
        const afterFirstBatch = foldMetricsBatch(new Map(), [entry({ step: 1, train_loss: 0.5 })]);
        const afterSecondBatch = foldMetricsBatch(afterFirstBatch, [entry({ step: 2, train_loss: 0.4 })]);

        expect(afterSecondBatch.size).toBe(2);
        expect(afterSecondBatch.get(1)).toMatchObject({ train_loss: 0.5 });
        expect(afterSecondBatch.get(2)).toMatchObject({ train_loss: 0.4 });
    });

    it('merges a later batch for the same step into the existing entry, coalescing null fields', () => {
        const afterFirstBatch = foldMetricsBatch(new Map(), [entry({ step: 1, train_loss: 0.5, val_loss: null })]);
        const afterSecondBatch = foldMetricsBatch(afterFirstBatch, [
            entry({ step: 1, train_loss: null, val_loss: 0.3 }),
        ]);

        expect(afterSecondBatch.size).toBe(1);
        expect(afterSecondBatch.get(1)).toMatchObject({ train_loss: 0.5, val_loss: 0.3 });
    });

    it('returns a new map reference each call so reference-equality consumers detect the update', () => {
        const initial = new Map<number, MetricsEntry>();
        const result = foldMetricsBatch(initial, [entry({ step: 1 })]);

        expect(result).not.toBe(initial);
    });
});

describe('sortMetricsByStep', () => {
    it('returns an empty array for an empty map', () => {
        expect(sortMetricsByStep(new Map())).toEqual([]);
    });

    it('sorts entries by step ascending regardless of insertion order', () => {
        const byStep = new Map<number, MetricsEntry>([
            [3, entry({ step: 3 })],
            [1, entry({ step: 1 })],
            [2, entry({ step: 2 })],
        ]);

        expect(sortMetricsByStep(byStep).map((e) => e.step)).toEqual([1, 2, 3]);
    });
});

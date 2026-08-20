// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { mergeMetricsByStep } from './merge-metrics-by-step';
import { MetricsEntry } from './types';

const entry = (overrides: Partial<MetricsEntry>): MetricsEntry => ({
    step: 0,
    epoch: null,
    train_loss: null,
    val_loss: null,
    'lr-AdamW': null,
    ...overrides,
});

describe('mergeMetricsByStep', () => {
    it('returns an empty array for empty input', () => {
        expect(mergeMetricsByStep([])).toEqual([]);
    });

    it('returns a single entry unchanged', () => {
        const single = entry({ step: 1, train_loss: 0.5 });

        expect(mergeMetricsByStep([single])).toEqual([single]);
    });

    it('sorts entries by step ascending', () => {
        const result = mergeMetricsByStep([entry({ step: 3 }), entry({ step: 1 }), entry({ step: 2 })]);

        expect(result.map((e) => e.step)).toEqual([1, 2, 3]);
    });

    it('merges multiple entries for the same step into one', () => {
        const result = mergeMetricsByStep([entry({ step: 1, train_loss: 0.5 }), entry({ step: 1, val_loss: 0.3 })]);

        expect(result).toHaveLength(1);
    });

    it('fills a null field on an earlier entry with a later non-null value for the same step', () => {
        const result = mergeMetricsByStep([
            entry({ step: 1, train_loss: 0.5, val_loss: null }),
            entry({ step: 1, train_loss: null, val_loss: 0.3 }),
        ]);

        expect(result[0]).toMatchObject({ train_loss: 0.5, val_loss: 0.3 });
    });

    it('lets a later non-null value overwrite an earlier value for the same field', () => {
        const result = mergeMetricsByStep([entry({ step: 1, train_loss: 0.5 }), entry({ step: 1, train_loss: 0.9 })]);

        expect(result[0].train_loss).toBe(0.9);
    });

    it('keeps the earlier value when the later entry has null/undefined for that field', () => {
        const result = mergeMetricsByStep([
            entry({ step: 1, train_loss: 0.5 }),
            entry({ step: 1, train_loss: undefined }),
        ]);

        expect(result[0].train_loss).toBe(0.5);
    });

    it('normalizes a missing field to null when never provided across merged entries', () => {
        const result = mergeMetricsByStep([
            entry({ step: 1, epoch: null, train_loss: null, val_loss: null, 'lr-AdamW': null }),
        ]);

        expect(result[0]).toEqual({ step: 1, epoch: null, train_loss: null, val_loss: null, 'lr-AdamW': null });
    });

    it('keeps distinct steps as separate entries', () => {
        const result = mergeMetricsByStep([entry({ step: 1, train_loss: 0.5 }), entry({ step: 2, train_loss: 0.4 })]);

        expect(result).toHaveLength(2);
        expect(result.map((e) => e.step)).toEqual([1, 2]);
    });
});

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TickItem } from 'recharts';

import { syncNearestStep } from './sync-nearest-step';

const ticks = (steps: number[]): TickItem[] => steps.map((value, index) => ({ value, coordinate: index * 10, index }));

describe('syncNearestStep', () => {
    it.each([
        [65, 0],
        [99, 1],
        [149, 1],
        [0, 0],
        [1000, 2],
    ])('selects the closest receiving sample for step %s', (step, expected) => {
        expect(syncNearestStep(ticks([65, 131, 197]), { activeLabel: String(step) })).toBe(expected);
    });

    it('breaks ties towards the earlier step regardless of tick order', () => {
        expect(syncNearestStep(ticks([100, 50]), { activeLabel: 75 })).toBe(1);
    });

    it('handles a single sample', () => {
        expect(syncNearestStep(ticks([65]), { activeLabel: 999 })).toBe(0);
    });

    it('returns no selection for an empty chart', () => {
        expect(syncNearestStep([], { activeLabel: 65 })).toBe(-1);
    });

    it.each([undefined, 'not a step', NaN, Infinity])('ignores invalid source step %s', (activeLabel) => {
        expect(syncNearestStep(ticks([65]), { activeLabel })).toBe(-1);
    });
});

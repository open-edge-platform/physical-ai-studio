// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { formatElapsed, formatEpisodeCounts } from './format';

describe('formatElapsed', () => {
    it.each([
        [0, '00:00'],
        [999, '00:00'],
        [14_000, '00:14'],
        [61_000, '01:01'],
        [59 * 60_000 + 59_000, '59:59'],
    ])('formats %i ms as mm:ss (%s)', (ms, expected) => {
        expect(formatElapsed(ms)).toBe(expected);
    });

    it.each([
        [3_600_000, '01:00:00'],
        [3_600_000 + 2 * 60_000 + 3_000, '01:02:03'],
    ])('switches to hh:mm:ss from one hour onwards (%i ms)', (ms, expected) => {
        expect(formatElapsed(ms)).toBe(expected);
    });

    it('clamps negative durations to zero', () => {
        expect(formatElapsed(-5_000)).toBe('00:00');
    });
});

describe('formatEpisodeCounts', () => {
    it('returns null for an empty dataset', () => {
        expect(formatEpisodeCounts(0, 0)).toBeNull();
    });

    it('omits the session part when nothing was recorded in this session', () => {
        expect(formatEpisodeCounts(12, 0)).toBe('12 episodes in dataset');
    });

    it('uses the singular form for one episode', () => {
        expect(formatEpisodeCounts(1, 0)).toBe('1 episode in dataset');
    });

    it('appends the session count when episodes were recorded in this session', () => {
        expect(formatEpisodeCounts(12, 3)).toBe('12 episodes in dataset · 3 this session');
        expect(formatEpisodeCounts(1, 1)).toBe('1 episode in dataset · 1 this session');
    });
});

import { describe, expect, it } from '@rstest/core';

import {
    canSubmitTrim,
    formatTrimRangeLabel,
    getFrameAt,
    getFullTrimRange,
    isFullTrimRange,
    MIN_TRIM_FRAMES,
    MOTION_PADDING_SECONDS,
    MOTION_THRESHOLD,
    suggestTrimRange,
    withTrimEnd,
    withTrimStart,
} from './trim-range';

const FPS = 30;
const PAD = Math.round(MOTION_PADDING_SECONDS * FPS);
const KEYS = ['shoulder_pan.pos'];

const buildActions = (length: number, values: Record<number, number>): number[][] => {
    const rows: number[][] = [];
    let current = 0;
    for (let frame = 0; frame < length; frame++) {
        current = values[frame] ?? current;
        rows.push([current]);
    }
    return rows;
};

describe('getFullTrimRange / isFullTrimRange', () => {
    it('returns the full range', () => {
        expect(getFullTrimRange(100)).toEqual({ start: 0, end: 100 });
    });

    it('detects the full range', () => {
        expect(isFullTrimRange({ start: 0, end: 100 }, 100)).toBe(true);
        expect(isFullTrimRange({ start: 1, end: 100 }, 100)).toBe(false);
        expect(isFullTrimRange({ start: 0, end: 99 }, 100)).toBe(false);
    });
});

describe('canSubmitTrim', () => {
    it('is false for the full range', () => {
        expect(canSubmitTrim({ start: 0, end: 100 }, 100)).toBe(false);
    });

    it('is false for a 1-frame range', () => {
        expect(canSubmitTrim({ start: 10, end: 11 }, 100)).toBe(false);
    });

    it('is false when out of bounds', () => {
        expect(canSubmitTrim({ start: -1, end: 50 }, 100)).toBe(false);
        expect(canSubmitTrim({ start: 10, end: 101 }, 100)).toBe(false);
    });

    it('is true for a valid partial range', () => {
        expect(canSubmitTrim({ start: 10, end: 90 }, 100)).toBe(true);
        expect(canSubmitTrim({ start: 10, end: 12 }, 100)).toBe(true);
    });
});

describe('formatTrimRangeLabel', () => {
    it('formats time only, without trailing zeros', () => {
        expect(formatTrimRangeLabel({ start: 30, end: 270 }, FPS)).toBe('1s – 9s');
    });

    it('shows up to two decimals', () => {
        expect(formatTrimRangeLabel({ start: 1, end: 10 }, FPS)).toBe('0.03s – 0.33s');
    });
});

describe('suggestTrimRange', () => {
    it('returns a padded range for idle head and tail', () => {
        const length = 300;

        const actions = buildActions(length, { 100: 10, 150: 20, 200: 30 });

        expect(suggestTrimRange(actions, KEYS, FPS)).toEqual({ start: 100 - 1 - PAD, end: 200 + 1 + PAD });
    });

    it('clamps start to 0 when motion begins at frame 1', () => {
        const actions = buildActions(100, { 1: 10, 50: 20 });

        expect(suggestTrimRange(actions, KEYS, FPS)).toEqual({ start: 0, end: 50 + 1 + PAD });
    });

    it('clamps end to length when motion reaches the last frame', () => {
        const length = 100;
        const actions = buildActions(length, { 50: 10, 99: 20 });

        expect(suggestTrimRange(actions, KEYS, FPS)).toEqual({ start: 50 - 1 - PAD, end: length });
    });

    it('returns undefined when all frames are idle', () => {
        expect(suggestTrimRange(buildActions(100, {}), KEYS, FPS)).toBeUndefined();
    });

    it('ignores motion on non-.pos keys', () => {
        const actions = [
            [0, 0],
            [0, 10],
            [0, 20],
            [0, 30],
        ];

        expect(suggestTrimRange(actions, ['shoulder_pan.pos', 'gripper.vel'], FPS)).toBeUndefined();
    });

    it('returns undefined when there are no .pos keys', () => {
        expect(suggestTrimRange(buildActions(10, { 5: 10 }), ['gripper.vel'], FPS)).toBeUndefined();
    });

    it('does not count a delta of exactly the threshold as motion', () => {
        const actions = buildActions(100, { 50: MOTION_THRESHOLD });

        expect(suggestTrimRange(actions, KEYS, FPS)).toBeUndefined();
    });

    it('counts a delta just above the threshold as motion', () => {
        const actions = buildActions(100, { 50: MOTION_THRESHOLD + 0.01 });

        expect(suggestTrimRange(actions, KEYS, FPS)).toEqual({ start: 50 - 1 - PAD, end: 50 + 1 + PAD });
    });
});

describe('withTrimStart / withTrimEnd', () => {
    it('moves the start within bounds', () => {
        expect(withTrimStart({ start: 0, end: 10 }, 4)).toEqual({ start: 4, end: 10 });
        expect(withTrimStart({ start: 0, end: 10 }, -3)).toEqual({ start: 0, end: 10 });
    });

    it('keeps at least MIN_TRIM_FRAMES frames', () => {
        expect(withTrimStart({ start: 0, end: 10 }, 50)).toEqual({ start: 10 - MIN_TRIM_FRAMES, end: 10 });
        expect(withTrimEnd({ start: 4, end: 10 }, 0, 10)).toEqual({ start: 4, end: 4 + MIN_TRIM_FRAMES });
    });

    it('moves the end within bounds', () => {
        expect(withTrimEnd({ start: 0, end: 10 }, 6, 10)).toEqual({ start: 0, end: 6 });
        expect(withTrimEnd({ start: 0, end: 5 }, 50, 10)).toEqual({ start: 0, end: 10 });
    });
});

describe('getFrameAt', () => {
    it('maps seconds to a clamped frame index', () => {
        expect(getFrameAt(0.5, 30, 100)).toBe(15);
        expect(getFrameAt(-1, 30, 100)).toBe(0);
        expect(getFrameAt(99, 30, 100)).toBe(99);
    });
});

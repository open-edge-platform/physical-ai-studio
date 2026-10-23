export interface TrimRange {
    start: number;
    end: number;
}

export const MIN_TRIM_FRAMES = 2;
export const MOTION_THRESHOLD = 0.5;
export const MOTION_PADDING_SECONDS = 0.25;

export const getFullTrimRange = (length: number): TrimRange => ({ start: 0, end: length });

export const isFullTrimRange = (range: TrimRange, length: number): boolean => range.start === 0 && range.end === length;

export const canSubmitTrim = (range: TrimRange, length: number): boolean =>
    range.start >= 0 &&
    range.end <= length &&
    range.end - range.start >= MIN_TRIM_FRAMES &&
    !isFullTrimRange(range, length);

const secondsFormatter = new Intl.NumberFormat('en-US', {
    style: 'unit',
    unit: 'second',
    unitDisplay: 'narrow',
    maximumFractionDigits: 2,
});

export const formatSeconds = (seconds: number): string => secondsFormatter.format(seconds);

export const formatTrimRangeLabel = (range: TrimRange, fps: number): string =>
    `${formatSeconds(range.start / fps)} – ${formatSeconds(range.end / fps)}`;

export const suggestTrimRange = (actions: number[][], actionKeys: string[], fps: number): TrimRange | undefined => {
    const posColumns = actionKeys.flatMap((key, column) => (key.endsWith('.pos') ? [column] : []));
    if (posColumns.length === 0) return undefined;

    const isMoving = (frame: number) =>
        posColumns.some((column) => Math.abs(actions[frame][column] - actions[frame - 1][column]) > MOTION_THRESHOLD);

    let firstMoving = -1;
    let lastMoving = -1;
    for (let frame = 1; frame < actions.length; frame++) {
        if (isMoving(frame)) {
            if (firstMoving === -1) firstMoving = frame;
            lastMoving = frame;
        }
    }
    if (firstMoving === -1) return undefined;

    const pad = Math.round(MOTION_PADDING_SECONDS * fps);
    const start = Math.max(0, firstMoving - 1 - pad);
    const end = Math.min(actions.length, lastMoving + 1 + pad);

    return end - start >= MIN_TRIM_FRAMES ? { start, end } : undefined;
};

export const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

/** Moves the first kept frame, keeping at least `MIN_TRIM_FRAMES` frames. */
export const withTrimStart = (range: TrimRange, frame: number): TrimRange => ({
    start: clamp(frame, 0, range.end - MIN_TRIM_FRAMES),
    end: range.end,
});

/** Moves the exclusive end of the range, keeping at least `MIN_TRIM_FRAMES` frames. */
export const withTrimEnd = (range: TrimRange, frame: number, length: number): TrimRange => ({
    start: range.start,
    end: clamp(frame, range.start + MIN_TRIM_FRAMES, length),
});

// Absorbs floating-point error (e.g. 0.29 * 100 = 28.999999999999996) so a time that was
// computed as `frame / fps` maps back to that same frame instead of the previous one.
const FRAME_EPSILON = 1e-6;

/** Index of the frame shown at `time` seconds. */
export const getFrameAt = (time: number, fps: number, length: number): number =>
    clamp(Math.floor(time * fps + FRAME_EPSILON), 0, length - 1);

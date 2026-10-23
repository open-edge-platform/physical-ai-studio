import { SchemaEpisode } from '../../../../api/openapi-spec';

export const JOINT_COLORS = ['#4e9cf5', '#e8684a', '#3fb27f', '#e0a33a', '#d85c8a', '#8e7cf0', '#5ac3f8'];
/** Pastel tones, lighter than `JOINT_COLORS`, so grippers stand apart from arm joints in the legend and tooltip. */
export const GRIPPER_COLORS = ['#a78bfa', '#f9a8d4', '#fde68a', '#99f6e4'];

export interface JointSeries {
    key: string;
    label: string;
    color: string;
    isGripper: boolean;
    values: number[];
}

export type Extent = [min: number, max: number];

export const isGripperKey = (key: string): boolean => key.includes('gripper');

export const getJointSeries = ({ action_keys, actions }: SchemaEpisode): JointSeries[] => {
    const series = action_keys.flatMap((key, column) =>
        key.endsWith('.pos')
            ? [
                  {
                      key,
                      label: key.replace(/\.pos$/, '').replace(/_/g, ' '),
                      isGripper: isGripperKey(key),
                      values: actions.map((row) => row[column]),
                  },
              ]
            : []
    );

    let armIndex = 0;
    let gripperIndex = 0;
    return series.map((item) => ({
        ...item,
        color: item.isGripper
            ? GRIPPER_COLORS[gripperIndex++ % GRIPPER_COLORS.length]
            : JOINT_COLORS[armIndex++ % JOINT_COLORS.length],
    }));
};

/** Σ|Δjoint| per frame; frame 0 is 0. Near zero means the arm is idle. */
export const getMotion = (series: JointSeries[], length: number): number[] =>
    Array.from({ length }, (_, frame) =>
        frame === 0 ? 0 : series.reduce((sum, { values }) => sum + Math.abs(values[frame] - values[frame - 1]), 0)
    );

export const getExtent = (values: number[]): Extent => {
    const min = Math.min(...values);
    const max = Math.max(...values);

    return min === max ? [min - 1, max + 1] : [min, max];
};

/** Position of `value` within `total` as a CSS percentage, e.g. a frame within the episode length. */
export const toPercent = (value: number, total: number): string => `${(value / total) * 100}%`;

export const TRACK_WIDTH = 1000;
export const TRACK_HEIGHT = 100;

/** Maps `value` to 0..1 within `extent`: 0 = min, 1 = max. */
export const normalize = (value: number, [min, max]: Extent): number => (value - min) / (max - min);

/**
 * Horizontal SVG coordinate of the `index`-th point (usually a frame) out of `count` points.
 *
 * Points are spaced evenly across the track: the first lands at x = 0 and the last at x = `TRACK_WIDTH`.
 * `count - 1` is the number of gaps between points, e.g. 3 points → 2 gaps → x = 0, 500, 1000.
 */
const toTrackX = (index: number, count: number): number => (index / Math.max(1, count - 1)) * TRACK_WIDTH;

/**
 * Vertical SVG coordinate of a value already normalized to 0..1 (see `normalize`), with the axis inverted.
 *
 * SVG's origin is the top-left corner and y grows downwards, while a chart's y grows upwards from the
 * bottom. Using `normalized * TRACK_HEIGHT` directly would draw the chart upside down (max at the bottom,
 * min at the top), so the result is flipped by subtracting it from `TRACK_HEIGHT`:
 * 1 (max) → y = 0 (top), 0.5 → y = `TRACK_HEIGHT / 2` (middle), 0 (min) → y = `TRACK_HEIGHT` (bottom).
 */
const toTrackY = (normalized: number): number => TRACK_HEIGHT - normalized * TRACK_HEIGHT;

/**
 * SVG polyline `points` (`"x,y x,y ..."`) inside a `TRACK_WIDTH` × `TRACK_HEIGHT` viewBox.
 *
 * `extent` is passed in so the caller picks the scale (per joint or shared).
 */
export const toPolylinePoints = (values: number[], extent: Extent): string =>
    values.map((value, index) => `${toTrackX(index, values.length)},${toTrackY(normalize(value, extent))}`).join(' ');

import { getMockedEpisodeDetail } from '../../../../test-utils/mocks/mock-episode';
import {
    getJointSeries,
    GRIPPER_COLORS,
    isGripperKey,
    JOINT_COLORS,
    normalize,
    toPolylinePoints,
    TRACK_HEIGHT,
    TRACK_WIDTH,
} from './trim-timeline-data';

describe('isGripperKey', () => {
    it.each(['gripper.pos', 'left_gripper.pos', 'right_gripper.pos'])('detects %s as a gripper', (key) => {
        expect(isGripperKey(key)).toBe(true);
    });

    it.each(['shoulder_pan.pos', 'left_wrist_roll.pos', 'right_elbow_flex.pos'])(
        'does not detect %s as a gripper',
        (key) => {
            expect(isGripperKey(key)).toBe(false);
        }
    );
});

describe('getJointSeries', () => {
    it('marks a single-arm gripper and colors it apart from the joints', () => {
        const series = getJointSeries(
            getMockedEpisodeDetail({
                action_keys: ['shoulder_pan.pos', 'gripper.pos'],
                actions: [[1, 2]],
            })
        );

        expect(series).toEqual([
            { key: 'shoulder_pan.pos', label: 'shoulder pan', isGripper: false, color: JOINT_COLORS[0], values: [1] },
            { key: 'gripper.pos', label: 'gripper', isGripper: true, color: GRIPPER_COLORS[0], values: [2] },
        ]);
    });

    it('marks every gripper of a bimanual robot with a distinct color', () => {
        const series = getJointSeries(
            getMockedEpisodeDetail({
                action_keys: [
                    'left_shoulder_pan.pos',
                    'left_gripper.pos',
                    'right_shoulder_pan.pos',
                    'right_gripper.pos',
                ],
                actions: [[1, 2, 3, 4]],
            })
        );

        const grippers = series.filter(({ isGripper }) => isGripper);
        expect(grippers.map(({ label, color }) => ({ label, color }))).toEqual([
            { label: 'left gripper', color: GRIPPER_COLORS[0] },
            { label: 'right gripper', color: GRIPPER_COLORS[1] },
        ]);
        expect(series.filter(({ isGripper }) => !isGripper).map(({ color }) => color)).toEqual([
            JOINT_COLORS[0],
            JOINT_COLORS[1],
        ]);
    });
});

describe('normalize', () => {
    it('maps the extent to 0..1', () => {
        expect(normalize(10, [10, 30])).toBe(0);
        expect(normalize(20, [10, 30])).toBe(0.5);
        expect(normalize(30, [10, 30])).toBe(1);
    });
});

describe('toPolylinePoints', () => {
    it('spreads points across the track width and puts the max at the top', () => {
        expect(toPolylinePoints([0, 5, 10], [0, 10])).toBe(
            `0,${TRACK_HEIGHT} ${TRACK_WIDTH / 2},${TRACK_HEIGHT / 2} ${TRACK_WIDTH},0`
        );
    });

    it('places a single point at x = 0', () => {
        expect(toPolylinePoints([5], [0, 10])).toBe(`0,${TRACK_HEIGHT / 2}`);
    });
});

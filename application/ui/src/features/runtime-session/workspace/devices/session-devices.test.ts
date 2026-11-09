import { SchemaSessionEnvironment, SchemaStatusMessage } from '../../../../api/openapi-spec';
import { getMockedFollowerRobot, getMockedUnavailableRobot } from '../../../../test-utils/mocks/mock-robot';
import { getMockedSessionEnvironment } from '../../../../test-utils/mocks/mock-session-environment';
import { Statuses } from '../../runtime-session-provider';
import {
    getCameraStatus,
    getGenericStatus,
    getRobotStatus,
    getRobotSubtitle,
    toSessionDevices,
    UNAVAILABLE_ROBOT_STATUS,
} from './session-devices';

const status = (
    component: string,
    state: SchemaStatusMessage['state'] = 'running',
    error?: string
): SchemaStatusMessage => ({ type: 'status', component, state, error });

const toStatuses = (...list: SchemaStatusMessage[]): Statuses =>
    Object.fromEntries(list.map((item) => [item.component, item]));

const environment = (overrides: Partial<SchemaSessionEnvironment> = {}) =>
    getMockedSessionEnvironment({
        robot_id: 'follower-id',
        leader_id: 'leader-id',
        cameras: [
            { camera_id: 'cam-front', name: 'Front', key: 'front', width: 640, height: 480 },
            { camera_id: 'cam-wrist', name: 'Wrist', key: 'wrist', width: 320, height: 240 },
        ],
        ...overrides,
    });

describe('toSessionDevices', () => {
    it('maps follower and leader to the environment robot ids and cameras by key, in display order', () => {
        const statuses = toStatuses(
            status('environment/wrist'),
            status('environment/leader'),
            status('environment/front'),
            status('environment/follower')
        );

        const devices = toSessionDevices(statuses, environment());

        expect(devices.map(({ kind }) => kind)).toEqual(['robot', 'robot', 'camera', 'camera']);
        expect(devices[0]).toMatchObject({ role: 'follower', id: 'follower-id' });
        expect(devices[1]).toMatchObject({ role: 'leader', id: 'leader-id' });
        expect(devices[2]).toMatchObject({ id: 'cam-front', camera: { key: 'front' } });
        expect(devices[3]).toMatchObject({ id: 'cam-wrist', camera: { key: 'wrist' } });
    });

    it('works without a leader: no leader device when the environment has none', () => {
        const statuses = toStatuses(status('environment/follower'), status('environment/front'));

        const devices = toSessionDevices(statuses, environment({ leader_id: null }));

        expect(devices.map(({ kind }) => kind)).toEqual(['robot', 'camera']);
        expect(devices.some((device) => device.kind === 'robot' && device.role === 'leader')).toBe(false);
    });

    it('works for a leader-only status list, whatever the order the statuses arrived in', () => {
        const devices = toSessionDevices(toStatuses(status('environment/leader')), environment());

        expect(devices).toEqual([expect.objectContaining({ kind: 'robot', role: 'leader', id: 'leader-id' })]);
    });

    it('keeps a leader status as an unknown device when the environment has no leader id', () => {
        const devices = toSessionDevices(
            toStatuses(status('environment/leader', 'failed', 'boom')),
            environment({ leader_id: null })
        );

        expect(devices).toEqual([expect.objectContaining({ kind: 'unknown', name: 'leader' })]);
    });

    it('only has cards for devices that reported a status', () => {
        const devices = toSessionDevices(toStatuses(status('environment/front')), environment());

        expect(devices).toHaveLength(1);
        expect(devices[0]).toMatchObject({ kind: 'camera', id: 'cam-front' });
    });

    it('keeps an unrecognised component as an unknown device after the others', () => {
        const statuses = toStatuses(status('environment/mystery', 'failed', 'nope'), status('environment/follower'));

        const devices = toSessionDevices(statuses, environment());

        expect(devices.map(({ kind }) => kind)).toEqual(['robot', 'unknown']);
        expect(devices[1]).toMatchObject({ name: 'mystery', status: { error: 'nope' } });
    });

    it('ignores components outside the environment', () => {
        const statuses = toStatuses(
            status('environment'),
            status('control'),
            status('dataset'),
            status('environment/follower')
        );

        expect(toSessionDevices(statuses, environment())).toHaveLength(1);
    });
});

describe('getRobotStatus', () => {
    const recording = { dataset: { is_recording: true } } as Parameters<typeof getRobotStatus>[2];
    const teleop = { control: { type: 'teleop' } } as Parameters<typeof getRobotStatus>[2];

    it('follower: connecting while missing or starting', () => {
        const connecting = { label: 'Connecting', variant: 'yellow' };

        expect(getRobotStatus('follower', undefined, null)).toEqual(connecting);
        expect(getRobotStatus('follower', status('environment/follower', 'starting'), null)).toEqual(connecting);
    });

    it('follower: error when failed, stopped when stopped', () => {
        expect(getRobotStatus('follower', status('environment/follower', 'failed'), null)).toEqual({
            label: 'Error',
            variant: 'negative',
        });
        expect(getRobotStatus('follower', status('environment/follower', 'stopped'), null)).toEqual({
            label: 'Stopped',
            variant: 'neutral',
        });
    });

    it('follower: ready when running, recording while the dataset records', () => {
        const running = status('environment/follower');

        expect(getRobotStatus('follower', running, null)).toEqual({ label: 'Ready', variant: 'positive' });
        expect(getRobotStatus('follower', running, recording)).toEqual({ label: 'Recording', variant: 'positive' });
    });

    it('leader: connecting while missing or starting', () => {
        const connecting = { label: 'Connecting', variant: 'yellow' };

        expect(getRobotStatus('leader', undefined, null)).toEqual(connecting);
        expect(getRobotStatus('leader', status('environment/leader', 'starting'), null)).toEqual(connecting);
    });

    it('leader: error when failed, stopped when stopped', () => {
        expect(getRobotStatus('leader', status('environment/leader', 'failed'), null)).toEqual({
            label: 'Error',
            variant: 'negative',
        });
        expect(getRobotStatus('leader', status('environment/leader', 'stopped'), null)).toEqual({
            label: 'Stopped',
            variant: 'neutral',
        });
    });

    it('leader: idle when running, driving while teleoperating', () => {
        const running = status('environment/leader');

        expect(getRobotStatus('leader', running, null)).toEqual({ label: 'Idle', variant: 'neutral' });
        expect(getRobotStatus('leader', running, teleop)).toEqual({ label: 'Driving', variant: 'info' });
    });

    it('an unavailable robot is an error whatever the role', () => {
        expect(UNAVAILABLE_ROBOT_STATUS).toEqual({ label: 'Error', variant: 'negative' });
    });
});

describe('getCameraStatus', () => {
    it.each([
        [undefined, 'Connecting', 'yellow'],
        ['starting', 'Connecting', 'yellow'],
        ['running', 'Streaming', 'positive'],
        ['failed', 'Error', 'negative'],
        ['stopped', 'Stopped', 'neutral'],
    ] as const)('%s -> %s', (state, label, variant) => {
        expect(getCameraStatus(state && status('environment/front', state))).toEqual({ label, variant });
    });
});

describe('getGenericStatus', () => {
    it.each([
        ['starting', 'Connecting', 'yellow'],
        ['running', 'Running', 'positive'],
        ['failed', 'Error', 'negative'],
        ['stopped', 'Stopped', 'neutral'],
    ] as const)('%s -> %s', (state, label, variant) => {
        expect(getGenericStatus(status('environment/x', state))).toEqual({ label, variant });
    });
});

describe('getRobotSubtitle', () => {
    it('names the type and the role', () => {
        expect(getRobotSubtitle(getMockedFollowerRobot(), 'follower')).toBe('SO101 Follower · Follower arm');
        expect(getRobotSubtitle(getMockedFollowerRobot(), 'leader')).toBe('SO101 Follower · Leader arm');
    });

    it('turns underscores of a bimanual type into spaces', () => {
        const bimanual = getMockedFollowerRobot({ type: 'Bimanual_SO101_Follower' } as never);

        expect(getRobotSubtitle(bimanual, 'follower')).toBe('Bimanual SO101 Follower · Follower arm');
    });

    it('marks an unavailable robot', () => {
        expect(getRobotSubtitle(getMockedUnavailableRobot(), 'follower')).toBe(
            'Acme Follower · Follower arm (unavailable)'
        );
    });
});

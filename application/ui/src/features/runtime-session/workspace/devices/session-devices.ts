import {
    SchemaSessionCamera,
    SchemaSessionEnvironment,
    SchemaSessionState,
    SchemaStatusMessage,
} from '../../../../api/openapi-spec';
import { isUnavailableRobot, SchemaRobot } from '../../../robots/robot-types';
import { Statuses } from '../../runtime-session-provider';

export type DeviceStatusVariant = 'neutral' | 'info' | 'positive' | 'negative' | 'yellow';

export interface DeviceStatus {
    label: string;
    variant: DeviceStatusVariant;
}

export type RobotRole = 'follower' | 'leader';

export type SessionDevice =
    | { kind: 'robot'; role: RobotRole; id: string; status: SchemaStatusMessage }
    | { kind: 'camera'; id: string; camera: SchemaSessionCamera; status: SchemaStatusMessage }
    | { kind: 'unknown'; name: string; status: SchemaStatusMessage };

type RobotRuntimeState = Pick<SchemaSessionState, 'dataset' | 'control'>;

const DEVICE_PREFIX = 'environment/';

const CONNECTING: DeviceStatus = { label: 'Connecting', variant: 'yellow' };
const STOPPED: DeviceStatus = { label: 'Stopped', variant: 'neutral' };

/**
 * The environment's devices that reported a status, in display order: follower, leader, cameras, anything else.
 *
 * Robots are matched by the ids in the environment, so a follower-only environment simply has no leader device.
 * A robot status whose id is missing is kept as an `unknown` device, so no failure is hidden.
 */
export const toSessionDevices = (statuses: Statuses, environment: SchemaSessionEnvironment): SessionDevice[] => {
    const reported = new Map<string, SchemaStatusMessage>();
    for (const status of Object.values(statuses)) {
        if (status.component.startsWith(DEVICE_PREFIX)) {
            reported.set(status.component.slice(DEVICE_PREFIX.length), status);
        }
    }

    const devices: SessionDevice[] = [];
    const claimed = new Set<string>();

    const robotIds: Record<RobotRole, string | null | undefined> = {
        follower: environment.robot_id,
        leader: environment.leader_id,
    };
    for (const role of ['follower', 'leader'] as const) {
        const status = reported.get(role);
        const id = robotIds[role];
        if (status === undefined) {
            continue;
        }
        claimed.add(role);
        devices.push(id ? { kind: 'robot', role, id, status } : { kind: 'unknown', name: role, status });
    }

    for (const camera of environment.cameras) {
        const status = reported.get(camera.key);
        if (status === undefined || claimed.has(camera.key)) {
            continue;
        }
        claimed.add(camera.key);
        devices.push({ kind: 'camera', id: camera.camera_id, camera, status });
    }

    for (const [name, status] of reported) {
        if (!claimed.has(name)) {
            devices.push({ kind: 'unknown', name, status });
        }
    }

    return devices;
};

/** The follower's state in the session, or the leader's: what it is doing, not only whether it is up. */
export const getRobotStatus = (
    role: RobotRole,
    status: SchemaStatusMessage | undefined,
    state: RobotRuntimeState | null | undefined
): DeviceStatus => {
    switch (status?.state) {
        case 'failed':
            return { label: 'Error', variant: 'negative' };
        case 'stopped':
            return STOPPED;
        case 'running':
            if (role === 'follower') {
                return { label: state?.dataset?.is_recording ? 'Recording' : 'Ready', variant: 'positive' };
            }
            return state?.control?.type === 'teleop'
                ? { label: 'Driving', variant: 'info' }
                : { label: 'Idle', variant: 'neutral' };
        default:
            return CONNECTING;
    }
};

/** Status for an unavailable (plugin missing) robot, regardless of role. */
export const UNAVAILABLE_ROBOT_STATUS: DeviceStatus = { label: 'Error', variant: 'negative' };

export const getCameraStatus = (status: SchemaStatusMessage | undefined): DeviceStatus => {
    switch (status?.state) {
        case 'running':
            return { label: 'Streaming', variant: 'positive' };
        case 'failed':
            return { label: 'Error', variant: 'negative' };
        case 'stopped':
            return STOPPED;
        default:
            return CONNECTING;
    }
};

/** A device that is neither robot nor camera: only its raw state is known. */
export const getGenericStatus = (status: SchemaStatusMessage): DeviceStatus => {
    switch (status.state) {
        case 'running':
            return { label: 'Running', variant: 'positive' };
        case 'failed':
            return { label: 'Error', variant: 'negative' };
        case 'stopped':
            return STOPPED;
        default:
            return CONNECTING;
    }
};

/** Second line of a robot card, e.g. `so101 follower · Follower arm`. */
export const getRobotSubtitle = (robot: SchemaRobot, role: RobotRole): string => {
    const base = `${robot.type.replaceAll('_', ' ')} · ${role === 'follower' ? 'Follower arm' : 'Leader arm'}`;

    return isUnavailableRobot(robot) ? `${base} (unavailable)` : base;
};

import { SchemaEnvironmentWithRelations, SchemaRuntimeSessionInfo } from '../../../api/openapi-spec';
import { isUnavailableRobot, SchemaRobot } from '../../robots/robot-types';
import type { RuntimeSessionState } from '../../robots/runtime-session-provider';

export type RobotStatusVariant = 'neutral' | 'info' | 'positive' | 'negative' | 'yellow';

export interface RobotStatus {
    label: string;
    variant: RobotStatusVariant;
}

export type RobotRole = 'follower' | 'leader';

export interface SessionRobot {
    robot: SchemaRobot;
    role: RobotRole;
}

type FollowerRuntimeState = Pick<RuntimeSessionState, 'connected' | 'is_recording'>;
type LeaderRuntimeState = Pick<RuntimeSessionState, 'connected' | 'has_leader' | 'follower_source'>;

export const getFollowerStatus = (
    state: FollowerRuntimeState,
    session: SchemaRuntimeSessionInfo | undefined
): RobotStatus => {
    if (session?.status === 'error' || session?.status === 'unreachable') {
        return { label: 'Error', variant: 'negative' };
    }
    if (!state.connected || session?.status === 'starting') {
        return { label: 'Connecting', variant: 'yellow' };
    }
    if (state.is_recording) {
        return { label: 'Recording', variant: 'positive' };
    }
    return { label: 'Ready', variant: 'positive' };
};

export const getLeaderStatus = (state: LeaderRuntimeState): RobotStatus => {
    if (!state.connected) {
        return { label: 'Connecting', variant: 'yellow' };
    }
    if (!state.has_leader) {
        return { label: 'Disconnected', variant: 'negative' };
    }
    if (state.follower_source === 'teleop') {
        return { label: 'Driving', variant: 'info' };
    }
    return { label: 'Idle', variant: 'neutral' };
};

/** Status for an unavailable (plugin missing) robot, regardless of role. */
export const UNAVAILABLE_ROBOT_STATUS: RobotStatus = { label: 'Error', variant: 'negative' };

/** Robots taking part in the session: the follower, followed by the leader when teleoperated by a robot. */
export const getSessionRobots = (environment: SchemaEnvironmentWithRelations): SessionRobot[] => {
    const entry = environment.robots?.[0];
    if (entry === undefined) {
        return [];
    }

    const robots: SessionRobot[] = [{ robot: entry.robot, role: 'follower' }];
    if (entry.tele_operator.type === 'robot' && entry.tele_operator.robot) {
        robots.push({ robot: entry.tele_operator.robot, role: 'leader' });
    }

    return robots;
};

/** Second line of a robot card, e.g. `so101 follower · Follower arm`. */
export const getRobotSubtitle = ({ robot, role }: SessionRobot): string => {
    const base = `${robot.type.replaceAll('_', ' ')} · ${role === 'follower' ? 'Follower arm' : 'Leader arm'}`;

    return isUnavailableRobot(robot) ? `${base} (unavailable)` : base;
};

// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaEnvironmentWithRelations, SchemaRuntimeSessionInfo } from '../../../api/openapi-spec';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import {
    getMockedFollowerRobot,
    getMockedLeaderRobot,
    getMockedUnavailableRobot,
} from '../../../test-utils/mocks/mock-robot';
import { getMockedRuntimeSessionInfo } from '../../../test-utils/mocks/mock-runtime-session-info';
import { getFollowerStatus, getLeaderStatus, getRobotSubtitle, getSessionRobots } from './robot-status';

const session = (status: SchemaRuntimeSessionInfo['status']) => getMockedRuntimeSessionInfo({ status });

describe('getFollowerStatus', () => {
    const connected = { connected: true, is_recording: false };

    it.each(['error', 'unreachable'] as const)('reports Error when the session is %s', (status) => {
        expect(getFollowerStatus(connected, session(status))).toEqual({ label: 'Error', variant: 'negative' });
    });

    it('reports Error even while recording', () => {
        expect(getFollowerStatus({ connected: true, is_recording: true }, session('error'))).toEqual({
            label: 'Error',
            variant: 'negative',
        });
    });

    it('reports Connecting while the socket is not connected', () => {
        expect(getFollowerStatus({ connected: false, is_recording: false }, undefined)).toEqual({
            label: 'Connecting',
            variant: 'yellow',
        });
    });

    it('reports Connecting while the session is starting', () => {
        expect(getFollowerStatus(connected, session('starting')).label).toBe('Connecting');
    });

    it('reports Recording while an episode is being recorded', () => {
        expect(getFollowerStatus({ connected: true, is_recording: true }, session('running'))).toEqual({
            label: 'Recording',
            variant: 'positive',
        });
    });

    it('reports Ready when connected and idle, with or without a known session', () => {
        expect(getFollowerStatus(connected, session('running'))).toEqual({ label: 'Ready', variant: 'positive' });
        expect(getFollowerStatus(connected, undefined)).toEqual({ label: 'Ready', variant: 'positive' });
    });
});

describe('getLeaderStatus', () => {
    const base = { connected: true, has_leader: true, follower_source: 'hold' as const };

    it('reports Connecting while the socket is not connected', () => {
        expect(getLeaderStatus({ ...base, connected: false })).toEqual({ label: 'Connecting', variant: 'yellow' });
    });

    it('reports Disconnected when the leader is lost', () => {
        expect(getLeaderStatus({ ...base, has_leader: false, follower_source: 'teleop' })).toEqual({
            label: 'Disconnected',
            variant: 'negative',
        });
    });

    it('reports Driving while the follower is teleoperated', () => {
        expect(getLeaderStatus({ ...base, follower_source: 'teleop' })).toEqual({ label: 'Driving', variant: 'info' });
    });

    it.each(['hold', 'policy'] as const)('reports Idle when the follower source is %s', (follower_source) => {
        expect(getLeaderStatus({ ...base, follower_source })).toEqual({ label: 'Idle', variant: 'neutral' });
    });
});

describe('getSessionRobots', () => {
    const follower = getMockedFollowerRobot();
    const leader = getMockedLeaderRobot();

    const environmentWith = (
        teleOperator: NonNullable<SchemaEnvironmentWithRelations['robots']>[number]['tele_operator']
    ) => getMockedEnvironment({ robots: [{ robot: follower, tele_operator: teleOperator }] });

    it('returns no robots for an environment without robots', () => {
        expect(getSessionRobots(getMockedEnvironment({ robots: [] }))).toEqual([]);
        expect(getSessionRobots(getMockedEnvironment({ robots: undefined }))).toEqual([]);
    });

    it('returns only the follower when there is no teleoperator', () => {
        expect(getSessionRobots(environmentWith({ type: 'none' }))).toEqual([{ robot: follower, role: 'follower' }]);
    });

    it('returns the follower followed by the leader robot', () => {
        expect(getSessionRobots(environmentWith({ type: 'robot', robot_id: leader.id, robot: leader }))).toEqual([
            { robot: follower, role: 'follower' },
            { robot: leader, role: 'leader' },
        ]);
    });

    it('skips the leader when its robot was not loaded', () => {
        expect(getSessionRobots(environmentWith({ type: 'robot', robot_id: leader.id }))).toEqual([
            { robot: follower, role: 'follower' },
        ]);
    });
});

describe('getRobotSubtitle', () => {
    it('combines the robot type and role', () => {
        expect(getRobotSubtitle({ robot: getMockedFollowerRobot(), role: 'follower' })).toBe(
            'SO101 Follower · Follower arm'
        );
        expect(getRobotSubtitle({ robot: getMockedLeaderRobot(), role: 'leader' })).toBe('SO101 Leader · Leader arm');
    });

    it('flags an unavailable robot', () => {
        expect(getRobotSubtitle({ robot: getMockedUnavailableRobot(), role: 'follower' })).toMatch(/unavailable/i);
    });
});

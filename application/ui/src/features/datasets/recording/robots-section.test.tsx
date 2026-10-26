// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { SchemaEnvironmentWithRelations, SchemaRuntimeSessionInfo } from '../../../api/openapi-spec';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedEnvironment } from '../../../test-utils/mocks/mock-environment';
import {
    getMockedFollowerRobot,
    getMockedLeaderRobot,
    getMockedUnavailableRobot,
} from '../../../test-utils/mocks/mock-robot';
import { getMockedRuntimeSession, getMockedRuntimeSessionState } from '../../../test-utils/mocks/mock-runtime-session';
import { getMockedRuntimeSessionInfo } from '../../../test-utils/mocks/mock-runtime-session-info';
import { render } from '../../../test-utils/render';
import { useRuntimeSession } from '../../robots/runtime-session-provider';
import { RobotsSection } from './robots-section';

rs.mock('../../robots/runtime-session-provider', () => ({
    useRuntimeSession: rs.fn(),
}));

const follower = getMockedFollowerRobot({ id: 'follower-1', name: 'Follower arm 1' });
const leader = getMockedLeaderRobot({ id: 'leader-1', name: 'Leader arm 1' });

type RobotEntry = NonNullable<SchemaEnvironmentWithRelations['robots']>[number];

const withLeader: RobotEntry = {
    robot: follower,
    tele_operator: { type: 'robot', robot_id: leader.id, robot: leader },
};

interface Options {
    robots?: RobotEntry[];
    state?: Parameters<typeof getMockedRuntimeSessionState>[0];
    sessions?: SchemaRuntimeSessionInfo[];
}

const renderRobots = ({ robots = [withLeader], state, sessions = [] }: Options = {}) => {
    rs.mocked(useRuntimeSession).mockReturnValue(
        getMockedRuntimeSession({
            environment: getMockedEnvironment({ robots }),
            state: getMockedRuntimeSessionState(state),
        })
    );
    server.use(http.get('/api/runtime/sessions', () => HttpResponse.json(sessions)));

    return render(<RobotsSection />);
};

describe('RobotsSection', () => {
    beforeEach(() => {
        rs.clearAllMocks();
    });

    it('lists the follower followed by the leader', async () => {
        renderRobots();

        expect(await screen.findByText('Follower arm 1')).toBeInTheDocument();
        expect(screen.getByText('Leader arm 1')).toBeInTheDocument();
        expect(screen.getByText(/Follower arm$/)).toBeInTheDocument();
        expect(screen.getByText(/Leader arm$/)).toBeInTheDocument();
    });

    it('lists only the follower when there is no leader robot', async () => {
        renderRobots({ robots: [{ robot: follower, tele_operator: { type: 'none' } }] });

        expect(await screen.findByText('Follower arm 1')).toBeInTheDocument();
        expect(screen.queryByText('Leader arm 1')).not.toBeInTheDocument();
    });

    it('renders nothing for an environment without robots', async () => {
        renderRobots({ robots: [] });

        expect(screen.queryByText('Robots')).not.toBeInTheDocument();
    });

    it('derives the follower and leader status from the session state', async () => {
        renderRobots({
            state: { follower_source: 'teleop', is_recording: true },
            sessions: [getMockedRuntimeSessionInfo()],
        });

        expect(await screen.findByText('Recording')).toBeInTheDocument();
        expect(screen.getByText('Driving')).toBeInTheDocument();
    });

    it('shows the follower as Error when its runtime session failed', async () => {
        renderRobots({
            robots: [{ robot: follower, tele_operator: { type: 'none' } }],
            sessions: [getMockedRuntimeSessionInfo({ session_name: `rt-${follower.id}`, status: 'error' })],
        });

        expect(await screen.findByText('Error')).toBeInTheDocument();
    });

    it('ignores runtime sessions of other robots', async () => {
        renderRobots({
            robots: [{ robot: follower, tele_operator: { type: 'none' } }],
            sessions: [getMockedRuntimeSessionInfo({ session_name: 'rt-other-robot', status: 'error' })],
        });

        expect(await screen.findByText('Ready')).toBeInTheDocument();
        expect(screen.queryByText('Error')).not.toBeInTheDocument();
    });

    it('shows the unavailable robot as Error and flags it', async () => {
        renderRobots({ robots: [{ robot: getMockedUnavailableRobot(), tele_operator: { type: 'none' } }] });

        expect(await screen.findByText('Missing arm')).toBeInTheDocument();
        expect(screen.getByText('Error')).toBeInTheDocument();
        expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
    });
});

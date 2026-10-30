// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, rs } from '@rstest/core';
import { act, screen } from '@testing-library/react';
import { Options } from 'react-use-websocket';

import useWebSocketWithResponse from '../../components/websockets/use-websocket-with-response';
import { getMockedEnvironment } from '../../test-utils/mocks/mock-environment';
import { render } from '../../test-utils/render';
import { useRobotSimulationLaunch } from '../simulation/robot-simulation';
import { useRobotSimulation } from '../simulation/use-robot-simulation';
import { RuntimeSessionProvider, useRuntimeSession } from './runtime-session-provider';

rs.mock('../../components/websockets/use-websocket-with-response', () => ({ default: rs.fn() }));
rs.mock('../simulation/use-robot-simulation', () => ({ useRobotSimulation: rs.fn() }));
rs.mock('../simulation/robot-simulation', () => ({ useRobotSimulationLaunch: rs.fn() }));

const environment = getMockedEnvironment({
    robots: [
        {
            robot: { id: 'follower-1', name: 'Sim arm', type: 'MuJoCo_SO101_Follower', payload: {} } as never,
            tele_operator: { type: 'none' } as never,
        },
    ],
});

const lastSocket = () => {
    const calls = rs.mocked(useWebSocketWithResponse).mock.calls;
    const [, options, connect] = calls[calls.length - 1];
    return { options: options as Options, connect };
};

const receive = (message: object) =>
    act(() => lastSocket().options.onMessage?.({ data: JSON.stringify(message) } as MessageEvent));

const open = () => act(() => lastSocket().options.onOpen?.({} as Event));

/** The robot simulation probe as `useRobotSimulation` reports it; `at` is when it was made. */
const probe = (running: boolean | undefined, at = Date.now()) =>
    rs.mocked(useRobotSimulation).mockReturnValue({
        data: running === undefined ? undefined : { running },
        dataUpdatedAt: running === undefined ? 0 : at,
        isFetchedAfterMount: running !== undefined,
    } as never);

const Waiting = () => <div>{useRuntimeSession().waitingForSimulation ? 'waiting' : 'not waiting'}</div>;

const renderProvider = async () => {
    const onError = rs.fn();
    const view = render(
        <RuntimeSessionProvider environment={environment} onError={onError}>
            <Waiting />
        </RuntimeSessionProvider>,
        { route: '/projects/project-1', path: '/projects/:project_id' }
    );
    await screen.findByText(/waiting/);
    return { ...view, onError };
};

describe('RuntimeSessionProvider', () => {
    beforeEach(() => {
        rs.mocked(useWebSocketWithResponse).mockReturnValue({
            sendJsonMessage: rs.fn(),
            sendJsonMessageAndWait: rs.fn(),
            readyState: 1,
        } as never);
        rs.mocked(useRobotSimulationLaunch).mockReturnValue({} as never);
        probe(undefined);
    });

    it('connects a simulated follower only once its simulation runs', async () => {
        await renderProvider();
        expect(lastSocket().connect).toBe(false);
        expect(screen.getByText('not waiting')).toBeInTheDocument();

        probe(false);
        await receive({ event: 'state', data: {} });
        expect(lastSocket().connect).toBe(false);
        expect(screen.getByText('waiting')).toBeInTheDocument();

        probe(true);
        await receive({ event: 'state', data: {} });
        expect(lastSocket().connect).toBe(true);
        expect(screen.getByText('not waiting')).toBeInTheDocument();
    });

    it('does not trust a cached probe from before the page opened', async () => {
        rs.mocked(useRobotSimulation).mockReturnValue({
            data: { running: true },
            dataUpdatedAt: 1,
            isFetchedAfterMount: false,
        } as never);
        await renderProvider();

        expect(lastSocket().connect).toBe(false);
    });

    it('stays open when a probe misses, and waits for a fresh probe after the simulation is lost', async () => {
        probe(true);
        const { onError } = await renderProvider();
        await open();

        probe(false);
        await receive({ event: 'state', data: {} });
        expect(lastSocket().connect).toBe(true);

        const lostAt = Date.now();
        await receive({
            event: 'error',
            message: 'No MuJoCo simulation named …',
            error_code: 'simulation_not_running',
        });
        expect(onError).not.toHaveBeenCalled();
        expect(lastSocket().connect).toBe(false);
        expect((lastSocket().options.shouldReconnect as () => boolean)()).toBe(false);
        expect(screen.getByText('waiting')).toBeInTheDocument();

        probe(true, lostAt - 1_000);
        await receive({ event: 'state', data: {} });
        expect(lastSocket().connect).toBe(false);

        probe(true, lostAt + 1_000);
        await receive({ event: 'state', data: {} });
        expect(lastSocket().connect).toBe(true);
        expect((lastSocket().options.shouldReconnect as () => boolean)()).toBe(true);
    });

    it('connects a real follower at once and reports its errors', async () => {
        rs.mocked(useRobotSimulationLaunch).mockReturnValue(null);
        const { onError } = await renderProvider();
        expect(lastSocket().connect).toBe(true);
        expect(rs.mocked(useRobotSimulation)).toHaveBeenLastCalledWith('project-1', 'follower-1', false);

        await receive({ event: 'error', message: 'Robot unplugged', error_code: 'robot_connection_failed' });

        expect(onError).toHaveBeenCalledWith('Robot unplugged');
        expect(lastSocket().connect).toBe(true);
    });
});

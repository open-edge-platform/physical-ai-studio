// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { describe, expect, it, rs } from '@rstest/core';
import { act, render, screen } from '@testing-library/react';

import { RuntimeSessionProvider, useRuntimeSession } from './runtime-session-provider';

/** The socket as the provider sees it: what it sent, and a way to answer as the server. */
const socket = {
    sent: [] as Array<{ command: string; request_id: string; environment?: { name: string } }>,
    receive: (_message: object) => {},
};

rs.mock('react-use-websocket', () => ({
    default: (_url: string, options: { onMessage: (event: MessageEvent) => void }) => {
        socket.receive = (message) => options.onMessage({ data: JSON.stringify(message) } as MessageEvent);
        return { sendJsonMessage: (message: (typeof socket.sent)[number]) => socket.sent.push(message) };
    },
}));

const draft = (name: string) => ({ name, robots: [{ robot_id: name, tele_operator: { type: 'none' as const } }] });

type Session = ReturnType<typeof useRuntimeSession>;
let session: Session;
const Probe = ({ onSession }: { onSession: (value: Session) => void }) => {
    const value = useRuntimeSession();
    useEffect(() => {
        onSession(value);
    });
    return <span>{JSON.stringify(value.environmentLoad)}</span>;
};

describe('RuntimeSessionProvider environment loads', () => {
    it('keeps the newest load, whichever answers last', () => {
        render(
            <RuntimeSessionProvider project_id='project-1' environment={null}>
                <Probe
                    onSession={(value) => {
                        session = value;
                    }}
                />
            </RuntimeSessionProvider>
        );

        act(() => {
            void session.send({ command: 'load_environment', environment: draft('A'), unload_incompatible: false });
            void session.send({ command: 'load_environment', environment: draft('B'), unload_incompatible: false });
        });
        const [first, second] = socket.sent;
        // The first load fails only after the second was sent: its failure is not what the session is loading.
        act(() =>
            socket.receive({
                type: 'ack',
                request_id: first.request_id,
                ok: false,
                error: { message: 'Start the simulation', error_code: 'simulation_not_running' },
            })
        );
        expect(session.environmentLoad).toEqual({ draft: draft('B'), error: null });

        act(() => socket.receive({ type: 'ack', request_id: second.request_id, ok: true }));
        expect(session.environmentLoad).toEqual({ draft: draft('B'), error: null });
        expect(screen.getByText(/"B"/)).toBeInTheDocument();
    });

    it('keeps why an environment failed while it loads again, until it answers', () => {
        socket.sent = [];
        render(
            <RuntimeSessionProvider project_id='project-1' environment={null}>
                <Probe
                    onSession={(value) => {
                        session = value;
                    }}
                />
            </RuntimeSessionProvider>
        );
        const notRunning = { message: 'Start the simulation', error_code: 'simulation_not_running' };

        act(() => {
            void session.send({ command: 'load_environment', environment: draft('A'), unload_incompatible: false });
        });
        act(() => socket.receive({ type: 'ack', request_id: socket.sent[0].request_id, ok: false, error: notRunning }));
        act(() => {
            void session.send({ command: 'load_environment', environment: draft('A'), unload_incompatible: false });
        });
        // Its simulation was started, and A loads again: what recovers it stays shown meanwhile.
        expect(session.environmentLoad?.error?.error_code).toBe('simulation_not_running');

        act(() => socket.receive({ type: 'ack', request_id: socket.sent[1].request_id, ok: true }));
        expect(session.environmentLoad).toEqual({ draft: draft('A'), error: null });
    });
});

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { AppFooter } from './app-footer';

class FakeWebSocket extends EventTarget {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSING = 2;
    static readonly CLOSED = 3;
    readyState = FakeWebSocket.OPEN;
    constructor(
        public url: string | URL,
        public protocols?: string | string[]
    ) {
        super();
    }
    close() {
        this.readyState = FakeWebSocket.CLOSED;
    }
    send() {}
}

class ImmediatelyClosingEventSource {
    onmessage: ((event: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;
    constructor() {
        queueMicrotask(() => this.onmessage?.({ data: 'DONE' }));
    }
    close() {}
}

const mockApi = () => {
    server.use(
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.get('/api/runtime/sessions/count', () => HttpResponse.json({ count: 0 })),
        http.get('/api/runtime/sessions', () => HttpResponse.json([])),
        http.get('/api/logs/sources', () =>
            HttpResponse.json([{ id: 'application', name: 'Application', type: 'application', created_at: null }])
        )
    );
};

describe('AppFooter', () => {
    beforeEach(() => {
        vi.stubGlobal('WebSocket', FakeWebSocket);
        vi.stubGlobal('EventSource', ImmediatelyClosingEventSource);
        mockApi();
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('shows enabled Logs and Jobs buttons with no jobs, on a project route', async () => {
        render(<AppFooter />, { route: '/projects/p1', path: '/projects/:project_id' });

        expect(await screen.findByRole('button', { name: 'Logs' })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'Jobs' })).toBeEnabled();
    });

    it('shows enabled Logs and Jobs buttons with no jobs, on a global route', async () => {
        render(<AppFooter />, { route: '/', path: '/' });

        expect(await screen.findByRole('button', { name: 'Logs' })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'Jobs' })).toBeEnabled();
    });

    it('scopes Jobs to the current project on a project route', async () => {
        const user = userEvent.setup();
        render(<AppFooter />, { route: '/projects/p1', path: '/projects/:project_id' });

        await user.click(await screen.findByRole('button', { name: 'Jobs' }));

        expect(await screen.findByText('Current project jobs')).toBeInTheDocument();
    });

    it("shows all projects' jobs on a global route", async () => {
        const user = userEvent.setup();
        render(<AppFooter />, { route: '/', path: '/' });

        await user.click(await screen.findByRole('button', { name: 'Jobs' }));

        expect(await screen.findByText('All projects jobs')).toBeInTheDocument();
    });

    it('opens application logs from the standalone Logs button, not the Jobs flow', async () => {
        const user = userEvent.setup();
        render(<AppFooter />, { route: '/', path: '/' });

        await user.click(await screen.findByRole('button', { name: 'Logs' }));

        expect(await screen.findByRole('heading', { name: 'Logs' })).toBeInTheDocument();
        expect(screen.queryByRole('tab', { name: /all jobs/i })).not.toBeInTheDocument();
    });
});

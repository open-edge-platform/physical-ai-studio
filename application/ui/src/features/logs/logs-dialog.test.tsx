// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { LogsDialog } from './logs-dialog';

/**
 * jsdom does not implement `EventSource`. `LogsDialog` streams via `fetchSSE`,
 * which uses the real browser `EventSource`, so it must be stubbed here (see the
 * same pattern in `job-table.test.tsx`). This stub lets the test control exactly
 * what payload the "stream" emits before completing.
 */
class StubEventSource {
    onmessage: ((event: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;

    constructor(private readonly messages: string[]) {
        queueMicrotask(() => {
            for (const data of this.messages) {
                this.onmessage?.({ data });
            }
            this.onmessage?.({ data: 'DONE' });
        });
    }

    close() {}
}

const stubEventSourceWith = (...messages: string[]) => {
    vi.stubGlobal(
        'EventSource',
        class extends StubEventSource {
            constructor() {
                super(messages);
            }
        }
    );
};

describe('LogsDialog', () => {
    afterEach(() => {
        vi.unstubAllGlobals();
    });

    beforeEach(() => {
        server.use(http.get('/api/logs/sources', () => HttpResponse.json([])));
    });

    it('renders a well-formed log entry', async () => {
        stubEventSourceWith(
            JSON.stringify({
                text: 'hello',
                record: {
                    message: 'hello world',
                    module: 'training_worker',
                    function: 'run',
                    line: 12,
                    level: { name: 'INFO', no: 20, icon: 'ℹ' },
                    time: { timestamp: 1_700_000_000, repr: '2023-11-14 22:13:20' },
                },
            })
        );

        render(<LogsDialog close={vi.fn()} initialSourceId='job-abc' />);

        expect(await screen.findByText('hello world')).toBeInTheDocument();
    });

    // Regression test: a just-started job's log source can briefly surface
    // malformed/partial payloads (e.g. before the underlying log file exists or
    // is fully written). Nothing in the tree used to validate the *type* of each
    // field, so a non-string `message` reached `message.includes(...)` deep in
    // rendering and threw -- with no boundary to catch it, that crashed the
    // whole app instead of just this dialog.
    it('drops a malformed log entry instead of crashing the whole dialog', async () => {
        stubEventSourceWith(
            JSON.stringify({
                text: 'bad',
                record: {
                    // `message` is a number instead of a string.
                    message: 12345,
                    module: 'training_worker',
                    function: 'run',
                    line: 12,
                    level: { name: 'INFO', no: 20, icon: 'ℹ' },
                    time: { timestamp: 1_700_000_000, repr: '2023-11-14 22:13:20' },
                },
            })
        );

        render(<LogsDialog close={vi.fn()} initialSourceId='job-abc' />);

        expect(await screen.findByText('No logs available')).toBeInTheDocument();
        // The dialog itself (and the rest of the app) must still be intact.
        expect(screen.getByText('Logs')).toBeInTheDocument();
        expect(screen.queryByText('Something went wrong while displaying these logs')).not.toBeInTheDocument();
    });
});

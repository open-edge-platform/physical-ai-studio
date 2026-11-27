// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { createQueryClient } from '../../query-client/query-client';
import { render } from '../../test-utils/render';
import { LogsDialog } from './logs-dialog';
import LOGURU_RECORDS from './loguru-records.fixture.json';

/**
 * jsdom does not implement `EventSource`. `LogsDialog` streams via `fetchSSE`,
 * which uses the real browser `EventSource`, so it must be stubbed here (see the
 * same pattern in `job-table.test.tsx`). This stub lets the test control exactly
 * what payload the "stream" emits before completing.
 */
class StubEventSource extends EventTarget {
    onmessage: ((event: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;

    constructor(private readonly messages: string[]) {
        super();
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
    rs.stubGlobal(
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
        rs.unstubAllGlobals();
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

        render(<LogsDialog close={rs.fn()} initialSourceId='job-abc' />);

        expect(await screen.findByText('hello world')).toBeInTheDocument();
    });

    it('reconnects to live logs when reopening a cached source', async () => {
        const queryClient = createQueryClient();
        let message = 'before closing';
        const close = rs.fn();
        rs.stubGlobal(
            'EventSource',
            class extends EventTarget {
                onmessage: ((event: { data: string }) => void) | null = null;
                onerror: (() => void) | null = null;

                constructor() {
                    super();
                    queueMicrotask(() => {
                        this.onmessage?.({
                            data: JSON.stringify({
                                record: {
                                    message,
                                    module: 'training_worker',
                                    function: 'run',
                                    line: 12,
                                    level: { name: 'INFO' },
                                    time: { timestamp: 1_700_000_000, repr: '2023-11-14 22:13:20' },
                                },
                            }),
                        });
                    });
                }

                close = close;
            }
        );

        const first = render(<LogsDialog close={rs.fn()} initialSourceId='job-abc' />, { queryClient });
        expect(await screen.findByText('before closing')).toBeInTheDocument();
        first.unmount();
        expect(close).toHaveBeenCalled();

        message = 'after reopening';
        const reopened = render(<LogsDialog close={rs.fn()} initialSourceId='job-abc' />, { queryClient });
        expect(await screen.findByText('after reopening')).toBeInTheDocument();
        reopened.unmount();
        queryClient.clear();
    });

    const TRACEBACK = [
        'Traceback (most recent call last):',
        '  File "workers/model_worker.py", line 125, in lifecycle',
        'RuntimeError: the export does not load',
    ].join('\n');

    const failedRecord = ({ message = 'Worker ModelWorker-1 failed', text = '' } = {}) =>
        JSON.stringify({
            // What the backend's serialized file sink writes: the traceback follows the message in `text`.
            text: text || `2026-11-26 10:29:28.336 | ERROR    | workers.base:_main:153 - ${message}\n${TRACEBACK}\n`,
            record: {
                message,
                name: 'workers.base',
                module: 'base',
                function: '_main',
                line: 153,
                level: { name: 'ERROR', no: 40, icon: '❌' },
                time: { timestamp: 1_700_000_000, repr: '2023-11-14 22:13:20' },
                exception: { type: 'RuntimeError', value: 'the export does not load', traceback: true },
                extra: { worker: 'ModelWorker' },
            },
        });

    // Written by Loguru 0.8 with serialize=True, as the backend's file sinks write them, after logging an exception
    // under the default format and three custom LOGURU_FORMAT values. Only the script path was shortened to /app.
    // Under the "level line" format ("{level}\n{message}") a message "ERROR" is also the text's first line.
    it.each(LOGURU_RECORDS)(
        'shows the whole traceback of a real Loguru record, format $format, message $message',
        async ({ entry }) => {
            stubEventSourceWith(JSON.stringify(entry));

            render(<LogsDialog close={rs.fn()} initialSourceId='inference' />);

            const exception = (await screen.findByLabelText('Exception')).textContent ?? '';
            expect(exception.startsWith('Traceback (most recent call last):')).toBe(true);
            expect(exception).toContain('in open_model');
            expect(exception.endsWith('RuntimeError: the export does not load')).toBe(true);
        }
    );

    it('finds a record by its traceback in the search', async () => {
        const user = userEvent.setup();
        stubEventSourceWith(failedRecord());

        render(<LogsDialog close={rs.fn()} initialSourceId='inference' />);
        await screen.findByText('Worker ModelWorker-1 failed');
        await user.type(screen.getByRole('searchbox', { name: 'Search logs' }), 'model_worker.py');

        expect(screen.getByText('Worker ModelWorker-1 failed')).toBeInTheDocument();
        await user.type(screen.getByRole('searchbox', { name: 'Search logs' }), ' nowhere');
        expect(await screen.findByText('No logs match the current filters')).toBeInTheDocument();
    });

    it('shows a malformed exception without breaking the log stream', async () => {
        stubEventSourceWith(
            JSON.stringify({
                record: {
                    message: 'Worker failed',
                    module: 'base',
                    function: '_main',
                    line: 153,
                    level: { name: 'ERROR', no: 40, icon: '❌' },
                    time: { timestamp: 1_700_000_000, repr: '2023-11-14 22:13:20' },
                    exception: { type: 'RuntimeError', value: { toString: null, valueOf: null } },
                    extra: {},
                },
            }),
            JSON.stringify({
                record: {
                    message: 'the next record',
                    module: 'base',
                    function: '_main',
                    line: 154,
                    level: { name: 'INFO', no: 20, icon: 'ℹ' },
                    time: { timestamp: 1_700_000_001, repr: '2023-11-14 22:13:21' },
                    exception: { type: 7 },
                },
            })
        );

        render(<LogsDialog close={rs.fn()} initialSourceId='inference' />);

        expect(await screen.findByText('the next record')).toBeInTheDocument();
        expect(screen.getByLabelText('Exception')).toHaveTextContent('RuntimeError: {"toString":null,"valueOf":null}');
        expect(screen.queryByText('Something went wrong while displaying these logs')).not.toBeInTheDocument();
    });

    it('shows the traceback of a record that has one', async () => {
        stubEventSourceWith(failedRecord());

        render(<LogsDialog close={rs.fn()} initialSourceId='inference' />);

        expect(await screen.findByText('Worker ModelWorker-1 failed')).toBeInTheDocument();
        const exception = screen.getByLabelText('Exception');
        expect(exception).toHaveTextContent('Traceback (most recent call last):');
        expect(exception).toHaveTextContent('RuntimeError: the export does not load');
    });

    it('copies the traceback with its record', async () => {
        const user = userEvent.setup();
        const writeText = rs.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
        stubEventSourceWith(failedRecord());

        render(<LogsDialog close={rs.fn()} initialSourceId='inference' />);
        await screen.findByText('Worker ModelWorker-1 failed');
        await user.click(screen.getByRole('button', { name: 'Copy logs to clipboard' }));

        expect(writeText).toHaveBeenCalledTimes(1);
        const copied = writeText.mock.calls[0][0] as string;
        expect(copied).toContain('- Worker ModelWorker-1 failed\nTraceback (most recent call last):');
        expect(copied).toContain('RuntimeError: the export does not load');
    });

    it('copies the traceback without the level line that equals the message', async () => {
        const user = userEvent.setup();
        const writeText = rs.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
        const [{ entry }] = LOGURU_RECORDS.filter(
            (record) => record.format === 'level line' && record.message === 'ERROR'
        );
        stubEventSourceWith(JSON.stringify(entry));

        render(<LogsDialog close={rs.fn()} initialSourceId='inference' />);
        await screen.findByLabelText('Exception');
        await user.click(screen.getByRole('button', { name: 'Copy logs to clipboard' }));

        const copied = writeText.mock.calls[0][0] as string;
        expect(copied).toContain(' - ERROR\nTraceback (most recent call last):');
        expect(copied).not.toContain('ERROR\nERROR');
    });

    it('takes no traceback from an empty message when the text has no exception block', async () => {
        stubEventSourceWith(
            JSON.stringify({
                text: 'ERROR\n\nnot a traceback\n',
                record: {
                    message: '',
                    module: 'base',
                    function: '_main',
                    line: 153,
                    level: { name: 'ERROR', no: 40, icon: '❌' },
                    time: { timestamp: 1_700_000_000, repr: '2023-11-14 22:13:20' },
                    exception: { type: 'ValueError', value: 'lost device', traceback: true },
                },
            })
        );

        render(<LogsDialog close={rs.fn()} initialSourceId='inference' />);

        expect(await screen.findByLabelText('Exception')).toHaveTextContent(/^ValueError: lost device$/);
    });

    it('shows the exception type and value when the text has no traceback', async () => {
        stubEventSourceWith(
            JSON.stringify({
                record: {
                    message: 'Worker failed',
                    module: 'base',
                    function: '_main',
                    line: 153,
                    level: { name: 'ERROR', no: 40, icon: '❌' },
                    time: { timestamp: 1_700_000_000, repr: '2023-11-14 22:13:20' },
                    exception: { type: 'ValueError', value: 'lost device', traceback: true },
                },
            })
        );

        render(<LogsDialog close={rs.fn()} initialSourceId='inference' />);

        expect(await screen.findByLabelText('Exception')).toHaveTextContent('ValueError: lost device');
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

        render(<LogsDialog close={rs.fn()} initialSourceId='job-abc' />);

        expect(await screen.findByText('No logs available')).toBeInTheDocument();
        // The dialog itself (and the rest of the app) must still be intact.
        expect(screen.getByText('Logs')).toBeInTheDocument();
        expect(screen.queryByText('Something went wrong while displaying these logs')).not.toBeInTheDocument();
    });
});

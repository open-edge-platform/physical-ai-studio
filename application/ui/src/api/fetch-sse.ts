// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

interface FetchSSEOptions {
    /** Abort signal used by query cancellation. */
    signal?: AbortSignal;
}

/**
 * Connect to an SSE endpoint and yield parsed messages as an async iterable.
 *
 * Uses the browser-native EventSource API.  The iterable completes when the
 * server sends a `"DONE"` or `"COMPLETED"` data payload, or when the
 * connection errors out.
 */
export function fetchSSE<T = unknown>(url: string, options: FetchSSEOptions = {}) {
    return {
        async *[Symbol.asyncIterator](): AsyncGenerator<T> {
            if (options.signal?.aborted) {
                return;
            }

            const eventSource = new EventSource(url);
            let aborted = false;
            let onAbort: (() => void) | undefined;

            try {
                let { promise, resolve, reject } = Promise.withResolvers<string>();

                onAbort = () => {
                    aborted = true;
                    eventSource.close();
                    resolve('DONE');
                };

                options.signal?.addEventListener('abort', onAbort, { once: true });

                eventSource.onmessage = (event) => {
                    if (event.data === 'DONE' || event.data.includes('COMPLETED')) {
                        eventSource.close();
                        resolve('DONE');
                        return;
                    }
                    resolve(event.data);
                };

                eventSource.onerror = () => {
                    eventSource.close();
                    reject(new Error('EventSource connection failed'));
                };

                while (true) {
                    const message = await promise;

                    if (message === 'DONE') {
                        break;
                    }

                    try {
                        yield JSON.parse(message);
                    } catch {
                        // Skip unparseable messages
                    }

                    ({ promise, resolve, reject } = Promise.withResolvers<string>());
                }
            } finally {
                if (onAbort) {
                    options.signal?.removeEventListener('abort', onAbort);
                }
                if (!aborted && options.signal?.aborted) {
                    aborted = true;
                }
                eventSource.close();
            }
        },
    };
}

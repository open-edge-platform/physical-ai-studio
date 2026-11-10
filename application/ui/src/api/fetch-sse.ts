// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

interface FetchSSEOptions {
    /** Abort signal used by query cancellation. */
    signal?: AbortSignal;
    onIssues?: (data: string) => void;
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
            eventSource.addEventListener('issues', (event) => {
                options.onIssues?.((event as MessageEvent<string>).data);
            });
            const messages: string[] = [];
            let wake = Promise.withResolvers<void>();
            let finished = false;
            let error: Error | undefined;
            const onAbort = () => {
                finished = true;
                messages.length = 0;
                eventSource.close();
                wake.resolve();
            };

            try {
                options.signal?.addEventListener('abort', onAbort, { once: true });

                eventSource.onmessage = (event) => {
                    if (finished) return;
                    if (event.data === 'DONE' || event.data.includes('COMPLETED')) {
                        finished = true;
                        eventSource.close();
                    } else {
                        // Preserve bursts and rows received while the consumer is paused at yield.
                        messages.push(event.data);
                    }
                    wake.resolve();
                };

                eventSource.onerror = () => {
                    if (finished) return;
                    finished = true;
                    error = new Error('EventSource connection failed');
                    eventSource.close();
                    wake.resolve();
                };

                while (!finished || messages.length > 0) {
                    if (messages.length === 0) {
                        await wake.promise;
                        wake = Promise.withResolvers<void>();
                        continue;
                    }
                    const message = messages.shift()!;
                    try {
                        yield JSON.parse(message);
                    } catch {
                        // Skip unparseable messages
                    }
                }
                if (error) throw error;
            } finally {
                options.signal?.removeEventListener('abort', onAbort);
                eventSource.close();
            }
        },
    };
}

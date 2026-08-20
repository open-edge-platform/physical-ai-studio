// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Wraps an async iterable so that items are buffered and re-emitted as
 * arrays ("batches") on a fixed cadence, instead of one at a time.
 *
 * This is used to protect the UI from high-frequency sources (e.g. an SSE
 * stream tailing a file with hundreds of thousands of rows) where consuming
 * every single item individually would trigger one state update/render per
 * item. Batching bounds the number of downstream updates to roughly
 * `sourceDurationMs / intervalMs`, regardless of how many items the source
 * produces or how bursty it is.
 *
 * Any items still buffered when the source completes are flushed as a
 * final batch. Errors from the source are propagated once, after any
 * already-buffered items have been flushed.
 */
export function batchAsyncIterable<T>(source: AsyncIterable<T>, intervalMs: number): AsyncIterable<T[]> {
    return {
        async *[Symbol.asyncIterator](): AsyncGenerator<T[]> {
            let buffer: T[] = [];
            let sourceDone = false;
            let sourceError: unknown;

            const pump = (async () => {
                try {
                    for await (const item of source) {
                        buffer.push(item);
                    }
                } catch (error) {
                    sourceError = error;
                } finally {
                    sourceDone = true;
                }
            })();

            const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

            try {
                while (!sourceDone) {
                    await sleep(intervalMs);

                    if (buffer.length > 0) {
                        const batch = buffer;
                        buffer = [];
                        yield batch;
                    }
                }

                if (buffer.length > 0) {
                    const batch = buffer;
                    buffer = [];
                    yield batch;
                }

                if (sourceError) {
                    throw sourceError;
                }
            } finally {
                // Ensure the source's own cleanup (e.g. closing an EventSource) has
                // run before we consider this generator fully done.
                await pump;
            }
        },
    };
}

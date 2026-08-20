// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { batchAsyncIterable } from './batch-async-iterable';

const collect = async <T>(iterable: AsyncIterable<T>): Promise<T[]> => {
    const results: T[] = [];

    for await (const item of iterable) {
        results.push(item);
    }

    return results;
};

// A source that yields items with delays so we can control when they land
// relative to the batching interval.
async function* timedSource<T>(items: Array<{ value: T; delayMs: number }>): AsyncGenerator<T> {
    for (const { value, delayMs } of items) {
        if (delayMs > 0) {
            await new Promise((resolve) => setTimeout(resolve, delayMs));
        }
        yield value;
    }
}

describe('batchAsyncIterable', () => {
    it('groups items produced within one interval window into a single batch', async () => {
        const source = timedSource([
            { value: 1, delayMs: 0 },
            { value: 2, delayMs: 0 },
            { value: 3, delayMs: 0 },
        ]);

        const batches = await collect(batchAsyncIterable(source, 50));

        expect(batches).toEqual([[1, 2, 3]]);
    });

    it('produces a separate batch per interval window when items are spread out over time', async () => {
        const source = timedSource([
            { value: 1, delayMs: 0 },
            { value: 2, delayMs: 60 },
            { value: 3, delayMs: 60 },
        ]);

        const batches = await collect(batchAsyncIterable(source, 20));

        expect(batches).toEqual([[1], [2], [3]]);
    });

    it('flushes any items buffered right before the source completes', async () => {
        const source = timedSource([
            { value: 1, delayMs: 0 },
            { value: 2, delayMs: 0 },
        ]);

        const batches = await collect(batchAsyncIterable(source, 1000));

        expect(batches).toEqual([[1, 2]]);
    });

    it('yields nothing for a source that produces no items', async () => {
        const source = timedSource<number>([]);

        const batches = await collect(batchAsyncIterable(source, 20));

        expect(batches).toEqual([]);
    });

    it('propagates an error from the source after flushing already-buffered items', async () => {
        const source = (async function* () {
            yield 1;
            yield 2;
            throw new Error('stream failed');
        })();

        const batches: number[][] = [];
        const iterator = batchAsyncIterable(source, 1000);

        await expect(
            (async () => {
                for await (const batch of iterator) {
                    batches.push(batch);
                }
            })()
        ).rejects.toThrow('stream failed');

        expect(batches).toEqual([[1, 2]]);
    });
});

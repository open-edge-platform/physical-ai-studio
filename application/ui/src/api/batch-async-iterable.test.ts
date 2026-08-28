// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { vi } from 'vitest';

import { batchAsyncIterable } from './batch-async-iterable';

// A source that yields items with delays so we can control when they land
// relative to the batching interval. Delays are relative to the previous item.
async function* timedSource<T>(items: Array<{ value: T; delayMs: number }>): AsyncGenerator<T> {
    for (const { value, delayMs } of items) {
        if (delayMs > 0) {
            await new Promise((resolve) => setTimeout(resolve, delayMs));
        }
        yield value;
    }
}

/**
 * Starts consuming `iterable` without awaiting it, then drives the fake clock
 * forward by `elapsedMs` so the buffered timers resolve deterministically.
 */
const collectWhileAdvancing = async <T>(iterable: AsyncIterable<T>, elapsedMs: number): Promise<T[]> => {
    const results: T[] = [];

    const consumed = (async () => {
        for await (const item of iterable) {
            results.push(item);
        }
    })();

    await vi.advanceTimersByTimeAsync(elapsedMs);
    await consumed;

    return results;
};

describe('batchAsyncIterable', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('groups items produced within one interval window into a single batch', async () => {
        const source = timedSource([
            { value: 1, delayMs: 0 },
            { value: 2, delayMs: 0 },
            { value: 3, delayMs: 0 },
        ]);

        const batches = await collectWhileAdvancing(batchAsyncIterable(source, 100), 300);

        expect(batches).toEqual([[1, 2, 3]]);
    });

    it('produces a separate batch per interval window when items are spread out over time', async () => {
        // Items land at t=0, t=250 and t=450, so each falls into a different
        // 100ms window and none collides with a window boundary.
        const source = timedSource([
            { value: 1, delayMs: 0 },
            { value: 2, delayMs: 250 },
            { value: 3, delayMs: 200 },
        ]);

        const batches = await collectWhileAdvancing(batchAsyncIterable(source, 100), 1000);

        expect(batches).toEqual([[1], [2], [3]]);
    });

    it('flushes any items buffered right before the source completes', async () => {
        const source = timedSource([
            { value: 1, delayMs: 0 },
            { value: 2, delayMs: 0 },
        ]);

        const batches = await collectWhileAdvancing(batchAsyncIterable(source, 1000), 2000);

        expect(batches).toEqual([[1, 2]]);
    });

    it('yields nothing for a source that produces no items', async () => {
        const source = timedSource<number>([]);

        const batches = await collectWhileAdvancing(batchAsyncIterable(source, 100), 300);

        expect(batches).toEqual([]);
    });

    it('propagates an error from the source after flushing already-buffered items', async () => {
        const source = (async function* () {
            yield 1;
            yield 2;
            throw new Error('stream failed');
        })();

        const batches: number[][] = [];
        const consumed = (async () => {
            for await (const batch of batchAsyncIterable(source, 100)) {
                batches.push(batch);
            }
        })();
        const assertion = expect(consumed).rejects.toThrow('stream failed');

        await vi.advanceTimersByTimeAsync(300);
        await assertion;

        expect(batches).toEqual([[1, 2]]);
    });
});

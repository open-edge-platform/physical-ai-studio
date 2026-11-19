// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { rs } from '@rstest/core';
import { QueryClient } from '@tanstack/react-query';
import { act } from '@testing-library/react';

import { getQueryKey } from '../../../query-client/query-client.interface';
import { render } from '../../../test-utils/render';
import { JobMetricsContent } from './metrics';
import type { MetricsEntry } from './types';

class TestEventSource extends EventTarget {
    static instances: TestEventSource[] = [];
    onmessage: ((event: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;
    close = rs.fn();

    constructor() {
        super();
        TestEventSource.instances.push(this);
    }
}

const queryKey = getQueryKey(['get', '/api/jobs/{job_id}/model_metrics', { params: { path: { job_id: 'job-1' } } }]);
const advance = async (ms: number) => {
    await act(async () => {
        await rs.advanceTimersByTimeAsync(ms);
    });
};

describe('live job metrics', () => {
    let queryClient: QueryClient;

    beforeEach(() => {
        rs.useFakeTimers();
        TestEventSource.instances = [];
        rs.stubGlobal('EventSource', TestEventSource);
        queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    });

    afterEach(() => {
        queryClient.clear();
        rs.useRealTimers();
        rs.unstubAllGlobals();
    });

    it('reconnects when metrics are reopened instead of treating cached rows as permanently fresh', async () => {
        const firstView = render(<JobMetricsContent jobId='job-1' />, { queryClient });
        await advance(0);
        TestEventSource.instances[0].onmessage?.({ data: '{"step":49,"train_loss":0.8}' });
        await advance(250);
        expect(queryClient.getQueryData<Map<number, MetricsEntry>>(queryKey)?.get(49)?.train_loss).toBe(0.8);
        firstView.unmount();
        await advance(250);

        const secondView = render(<JobMetricsContent jobId='job-1' />, { queryClient });
        await advance(0);
        expect(TestEventSource.instances).toHaveLength(2);
        TestEventSource.instances[1].onmessage?.({ data: '{"step":99,"train_loss":0.4}' });
        await advance(250);
        expect(queryClient.getQueryData<Map<number, MetricsEntry>>(queryKey)?.get(99)?.train_loss).toBe(0.4);
        secondView.unmount();
        await advance(250);
    });

    it.each(['completed', 'failed'])(
        'retries a %s stream without interrupting an active connection',
        async (status) => {
            const view = render(<JobMetricsContent jobId='job-1' />, { queryClient });
            await advance(0);
            if (status === 'failed') {
                TestEventSource.instances[0].onerror?.();
            } else {
                TestEventSource.instances[0].onmessage?.({ data: 'DONE' });
            }
            await advance(250);
            await advance(5000);
            expect(TestEventSource.instances).toHaveLength(2);
            const source = TestEventSource.instances[1];
            source.onmessage?.({ data: '{"step":149,"train_loss":0.2}' });
            await advance(250);
            expect(queryClient.getQueryData<Map<number, MetricsEntry>>(queryKey)?.get(149)?.train_loss).toBe(0.2);
            await advance(10000);
            expect(TestEventSource.instances).toHaveLength(2);
            expect(source.close).not.toHaveBeenCalled();
            view.unmount();
            await advance(250);
        }
    );
});

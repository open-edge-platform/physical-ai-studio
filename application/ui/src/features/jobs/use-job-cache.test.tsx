// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';

import { createQueryClient } from '../../query-client/query-client';
import { getMockedTrainJob } from '../../test-utils/mocks/mock-train-job';
import { renderHook } from '../../test-utils/render';
import { JOBS_QUERY_KEY, SchemaJob, useJobCache } from './use-job-cache';

describe('useJobCache', () => {
    it('inserts a newly submitted job immediately', () => {
        const queryClient = createQueryClient();
        const existing = getMockedTrainJob({ id: 'job-1' });
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, [existing]);

        const { result } = renderHook(() => useJobCache(), { queryClient });
        const newJob = getMockedTrainJob({ id: 'job-2' });

        act(() => result.current.addJob(newJob));

        expect(queryClient.getQueryData<SchemaJob[]>(JOBS_QUERY_KEY)).toEqual([existing, newJob]);
    });

    it('does not duplicate a job that already exists, updating it in place instead', () => {
        const queryClient = createQueryClient();
        const original = getMockedTrainJob({ id: 'job-1', status: 'pending', progress: 0 });
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, [original]);

        const { result } = renderHook(() => useJobCache(), { queryClient });
        const updated = getMockedTrainJob({ id: 'job-1', status: 'running', progress: 50 });

        act(() => result.current.addJob(updated));

        const jobs = queryClient.getQueryData<SchemaJob[]>(JOBS_QUERY_KEY);
        expect(jobs).toHaveLength(1);
        expect(jobs?.[0]).toEqual(updated);
    });

    it('does not seed cache data before the initial fetch has populated it', () => {
        const queryClient = createQueryClient();
        // No initial data set — mirrors a query that has not fetched yet.

        const { result } = renderHook(() => useJobCache(), { queryClient });

        act(() => result.current.addJob(getMockedTrainJob({ id: 'job-1' })));

        expect(queryClient.getQueryData<SchemaJob[]>(JOBS_QUERY_KEY)).toBeUndefined();
    });
});

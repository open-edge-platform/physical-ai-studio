// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';
import { vi } from 'vitest';

import { createQueryClient } from '../../query-client/query-client';
import { getMockedDatasetImportJob } from '../../test-utils/mocks/mock-dataset-import-job';
import { getMockedTrainJob } from '../../test-utils/mocks/mock-train-job';
import { renderHook } from '../../test-utils/render';
import { JOBS_QUERY_KEY, SchemaJob } from './use-job-cache';
import { useJobUpdates } from './use-job-updates';

const notify = vi.fn();
vi.mock('../../components/notification/notification.component', () => ({
    notify: (...args: unknown[]) => notify(...args),
}));

let capturedOnMessage: ((event: { data: string }) => void) | undefined;
let capturedOnOpen: (() => void) | undefined;

vi.mock('react-use-websocket', () => ({
    default: (_url: unknown, options: { onMessage?: (event: { data: string }) => void; onOpen?: () => void }) => {
        capturedOnMessage = options.onMessage;
        capturedOnOpen = options.onOpen;
        return { readyState: 1 };
    },
}));

const deliver = (job: SchemaJob) => {
    act(() => {
        capturedOnMessage?.({ data: JSON.stringify({ event: 'JOB_UPDATE', data: job }) });
    });
};

const PROJECT_ID = 'project-1';
const OTHER_PROJECT_ID = 'project-2';

const modelsQueryKey = (project_id: string) =>
    ['get', '/api/projects/{project_id}/models', { params: { path: { project_id } } }] as const;

describe('useJobUpdates', () => {
    beforeEach(() => {
        notify.mockClear();
        capturedOnMessage = undefined;
        capturedOnOpen = undefined;
    });

    it('makes a previously unseen job visible in the shared cache', () => {
        const queryClient = createQueryClient();
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, []);

        renderHook(() => useJobUpdates(PROJECT_ID), { queryClient });

        const job = getMockedTrainJob({ id: 'job-new', project_id: PROJECT_ID });
        deliver(job);

        expect(queryClient.getQueryData<SchemaJob[]>(JOBS_QUERY_KEY)).toEqual([job]);
    });

    it('is idempotent by job id: an update for an already-cached job never duplicates it', () => {
        const queryClient = createQueryClient();
        const job = getMockedTrainJob({ id: 'job-1', project_id: PROJECT_ID, status: 'running', progress: 10 });
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, [job]);

        renderHook(() => useJobUpdates(PROJECT_ID), { queryClient });

        deliver({ ...job, progress: 55 });

        const jobs = queryClient.getQueryData<SchemaJob[]>(JOBS_QUERY_KEY);
        expect(jobs).toHaveLength(1);
        expect(jobs?.[0].progress).toBe(55);
    });

    it('does not let an event arriving before the initial fetch seed a partial cache', () => {
        const queryClient = createQueryClient();

        renderHook(() => useJobUpdates(PROJECT_ID), { queryClient });

        deliver(getMockedTrainJob({ id: 'job-1', project_id: PROJECT_ID }));

        expect(queryClient.getQueryData<SchemaJob[]>(JOBS_QUERY_KEY)).toBeUndefined();
    });

    it('keeps dataset-import entries intact in the shared cache', () => {
        const queryClient = createQueryClient();
        const importJob = getMockedDatasetImportJob({ id: 'import-1', project_id: PROJECT_ID });
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, [importJob]);

        renderHook(() => useJobUpdates(PROJECT_ID), { queryClient });

        const updatedImportJob = { ...importJob, progress: 80 };
        deliver(updatedImportJob);

        expect(queryClient.getQueryData<SchemaJob[]>(JOBS_QUERY_KEY)).toEqual([updatedImportJob]);
        expect(notify).not.toHaveBeenCalled();
    });

    it('notifies for a running job in the current project', () => {
        const queryClient = createQueryClient();
        const job = getMockedTrainJob({ id: 'job-1', project_id: PROJECT_ID });
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, [job]);

        renderHook(() => useJobUpdates(PROJECT_ID), { queryClient });

        deliver({ ...job, status: 'running', message: 'Training started' });

        expect(notify).toHaveBeenCalledWith('info', 'Training started');
    });

    it('does not notify about another project on a project-scoped page', () => {
        const queryClient = createQueryClient();
        const job = getMockedTrainJob({ id: 'job-1', project_id: OTHER_PROJECT_ID });
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, [job]);

        renderHook(() => useJobUpdates(PROJECT_ID), { queryClient });

        deliver({ ...job, status: 'running', message: 'Training started' });

        expect(notify).not.toHaveBeenCalled();
    });

    it('does not notify on a global (non-project) page', () => {
        const queryClient = createQueryClient();
        const job = getMockedTrainJob({ id: 'job-1', project_id: PROJECT_ID });
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, [job]);

        renderHook(() => useJobUpdates(undefined), { queryClient });

        deliver({ ...job, status: 'running', message: 'Training started' });

        expect(notify).not.toHaveBeenCalled();
    });

    it("invalidates the completed job's own project models query, not other projects'", () => {
        const queryClient = createQueryClient();
        const job = getMockedTrainJob({ id: 'job-1', project_id: OTHER_PROJECT_ID, status: 'running' });
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, [job]);

        const otherProjectModels = modelsQueryKey(OTHER_PROJECT_ID);
        const currentProjectModels = modelsQueryKey(PROJECT_ID);
        queryClient.setQueryData(otherProjectModels, []);
        queryClient.setQueryData(currentProjectModels, []);

        renderHook(() => useJobUpdates(PROJECT_ID), { queryClient });

        deliver({ ...job, status: 'completed' });

        expect(queryClient.getQueryState(otherProjectModels)?.isInvalidated).toBe(true);
        expect(queryClient.getQueryState(currentProjectModels)?.isInvalidated).toBe(false);
    });

    it('refetches jobs whenever the socket (re)connects', () => {
        const queryClient = createQueryClient();
        queryClient.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, []);

        renderHook(() => useJobUpdates(PROJECT_ID), { queryClient });

        act(() => {
            capturedOnOpen?.();
        });

        expect(queryClient.getQueryState(JOBS_QUERY_KEY)?.isInvalidated).toBe(true);
    });
});

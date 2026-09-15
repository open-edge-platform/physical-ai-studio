// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';

import { SchemaDatasetImportJob, SchemaTrainJob } from '../../api/openapi-spec';

export type SchemaJob = SchemaTrainJob | SchemaDatasetImportJob;

export const JOBS_QUERY_KEY = ['get', '/api/jobs'] as const;

/**
 * Subscription-free helper for writing a single job into the shared `/api/jobs`
 * query cache. Used both by callers that just submitted a new job (e.g.
 * `ModelsPage`) and by `useJobUpdates`'s websocket handler.
 *
 * Upserts by job id, so inserting a job and later receiving a websocket
 * confirmation for the same id never produces a duplicate row.
 *
 * If the cache has not been populated by the initial HTTP fetch yet, the
 * write is skipped rather than seeding a partial "apparently complete"
 * result — the upcoming initial fetch remains authoritative.
 */
export const useJobCache = () => {
    const client = useQueryClient();

    const addJob = (job: SchemaJob) => {
        client.setQueryData<SchemaJob[]>(JOBS_QUERY_KEY, (old) => {
            if (old === undefined) {
                return old;
            }

            const exists = old.some((existing) => existing.id === job.id);

            if (exists) {
                return old.map((existing) => (existing.id === job.id ? job : existing));
            }

            return [...old, job];
        });
    };

    return { addJob };
};

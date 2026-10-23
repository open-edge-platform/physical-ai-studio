// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';
import { notify } from '../../components/notification/notification.component';
import { JOBS_QUERY_KEY, SchemaJob, useJobCache } from './use-job-cache';

/**
 * Owns the single jobs websocket subscription. Mounted once from `AppFooter`
 * so live updates keep flowing regardless of which page is active or whether
 * any jobs dialog is open.
 *
 * Keeps the shared `/api/jobs` query cache in sync with `JOB_UPDATE` events,
 * invalidates the affected project's models when training or optimization
 * completes, and refreshes jobs and model caches on every (re)connect so
 * changes missed while disconnected are recovered.
 *
 * `project_id` scopes the "job is running" toast to the current project; pass
 * `undefined` on global (non-project) pages to suppress that notification —
 * other projects' jobs still update the shared cache.
 */
export const useJobUpdates = (project_id?: string) => {
    const client = useQueryClient();
    const { addJob } = useJobCache();

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message_data = JSON.parse(data);

        if (message_data.event !== 'JOB_UPDATE') {
            return;
        }

        const job = (message_data as { event: string; data: SchemaJob }).data;

        addJob(job);

        if (job.type !== 'training' && job.type !== 'optimize_model') {
            return;
        }

        const isCurrentProject = project_id !== undefined && job.project_id === project_id;

        if (job.type === 'training' && isCurrentProject && job.message && job.status === 'running') {
            notify('info', job.message);
        }

        if (job.status === 'completed') {
            client.invalidateQueries({
                queryKey: [
                    'get',
                    '/api/projects/{project_id}/models',
                    { params: { path: { project_id: job.project_id } } },
                ],
            });
            if (job.type === 'optimize_model') {
                for (const model_id of [job.payload.model_id, job.id]) {
                    client.invalidateQueries({
                        queryKey: ['get', '/api/models/{model_id}', { params: { path: { model_id } } }],
                    });
                }
            }
        }
    };

    const onOpen = () => {
        for (const queryKey of [
            JOBS_QUERY_KEY,
            ['get', '/api/projects/{project_id}/models'],
            ['get', '/api/models/{model_id}'],
        ]) {
            client.invalidateQueries({ queryKey });
        }
    };

    useWebSocket(fetchClient.PATH('/api/jobs/ws'), {
        shouldReconnect: () => true,
        onMessage,
        onOpen,
    });
};

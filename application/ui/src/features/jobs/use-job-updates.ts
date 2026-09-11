// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';
import { SchemaTrainJob } from '../../api/openapi-spec';
import { notify } from '../../components/notification/notification.component';
import { JOBS_QUERY_KEY, SchemaJob, useJobCache } from './use-job-cache';

const isTrainJob = (job: SchemaJob): job is SchemaTrainJob => job.type === 'training';

/**
 * Owns the single jobs websocket subscription. Mounted once from `AppFooter`
 * so live updates keep flowing regardless of which page is active or whether
 * any jobs dialog is open.
 *
 * Keeps the shared `/api/jobs` query cache (training and dataset-import jobs
 * alike) in sync with `JOB_UPDATE` events, invalidates the affected project's
 * models query when a training job completes, and refetches the jobs list on
 * every (re)connect so changes missed while disconnected are recovered.
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

        if (!isTrainJob(job)) {
            return;
        }

        const isCurrentProject = project_id !== undefined && job.project_id === project_id;

        if (isCurrentProject && job.message && job.status === 'running') {
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
        }
    };

    const onOpen = () => {
        client.invalidateQueries({ queryKey: JOBS_QUERY_KEY });
    };

    useWebSocket(fetchClient.PATH('/api/jobs/ws'), {
        shouldReconnect: () => true,
        onMessage,
        onOpen,
    });
};

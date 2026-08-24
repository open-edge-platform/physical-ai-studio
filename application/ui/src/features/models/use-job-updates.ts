import { useQueryClient } from '@tanstack/react-query';
import useWebSocket from 'react-use-websocket';

import { fetchClient } from '../../api/client';
import { SchemaTrainJob as SchemaJob } from '../../api/openapi-spec';
import { notify } from '../../components/notification/notification.component';

/**
 * Subscribes to the jobs websocket for a project, keeping the `/api/jobs` query cache
 * in sync with `JOB_UPDATE` events, and returns `addJob` for optimistically inserting
 * a freshly submitted job into the cache.
 */
export const useJobUpdates = (project_id: string) => {
    const client = useQueryClient();

    const updateJob = (job: SchemaJob) => {
        client.setQueryData<SchemaJob[]>(['get', '/api/jobs'], (old = []) => {
            return old.map((m) => (m.id === job.id ? job : m));
        });
    };

    const addJob = (job: SchemaJob) => {
        client.setQueryData<SchemaJob[]>(['get', '/api/jobs'], (old = []) => {
            return [...old, job];
        });
    };

    const onMessage = ({ data }: WebSocketEventMap['message']) => {
        const message_data = JSON.parse(data);
        if (message_data.event === 'JOB_UPDATE') {
            const message = message_data as { event: string; data: SchemaJob };
            if (message.data.project_id !== project_id) {
                return;
            }

            updateJob(message.data);

            if (message.data.message && message.data.status === 'running') {
                notify('info', message.data.message);
            }

            if (message.data.status === 'completed') {
                client.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/models'] });
            }
        }
    };

    useWebSocket(fetchClient.PATH('/api/jobs/ws'), {
        shouldReconnect: () => true,
        onMessage,
    });

    return { addJob };
};

import { Button, Flex, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaModelExportAction, SchemaModelExportJob } from '../../../api/openapi-spec';
import { JOBS_QUERY_KEY, useJobCache } from '../../jobs/use-job-cache';

export const ExportJobControls = ({ modelId, action }: { modelId: string; action: SchemaModelExportAction }) => {
    const jobs = $api.useQuery('get', '/api/jobs');
    const { addJob } = useJobCache();
    const exportModel = $api.useMutation('post', '/api/models/{model_id}/exports/{backend}', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onSuccess: addJob,
    });
    const interrupt = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
    });
    const matching = (jobs.data ?? [])
        .filter(
            (job): job is SchemaModelExportJob =>
                job.type === 'model_export' &&
                job.payload.model_id === modelId &&
                job.payload.backend === action.backend
        )
        .toSorted((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
    const active = (jobs.data ?? []).find(
        (job) =>
            job.type === 'model_export' &&
            job.payload.model_id === modelId &&
            (job.status === 'pending' || job.status === 'running')
    );
    const latest = active ?? matching[0];
    const error = exportModel.error ?? interrupt.error;

    return (
        <Flex direction='column' gap='size-100'>
            <Text>Re-export the stored training checkpoint. This can take several minutes and additional memory.</Text>
            <Flex gap='size-100'>
                <Button
                    variant='secondary'
                    isPending={exportModel.isPending}
                    isDisabled={jobs.isPending || jobs.isError || active !== undefined}
                    onPress={() => {
                        interrupt.reset();
                        exportModel.mutate({ params: { path: { model_id: modelId, backend: action.backend } } });
                    }}
                >
                    Export model
                </Button>
                {active?.id && (
                    <Button
                        variant='secondary'
                        isPending={interrupt.isPending}
                        onPress={() => interrupt.mutate({ params: { path: { job_id: active.id! } } })}
                    >
                        Cancel job
                    </Button>
                )}
            </Flex>
            {latest && (
                <div role='status'>
                    Export {latest.status}: {latest.message}
                </div>
            )}
            {error && <div role='alert'>{getApiErrorMessage(error) ?? 'Export failed. Please try again.'}</div>}
            {jobs.isError && (
                <Button variant='secondary' onPress={() => jobs.refetch()}>
                    Retry job status
                </Button>
            )}
        </Flex>
    );
};

import { Button, Flex, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaOptimizeModelJob } from '../../../api/openapi-spec';
import { JOBS_QUERY_KEY, useJobCache } from '../../jobs/use-job-cache';

export const OptimizeModel = ({ modelId, hasInt8 }: { modelId: string; hasInt8: boolean }) => {
    const jobs = $api.useQuery('get', '/api/jobs');
    const { addJob } = useJobCache();
    const optimize = $api.useMutation('post', '/api/models/{model_id}/optimize', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onSuccess: addJob,
    });
    const interrupt = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
    });
    const matching = (jobs.data ?? [])
        .filter(
            (job): job is SchemaOptimizeModelJob => job.type === 'optimize_model' && job.payload.model_id === modelId
        )
        .toSorted((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
    const active = matching.find((job) => job.status === 'pending' || job.status === 'running');
    const error = optimize.error ?? interrupt.error;

    return (
        <Flex direction='column' gap='size-100'>
            <Text>INT8 compresses weights, not activations. The original export stays unchanged.</Text>
            <Flex gap='size-100'>
                <Button
                    variant='secondary'
                    isPending={optimize.isPending}
                    isDisabled={hasInt8 || jobs.isPending || jobs.isError || active !== undefined}
                    onPress={() => {
                        interrupt.reset();
                        optimize.mutate({ params: { path: { model_id: modelId } } });
                    }}
                >
                    Optimize to INT8
                </Button>
                {active?.id && (
                    <Button
                        variant='secondary'
                        isPending={interrupt.isPending}
                        onPress={() => interrupt.mutate({ params: { path: { job_id: active.id! } } })}
                    >
                        Cancel optimization
                    </Button>
                )}
            </Flex>
            {matching[0] && (
                <div role='status'>
                    Optimization {matching[0].status}: {matching[0].message}
                </div>
            )}
            {error && <div role='alert'>{getApiErrorMessage(error) ?? 'Optimization failed. Please try again.'}</div>}
            {jobs.isError && (
                <Button variant='secondary' onPress={() => jobs.refetch()}>
                    Retry optimization status
                </Button>
            )}
        </Flex>
    );
};

import { Button, Flex, ProgressBar, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaOptimizeModelJob } from '../../../api/openapi-spec';
import { JOBS_QUERY_KEY, useJobCache } from '../../jobs/use-job-cache';

const isActive = (job: SchemaOptimizeModelJob) => job.status === 'pending' || job.status === 'running';

export const OptimizeModel = ({ modelId }: { modelId: string }) => {
    const { data: jobs = [], isPending, isError, refetch } = $api.useQuery('get', '/api/jobs');
    const { addJob } = useJobCache();
    const optimize = $api.useMutation('post', '/api/models/{model_id}/optimize', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
        onSuccess: addJob,
    });
    const interrupt = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
    });

    const matchingJobs = jobs
        .filter(
            (job): job is SchemaOptimizeModelJob => job.type === 'optimize_model' && job.payload.model_id === modelId
        )
        .toSorted((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''));
    const activeJob = matchingJobs.findLast(isActive);
    const latestJob = matchingJobs.at(-1);

    return (
        <Flex direction='column' gap='size-100'>
            <Text>
                Compresses weights to symmetric INT8, not activations. The original model remains available. Validate
                policy accuracy before deployment.
            </Text>
            <Flex gap='size-100'>
                <Button
                    variant='secondary'
                    isDisabled={isPending || isError || optimize.isPending || activeJob !== undefined}
                    onPress={() => {
                        interrupt.reset();
                        optimize.mutate({ params: { path: { model_id: modelId } } });
                    }}
                >
                    Optimize to INT8
                </Button>
                {activeJob?.id && (
                    <Button
                        variant='secondary'
                        isDisabled={interrupt.isPending}
                        onPress={() => interrupt.mutate({ params: { path: { job_id: activeJob.id! } } })}
                    >
                        Cancel optimization
                    </Button>
                )}
            </Flex>
            {latestJob && (
                <div role='status'>
                    <Flex direction='column' gap='size-100'>
                        <Text>
                            Optimization {latestJob.status}: {latestJob.message}
                        </Text>
                        {isActive(latestJob) && (
                            <ProgressBar aria-label='Optimization progress' value={latestJob.progress} showValueLabel />
                        )}
                    </Flex>
                </div>
            )}
            {optimize.isError && (
                <div role='alert'>
                    <Text>
                        Could not start optimization. {getApiErrorMessage(optimize.error) ?? 'Please try again.'}
                    </Text>
                </div>
            )}
            {interrupt.isError && (
                <div role='alert'>
                    <Text>
                        Could not cancel optimization. {getApiErrorMessage(interrupt.error) ?? 'Please try again.'}
                    </Text>
                </div>
            )}
            {isError && (
                <Flex gap='size-100' alignItems='center'>
                    <div role='alert'>
                        <Text>Could not load optimization status.</Text>
                    </div>
                    <Button variant='secondary' onPress={() => refetch()}>
                        Retry status
                    </Button>
                </Flex>
            )}
        </Flex>
    );
};

import { Button, Flex, ProgressBar, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { getApiErrorMessage } from '../../../api/errors';
import { SchemaModelExportJob } from '../../../api/openapi-spec';
import { JOBS_QUERY_KEY } from '../../jobs/use-job-cache';

interface ExportJobControlsProps {
    label: string;
    job?: SchemaModelExportJob;
    isSubmitting: boolean;
    error?: unknown;
    onRetry?: () => void;
    isRetryDisabled: boolean;
}

const getExportStatus = (
    job: SchemaModelExportJob | undefined,
    isSubmitting: boolean,
    isCanceling: boolean
): string => {
    if (isSubmitting) return 'Requesting export…';
    if (isCanceling && (job?.status === 'pending' || job?.status === 'running')) return 'Canceling export…';

    switch (job?.status) {
        case 'pending':
            return 'Export queued';
        case 'running':
            return 'Exporting variant…';
        case 'completed':
            return 'The exported variant is no longer available.';
        case 'canceled':
            return 'Export canceled';
        default:
            return 'Export failed';
    }
};

export const ExportJobControls = ({
    label,
    job,
    isSubmitting,
    error,
    onRetry,
    isRetryDisabled,
}: ExportJobControlsProps) => {
    const interrupt = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [[...JOBS_QUERY_KEY]] },
    });
    const isActive = job?.status === 'pending' || job?.status === 'running';
    const isCanceling = interrupt.isSuccess || job?.payload.cancellation_requested === true;
    const failure = error ?? interrupt.error;
    const status = getExportStatus(job, isSubmitting, isCanceling);

    return (
        <Flex direction='column' gap='size-100'>
            <div role='status'>
                <Text>{status}</Text>
                {!isSubmitting && job?.message && <Text> — {job.message}</Text>}
            </div>
            {(isSubmitting || isActive) && (
                <>
                    <ProgressBar
                        aria-label={`${label} export progress`}
                        size='S'
                        width='100%'
                        isIndeterminate={isSubmitting || job?.status === 'pending' || job?.progress === 0}
                        value={job?.progress ?? 0}
                    />
                    <Text>Exporting the stored checkpoint can take several minutes.</Text>
                </>
            )}
            {isActive && job?.id && (
                <Button
                    variant='secondary'
                    isPending={interrupt.isPending}
                    isDisabled={isCanceling}
                    onPress={() => interrupt.mutate({ params: { path: { job_id: job.id! } } })}
                >
                    Cancel export
                </Button>
            )}
            {!isSubmitting && !isActive && onRetry && (
                <Button variant='secondary' isDisabled={isRetryDisabled} onPress={onRetry}>
                    Retry export
                </Button>
            )}
            {failure && <div role='alert'>{getApiErrorMessage(failure) ?? 'Export failed. Please try again.'}</div>}
        </Flex>
    );
};

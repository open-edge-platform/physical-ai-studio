import { Heading, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaTrainJob } from '../train-model-dialog/train-model-dialog';
import { TrainingHeader, TrainingRow } from './job-table';

interface JobListProps {
    jobs: SchemaTrainJob[];
    onViewLogs: (job: SchemaTrainJob) => void;
}

export const JobList = ({ jobs, onViewLogs }: JobListProps) => {
    const sortedJobs = jobs
        .filter((m) => m.status !== 'completed')
        .toSorted((a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime());

    const interruptMutation = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: {
            invalidates: [['get', '/api/jobs']],
        },
    });
    const onInterrupt = (job: SchemaTrainJob) => {
        if (job.id !== undefined) {
            interruptMutation.mutate({
                params: {
                    path: {
                        job_id: job.id,
                    },
                },
            });
        }
    };

    if (sortedJobs.length === 0) {
        return <></>;
    }

    return (
        <View marginBottom={'size-600'}>
            <Heading level={4} marginBottom={'size-100'}>
                Current Training
            </Heading>

            <TrainingHeader />
            {sortedJobs.map((job) => (
                <TrainingRow
                    key={job.id}
                    trainJob={job}
                    onInterrupt={() => onInterrupt(job)}
                    onViewLogs={() => onViewLogs(job)}
                />
            ))}
        </View>
    );
};

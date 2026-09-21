import { Heading, View } from '@geti-ui/ui';

import { SchemaTrainJob } from '../train-model-dialog/train-model-dialog';
import { TrainingJobsTable } from './training-jobs-table';

interface JobListProps {
    jobs: SchemaTrainJob[];
    onViewLogs: (job: SchemaTrainJob) => void;
}

export const JobList = ({ jobs, onViewLogs }: JobListProps) => {
    const activeJobs = jobs.filter((job) => job.status === 'running' || job.status === 'pending');

    if (activeJobs.length === 0) {
        return <></>;
    }

    return (
        <View>
            <Heading level={4} marginBottom={'size-100'}>
                Current Training
            </Heading>

            <TrainingJobsTable jobs={activeJobs} onViewLogs={onViewLogs} />
        </View>
    );
};

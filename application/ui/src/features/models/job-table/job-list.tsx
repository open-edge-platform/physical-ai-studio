import { Heading, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { Table, TableColumn } from '../../../components/table/table';
import { SchemaTrainJob } from '../train-model-dialog/train-model-dialog';
import { TrainingRow } from './job-table';

const JOB_COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: '2fr', header: 'Model name' },
    { width: '1fr', header: 'Loss' },
    { width: '1fr' },
    { width: '1fr', header: 'Architecture' },
    { width: '1fr' },
    { width: 'auto', align: 'end' },
];

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
        <View>
            <Heading level={4} marginBottom={'size-100'}>
                Current Training
            </Heading>

            <Table columns={JOB_COLUMNS}>
                {sortedJobs.map((job) => (
                    <TrainingRow
                        key={job.id}
                        trainJob={job}
                        onInterrupt={() => onInterrupt(job)}
                        onViewLogs={() => onViewLogs(job)}
                    />
                ))}
            </Table>
        </View>
    );
};

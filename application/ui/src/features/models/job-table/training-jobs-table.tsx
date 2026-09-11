import { ReactNode } from 'react';

import { $api } from '../../../api/client';
import { SchemaTrainJob } from '../../../api/openapi-spec';
import { Table, TableColumn } from '../../../components/table/table';
import { TrainingRow } from './job-table';

const JOB_COLUMNS: TableColumn[] = [
    { width: 'max-content' },
    { width: '2fr', header: 'Model name' },
    { width: '1fr', header: 'Loss' },
    { width: '1fr', header: 'Architecture' },
    { width: '1fr', header: 'Dataset' },
    { width: '1fr', header: 'Environment' },
    { width: '1fr', header: 'Trainer' },
    { width: '1fr' },
    { width: 'auto', align: 'end' },
];

const getTimestamp = (job: SchemaTrainJob): number => {
    if (!job.created_at) return 0;
    const time = new Date(job.created_at).getTime();
    return Number.isNaN(time) ? 0 : time;
};

interface TrainingJobsTableProps {
    jobs: SchemaTrainJob[];
    onViewLogs: (job: SchemaTrainJob) => void;
    emptyMessage?: ReactNode;
}

export const TrainingJobsTable = ({ jobs, onViewLogs, emptyMessage }: TrainingJobsTableProps) => {
    const sortedJobs = jobs.toSorted((a, b) => getTimestamp(b) - getTimestamp(a));

    const interruptMutation = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [['get', '/api/jobs']] },
    });
    const onInterrupt = (job: SchemaTrainJob) => {
        if (job.id !== undefined) {
            interruptMutation.mutate({ params: { path: { job_id: job.id } } });
        }
    };

    if (sortedJobs.length === 0) {
        return <>{emptyMessage ?? null}</>;
    }

    return (
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
    );
};

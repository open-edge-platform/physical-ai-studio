import { Content, Dialog, Divider, Heading, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { Table, TableColumn } from '../../../components/table/table';
import { SchemaTrainJob } from '../train-model-dialog/train-model-dialog';
import { TrainingRow } from './job-table';

// Lists every training job regardless of status (running, pending, completed,
// failed, canceled). Terminal jobs live here rather than in the always-visible
// "Current Training" section.
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

interface AllJobsDialogProps {
    jobs: SchemaTrainJob[];
    onViewLogs: (job: SchemaTrainJob) => void;
    close: () => void;
}

export const AllJobsDialog = ({ jobs, onViewLogs, close }: AllJobsDialogProps) => {
    const sortedJobs = jobs.toSorted((a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime());

    const interruptMutation = $api.useMutation('post', '/api/jobs/{job_id}:interrupt', {
        meta: { invalidates: [['get', '/api/jobs']] },
    });
    const onInterrupt = (job: SchemaTrainJob) => {
        if (job.id !== undefined) {
            interruptMutation.mutate({ params: { path: { job_id: job.id } } });
        }
    };

    return (
        <Dialog width={'90vw'} height={'70vh'} onDismiss={close}>
            <Heading>All jobs</Heading>
            <Divider />
            <Content>
                {sortedJobs.length === 0 ? (
                    <Text>No jobs yet.</Text>
                ) : (
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
                )}
            </Content>
        </Dialog>
    );
};

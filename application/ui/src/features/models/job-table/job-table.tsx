import {
    ActionButton,
    AlertDialog,
    Button,
    DialogTrigger,
    Flex,
    Item,
    Key,
    Menu,
    MenuTrigger,
    ProgressBar,
    Text,
    View,
} from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import { ElapsedDuration } from '../../../components/elapsed-duration.component';
import { notify } from '../../../components/notification/notification.component';
import { Table } from '../../../components/table/table';
import { durationBetween } from '../shared/duration';
import { SingleBadge, SplitBadge } from '../shared/split-badge';
import { SchemaTrainJob } from '../train-model-dialog/train-model-dialog';
import { JobRowContent } from './job-row-content';

import classes from './job-table.module.css';

/** Small pill naming the remote trainer a job runs on. Hidden entirely for local jobs. */
const TrainingLocationBadge = ({ payload }: { payload: SchemaTrainJob['payload'] }) => {
    const { data: remoteTrainers = [] } = $api.useQuery('get', '/api/remote-trainers');

    if (payload.training_target !== 'remote') {
        return null;
    }

    const remoteTrainer = remoteTrainers.find((trainer) => trainer.id === payload.remote_trainer_id);
    const label = remoteTrainer?.name ?? payload.remote_trainer_url ?? 'unknown';
    const text = `Remote · ${label}`;

    return <SingleBadge color='var(--spectrum-global-color-purple-600)' text={text} title={text} preserveCase />;
};
const TrainJobStatus = ({ job }: { job: SchemaTrainJob }) => {
    if (job.status === 'running') {
        return (
            <Flex direction={'column'} gap={'size-50'}>
                <Flex gap={'size-100'} alignItems={'center'} wrap>
                    <Text UNSAFE_style={{ fontWeight: 500 }}>{job.payload.model_name}</Text>
                    <SplitBadge first={job.status} second={job.message} />
                    <TrainingLocationBadge payload={job.payload} />
                </Flex>
                {job.start_time ? (
                    <Text UNSAFE_className={classes.rowInfo}>
                        Started: {new Date(job.start_time).toLocaleString()} | Elapsed:{' '}
                        <ElapsedDuration date={job.start_time} />
                    </Text>
                ) : (
                    <></>
                )}
            </Flex>
        );
    } else {
        const color = job.status === 'failed' ? 'var(--spectrum-negative-visual-color)' : 'var(--energy-blue)';
        return (
            <Flex direction={'column'} gap={'size-50'}>
                <Flex gap={'size-100'} alignItems={'center'} wrap>
                    <Text UNSAFE_style={{ fontWeight: 500 }}>{job.payload.model_name}</Text>
                    <SingleBadge color={color} text={job.status} />
                    <TrainingLocationBadge payload={job.payload} />
                </Flex>
                {job.start_time && job.end_time && (
                    <Text UNSAFE_className={classes.rowInfo}>
                        Elapsed: {durationBetween(job.start_time, job.end_time)}
                    </Text>
                )}
            </Flex>
        );
    }
};

const JobMenu = ({ trainJob, onViewLogs }: { trainJob: SchemaTrainJob; onViewLogs: () => void }) => {
    const queryClient = useQueryClient();
    const deleteJobMutation = $api.useMutation('delete', '/api/jobs/{job_id}', {
        meta: {
            invalidates: [['get', '/api/jobs']],
        },
        onSuccess: () => {
            // Remove the job from the cache immediately instead of waiting on the
            // fire-and-forget invalidation refetch, so the row disappears right away.
            queryClient.setQueryData<SchemaTrainJob[]>(['get', '/api/jobs'], (old = []) =>
                old.filter((job) => job.id !== trainJob.id)
            );
        },
        onError: () => {
            notify('error', `Failed to delete ${trainJob.payload.model_name}`);
        },
    });
    const onAction = (key: Key) => {
        const action = key.toString();
        if (action === 'logs') {
            onViewLogs();
        }
        if (action === 'delete') {
            deleteJobMutation.mutate({
                params: { path: { job_id: trainJob.id! } },
            });
        }
    };

    const isDeletable = trainJob.status === 'failed' || trainJob.status === 'canceled';
    const disabledKeys = isDeletable ? [] : ['delete'];

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label='Job options' isDisabled={deleteJobMutation.isPending}>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={onAction} disabledKeys={disabledKeys}>
                <Item key='logs'>Logs</Item>
                <Item key='delete'>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};

export const TrainingRow = ({
    trainJob,
    onInterrupt,
    onViewLogs,
}: {
    trainJob: SchemaTrainJob;
    onInterrupt: () => void;
    onViewLogs: () => void;
}) => {
    const loss = trainJob.extra_info && (trainJob.extra_info['train/loss_step'] as number | undefined);

    return (
        <Table.ExpandableRow
            id={trainJob.id}
            label={trainJob.payload.model_name}
            detail={<JobRowContent job={trainJob} />}
            after={
                trainJob.status === 'running' && (
                    <ProgressBar
                        size='S'
                        UNSAFE_className={classes.progressBar}
                        width={'100%'}
                        value={trainJob.progress}
                    />
                )
            }
        >
            <TrainJobStatus job={trainJob} />
            <Text>{loss ? loss.toFixed(2) : '...'}</Text>
            <div />
            <Text>{trainJob.payload.policy.toUpperCase()}</Text>
            <div onClick={(e) => e.stopPropagation()}>
                {trainJob.status === 'running' && (
                    <DialogTrigger>
                        <Button variant='secondary'>Stop</Button>
                        <AlertDialog
                            onPrimaryAction={onInterrupt}
                            title='Stop training?'
                            variant='destructive'
                            primaryActionLabel='Stop'
                            cancelLabel='Cancel'
                        >
                            Stop training for {trainJob.payload.model_name}?
                            <br />
                            <br />
                            Your model checkpoint will be saved at the current step. You cannot resume this run.
                        </AlertDialog>
                    </DialogTrigger>
                )}
            </div>
            <View>
                <JobMenu trainJob={trainJob} onViewLogs={onViewLogs} />
            </View>
        </Table.ExpandableRow>
    );
};

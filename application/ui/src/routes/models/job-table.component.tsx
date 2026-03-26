import { ActionButton, Button, Flex, Grid, Item, Key, Menu, MenuTrigger, ProgressBar, Text, View } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { $api, fetchClient } from '../../api/client';
import { GRID_COLUMNS } from './constants';
import { SingleBadge, SplitBadge } from './split-badge.component';
import { SchemaTrainJob } from './train-model-dialog';
import { durationBetween, elapsedSince } from './utils';

import classes from './model-table.module.scss';
import { experimental_streamedQuery as streamedQuery, queryOptions, useQuery } from '@tanstack/react-query';
import { JobMetricsContent } from './metrics';

export const TrainingHeader = () => {
    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelHeader}>
            <Text>Model name</Text>
            <Text>Loss</Text>
            <div />
            <Text>Architecture</Text>
            <div />
            <div />
        </Grid>
    );
};

const TrainJobStatus = ({ job }: { job: SchemaTrainJob }) => {
    if (job.status === 'running') {
        return (
            <View>
                <Flex gap={'size-100'}>
                    <Text UNSAFE_style={{ fontWeight: 500 }}>{job.payload.model_name}</Text>
                    <SplitBadge first={job.status} second={'Fine-tuning the model - epoch n/n'} />
                </Flex>
                {job.start_time ? (
                    <Text UNSAFE_className={classes.modelInfo}>
                        Started: {new Date(job.start_time).toLocaleString()} | Elapsed: {elapsedSince(job.start_time)}
                    </Text>
                ) : (
                    <></>
                )}
            </View>
        );
    } else {
        const color = job.status === 'failed' ? 'var(--spectrum-negative-visual-color)' : 'var(--energy-blue)';
        return (
            <View>
                <Flex gap={'size-100'}>
                    <Text UNSAFE_style={{ fontWeight: 500 }}>{job.payload.model_name}</Text>
                    <SingleBadge color={color} text={job.status} />
                </Flex>
                {job.start_time && job.end_time && (
                    <Text UNSAFE_className={classes.modelInfo}>
                        Elapsed: {durationBetween(job.start_time, job.end_time)}
                    </Text>
                )}
            </View>
        );
    }
};

const JobMenu = ({ trainJob, onViewLogs }: { trainJob: SchemaTrainJob; onViewLogs: () => void }) => {
    const deleteJobMutation = $api.useMutation('delete', '/api/jobs/{job_id}', {
        meta: {
            invalidates: [['get', '/api/jobs']],
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

    const disabledKeys = trainJob.status === 'failed' ? [] : ['delete'];

    return (
        <MenuTrigger>
            <ActionButton
                isQuiet
                UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}
                aria-label='Job options'
                isDisabled={deleteJobMutation.isPending}
            >
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
        <View>
            <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelRow}>
                <TrainJobStatus job={trainJob} />
                <Text>{loss ? loss.toFixed(2) : '...'}</Text>
                <div />
                <Text>{trainJob.payload.policy.toUpperCase()}</Text>
                <View>
                    {trainJob.status === 'running' && (
                        <Button variant='secondary' onPress={onInterrupt}>
                            Interrupt
                        </Button>
                    )}
                </View>
                <View justifySelf={'end'}>
                    <JobMenu trainJob={trainJob} onViewLogs={onViewLogs} />
                </View>
            </Grid>

            {trainJob.status === 'running' && (
                <ProgressBar size='S' UNSAFE_className={classes.progressBar} width={'100%'} value={trainJob.progress} />
            )}

            <Flex height="size-6000">
                <JobMetricsContent jobId={trainJob.id!}/>
            </Flex>
        </View>
    );
};

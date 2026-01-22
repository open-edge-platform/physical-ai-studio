import { Button, Flex, Grid, ProgressBar, Text, View } from '@geti/ui';

import { GRID_COLUMNS } from './constants';
import { SingleBadge, SplitBadge } from './split-badge.component';
import { SchemaTrainJob } from './train-model';

import classes from './model-table.module.scss';

const timeSince = (dateString: string) => {
    const date = new Date(dateString);
    const diff = (new Date().getTime() - date.getTime());

    const duration = new Date(diff);
    return new Intl.DateTimeFormat('en',{
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        timeZone: "UTC",
        hour12: false
    }).format(duration);
};

export const TrainingHeader = () => {
    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelHeader}>
            <Text>Model name</Text>
            <Text>Loss</Text>
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
                        Started: {new Date(job.start_time).toLocaleString()} | Elapsed: {timeSince(job.start_time)}
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
            </View>
        );
    }
};

export const TrainingRow = ({ trainJob, onInterrupt }: { trainJob: SchemaTrainJob; onInterrupt: () => void }) => {
    const loss = trainJob.extra_info && (trainJob.extra_info['train/loss_step'] as number | undefined);

    return (
        <View>
            <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelRow}>
                <TrainJobStatus job={trainJob} />
                <Text>{loss ? loss.toFixed(2) : '...'}</Text>
                <Text>{trainJob.payload.policy.toUpperCase()}</Text>
                <View>
                    {trainJob.status === 'running' && (
                        <Button variant='secondary' onPress={onInterrupt}>
                            Interrupt
                        </Button>
                    )}
                </View>
            </Grid>

            {trainJob.status === 'running' && (
                <ProgressBar size='S' UNSAFE_className={classes.progressBar} width={'100%'} value={trainJob.progress} />
            )}
        </View>
    );
};

import { Grid, Flex, Text, View, Button, ProgressBar } from "@geti/ui"


import classes from './model-table.module.scss';
import { SchemaTrainJob } from "./train-model";
import { GRID_COLUMNS } from "./constants";
import { SplitBadge } from "./split-badge.component";

const timeSince = (dateString: string) => {
    const date = new Date(dateString);
    const diff = (new Date().getTime() - date.getTime()) / 1000;

    const hours = Math.floor(diff / 3600)
    const minutes = Math.floor((diff % 3600) / 60)
    const seconds = Math.floor(diff % 60)
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

export const TrainingHeader = () => {
    return (
        <Grid
            columns={GRID_COLUMNS}
            alignItems={'center'}
            width={"100%"}
            UNSAFE_className={classes.modelHeader}
        >
            <Text>Model name</Text>
            <Text>Loss</Text>
            <Text>Architecture</Text>
            <div />
            <div />
        </Grid>
    )
}


export const TrainingRow = ({ trainJob, onInterrupt }: { trainJob: SchemaTrainJob, onInterrupt: () => void }) => {

    const loss = trainJob.extra_info["train/loss_step"] as number | undefined

    return (
        <View>
            <Grid
                columns={GRID_COLUMNS}
                alignItems={'center'}
                width={"100%"}
                UNSAFE_className={classes.modelRow}
            >
                <View>
                    <Flex gap={'size-100'}>
                        <Text UNSAFE_style={{ fontWeight: 500 }}>{trainJob.payload.model_name}</Text>
                        <SplitBadge first={"Training"} second={"Fine-tuning the model - epoch n/n"} />
                    </Flex>
                    {trainJob.start_time ?
                        <Text UNSAFE_className={classes.modelInfo}>
                            Started: {new Date(trainJob.start_time).toLocaleString()} | Elapsed: {timeSince(trainJob.start_time)}
                        </Text>
                        : <></>
                    }
                </View>
              <Text>{loss ? loss.toFixed(2) : "..."}</Text>
                <Text>{trainJob.payload.policy.toUpperCase()}</Text>
                <View>
                    <Button variant='secondary' onPress={onInterrupt}>
                        Interrupt
                    </Button>
                </View>
            </Grid>

            {trainJob.status === 'running' && <ProgressBar size='S' UNSAFE_className={classes.progressBar} width={'100%'} value={trainJob.progress} />}
        </View>
    )
}

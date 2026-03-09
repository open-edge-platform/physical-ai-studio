import { ActionButton, DialogTrigger, Button, Grid, Item, Key, Link, Menu, MenuTrigger, Text, View } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { SchemaJob, SchemaModel } from '../../api/openapi-spec';
import { paths } from '../../router';
import { GRID_COLUMNS } from './constants';
import { durationBetween } from './utils';

import classes from './model-table.module.scss';
import { StartInferenceDialog } from './start-model-modal.component';

export const ModelHeader = () => {
    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelHeader}>
            <Text>Model name</Text>
            <Text>Trained</Text>
            <Text>Duration</Text>
            <Text>Architecture</Text>
            <div />
            <div />
        </Grid>
    );
};

export const ModelRow = ({
    model,
    trainingJob,
    onDelete,
}: {
    model: SchemaModel;
    trainingJob?: SchemaJob;
    onDelete: () => void;
}) => {
    const onAction = (key: Key) => {
        const action = key.toString();
        if (action === 'delete') {
            onDelete();
        }
    };

    const duration =
        trainingJob?.start_time && trainingJob?.end_time
            ? durationBetween(trainingJob.start_time, trainingJob.end_time)
            : null;

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} UNSAFE_className={classes.modelRow}>
            <Text>{model.name}</Text>
            <Text>{new Date(model.created_at!).toLocaleString()}</Text>
            <Text UNSAFE_className={duration ? undefined : classes.modelInfo}>{duration ?? '—'}</Text>
            <Text>{model.policy.toUpperCase()}</Text>
            <View>
                <DialogTrigger>
                    <Button variant="secondary">Run model</Button>
                    {(close) => StartInferenceDialog(close, model.project_id, model.id!)}
                </DialogTrigger>
            </View>
            <View>
                <MenuTrigger>
                    <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }} aria-label='options'>
                        <MoreMenu />
                    </ActionButton>
                    <Menu onAction={onAction}>
                        <Item key='delete'>Delete</Item>
                    </Menu>
                </MenuTrigger>
            </View>
        </Grid>
    );
};

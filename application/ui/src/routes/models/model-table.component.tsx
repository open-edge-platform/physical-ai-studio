import { useState } from 'react';

import { ActionButton, Button, DialogTrigger, Flex, Grid, Item, Key, Menu, MenuTrigger, Text, View } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { SchemaModel, SchemaTrainJob } from '../../api/openapi-spec';
import { CollapsableRow } from './collapsable-row.component';
import { GRID_COLUMNS } from './constants';
import { ModelDownloadDialog } from './model-download-dialog.component';
import { ModelRowContent } from './model-row-content.component';
import { StartInferenceDialog } from './start-model-modal.component';
import { durationBetween } from './utils';

import classes from './model-table.module.scss';

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
    onRetrain,
    onViewLogs,
}: {
    model: SchemaModel;
    trainingJob?: SchemaTrainJob;
    onDelete: () => void;
    onRetrain: () => void;
    onViewLogs?: () => void;
}) => {
    const [isDownloadDialogOpen, setDownloadDialogOpen] = useState(false);

    const onAction = (key: Key) => {
        const action = key.toString();
        if (action === 'delete') {
            onDelete();
        }
        if (action === 'retrain') {
            onRetrain();
        }
        if (action === 'logs') {
            onViewLogs?.();
        }
        if (action === 'download') {
            setDownloadDialogOpen(true);
        }
    };

    const duration =
        trainingJob?.start_time && trainingJob?.end_time
            ? durationBetween(trainingJob.start_time, trainingJob.end_time)
            : null;

    // Disable logs if we don't know the training job
    const disabledKeys = !model.train_job_id ? ['logs'] : [];

    const version = model.version ?? 1;

    return (
        <View>
            <CollapsableRow
                header={
                    <Grid
                        columns={GRID_COLUMNS}
                        alignItems={'center'}
                        width={'100%'}
                        UNSAFE_className={classes.modelRow}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <Text>{model.name}</Text>
                            {version > 1 && (
                                <Text UNSAFE_style={{ color: 'var(--spectrum-gray-600)', fontSize: '0.85em' }}>
                                    v{version}
                                </Text>
                            )}
                        </Flex>
                        <Text>{new Date(model.created_at!).toLocaleString()}</Text>
                        <Text UNSAFE_className={duration ? undefined : classes.modelInfo}>{duration ?? '—'}</Text>
                        <Text>{model.policy.toUpperCase()}</Text>
                        <View>
                            <DialogTrigger>
                                <Button variant='secondary'>Run model</Button>
                                {(close) => (
                                    <StartInferenceDialog
                                        close={close}
                                        project_id={model.project_id}
                                        model_id={model.id!}
                                    />
                                )}
                            </DialogTrigger>
                        </View>
                        <View justifySelf={'end'}>
                            <MenuTrigger direction='left'>
                                <ActionButton
                                    isQuiet
                                    UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}
                                    aria-label='options'
                                >
                                    <MoreMenu />
                                </ActionButton>
                                <Menu onAction={onAction} disabledKeys={disabledKeys}>
                                    <Item key='logs'>Logs</Item>
                                    <Item key='download'>Download</Item>
                                    <Item key='retrain'>Retrain</Item>
                                    <Item key='delete'>Delete</Item>
                                </Menu>
                            </MenuTrigger>
                            <ModelDownloadDialog
                                modelId={model.id!}
                                isOpen={isDownloadDialogOpen}
                                onClose={() => setDownloadDialogOpen(false)}
                            />
                        </View>
                    </Grid>
                }
            >
                <ModelRowContent model={model} />
            </CollapsableRow>
        </View>
    );
};

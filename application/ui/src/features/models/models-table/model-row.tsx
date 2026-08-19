import { useState } from 'react';

import { ActionButton, Button, DialogTrigger, Flex, Item, Key, Menu, MenuTrigger, Text, View } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { SchemaModel, SchemaTrainJob } from '../../../api/openapi-spec';
import { Table } from '../../../components/table/table';
import { durationBetween } from '../shared/duration';
import { ModelDownloadDialog } from './model-download-dialog';
import { ModelRowContent } from './model-row-content';
import { StartInferenceDialog } from './start-inference-dialog';

import classes from './model-table.module.css';

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
        <Table.ExpandableRow id={model.id} label={model.name} detail={<ModelRowContent model={model} />}>
            <Flex alignItems='center' gap='size-100'>
                <Text>{model.name}</Text>
                {version > 1 && <Text UNSAFE_className={classes.versionBadge}>v{version}</Text>}
            </Flex>
            <Text>{new Date(model.created_at!).toLocaleString()}</Text>
            <Text UNSAFE_className={duration ? undefined : classes.rowInfo}>{duration ?? '—'}</Text>
            <Text>{model.policy.toUpperCase()}</Text>
            <div onClick={(e) => e.stopPropagation()}>
                <DialogTrigger>
                    <Button variant='secondary'>Run model</Button>
                    {(close) => <StartInferenceDialog close={close} model={model} />}
                </DialogTrigger>
            </div>
            <View>
                <MenuTrigger direction='left'>
                    <ActionButton isQuiet UNSAFE_className={classes.optionsButton} aria-label='options'>
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
        </Table.ExpandableRow>
    );
};

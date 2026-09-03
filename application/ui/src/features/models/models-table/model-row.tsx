import { useState } from 'react';

import { ActionButton, Button, DialogTrigger, Flex, Item, Key, Menu, MenuTrigger, Text, View } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { SchemaModel, SchemaTrainJob } from '../../../api/openapi-spec';
import { Table } from '../../../components/table/table';
import { durationBetween } from '../shared/duration';
import { ModelDownloadDialog } from './model-download-dialog';
import { ModelRowContent } from './model-row-content';
import { StartInferenceDialog } from './start-inference-dialog';

import classes from './model-table.module.css';

const useDatasetQuery = (datasetId: string | undefined | null) => {
    return $api.useQuery(
        'get',
        '/api/dataset/{dataset_id}',
        {
            params: { path: { dataset_id: String(datasetId) } },
        },
        {
            enabled: datasetId != null,
        }
    );
};

const useEnvironmentQuery = (projectId: string, environmentId: string | undefined) => {
    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/environments/{environment_id}',
        {
            params: {
                path: {
                    environment_id: String(environmentId),
                    project_id: projectId,
                },
            },
        },
        {
            enabled: environmentId !== undefined,
        }
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
    const { data: dataset } = useDatasetQuery(model.dataset_id);
    const { data: environment } = useEnvironmentQuery(model.project_id, dataset?.environment_id);

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
            <Text>{model.policy.toUpperCase()}</Text>
            <Text>{dataset?.name ?? '-'}</Text>
            <Text>{environment?.name ?? '-'}</Text>
            <Text>{new Date(model.created_at!).toLocaleString()}</Text>
            <Text UNSAFE_className={duration ? undefined : classes.rowInfo}>{duration ?? '—'}</Text>
            <div onClick={(e) => e.stopPropagation()}>
                <DialogTrigger>
                    <Button variant='secondary'>Run model</Button>
                    {(close) => <StartInferenceDialog close={close} model={model} />}
                </DialogTrigger>
            </div>
            <View>
                <MenuTrigger direction='left'>
                    <ActionButton isQuiet aria-label='options'>
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

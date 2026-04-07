import { Suspense, useState } from 'react';

import { ManagedTab, type ManagedTabAction } from '@geti-ui/blocks';
import { ActionButton, DialogContainer, Flex, Icon, Item, TabList } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';
import { useNavigate } from 'react-router';

import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { paths } from '../../router';
import { NewDatasetForm } from '../../routes/datasets/new-dataset.component';
import { useProjectId } from '../projects/use-project';
import { DeleteDatasetDialog } from './delete-dataset-dialog';
import { DatasetDownloadDialog } from './export-dataset-dialog';
import { RenameDatasetDialog } from './rename-dataset-dialog';

type Dataset = SchemaDatasetOutput;

const ACTIONS = [
    { key: 'rename', label: 'Edit dataset name' },
    { key: 'export', label: 'Export dataset' },
    { key: 'delete', label: 'Delete dataset' },
] satisfies ManagedTabAction[];

export const DatasetTabs = ({
    datasets,
    selectedDatasetId,
}: {
    datasets: Array<SchemaDatasetOutput>;
    selectedDatasetId: string | undefined;
}) => {
    const { project_id } = useProjectId();

    const [action, setAction] = useState<null | 'rename' | 'delete' | 'export' | 'add'>(null);
    const selectedDataset = datasets.find((dataset) => dataset.id === selectedDatasetId);

    const onItemAction = (itemAction: string) => {
        switch (itemAction) {
            case 'delete':
            case 'rename':
            case 'export':
                setAction(itemAction);
        }
    };

    const navigate = useNavigate();
    const onAddDataset = (dataset: SchemaDatasetOutput | undefined) => {
        setAction(null);

        if (dataset?.id) {
            navigate(paths.project.datasets.show({ project_id, dataset_id: dataset.id }));
        }
    };

    const onDatasetDeleteDone = (deletedDataset: Dataset) => {
        setAction(null);

        if (selectedDatasetId !== deletedDataset.id) {
            return;
        }

        const nextDataset = datasets.find((dataset) => dataset.id !== deletedDataset.id);

        if (nextDataset?.id) {
            navigate(paths.project.datasets.show({ project_id, dataset_id: nextDataset.id }));
            return;
        }

        navigate(paths.project.datasets.index({ project_id }));
    };

    return (
        <Flex>
            <TabList>
                {datasets.map((dataset) => {
                    return (
                        <Item
                            aria-label={dataset.name}
                            key={dataset.id}
                            href={paths.project.datasets.show({ project_id, dataset_id: dataset.id! })}
                        >
                            <ManagedTab
                                label={dataset.name}
                                isSelected={dataset.id === selectedDatasetId}
                                actions={ACTIONS}
                                onAction={onItemAction}
                            />
                        </Item>
                    );
                })}
            </TabList>

            <div
                style={{
                    display: 'flex',
                    flex: '1 1 auto',
                    alignItems: 'center',
                    borderBottom: 'var(--spectrum-alias-border-size-thick) solid var(--spectrum-global-color-gray-300)',
                }}
            >
                <ActionButton
                    isQuiet
                    aria-label='Add dataset'
                    onPress={() => {
                        setAction('add');
                    }}
                >
                    <Icon>
                        <Add />
                    </Icon>
                </ActionButton>
            </div>

            <DialogContainer
                onDismiss={() => {
                    setAction(null);
                }}
            >
                {action === 'add' && (
                    <Suspense>
                        <NewDatasetForm project_id={project_id} onDone={onAddDataset} />
                    </Suspense>
                )}
                {action === 'export' && selectedDatasetId !== undefined && (
                    <DatasetDownloadDialog
                        datasetId={selectedDatasetId}
                        onCloseDialog={() => {
                            setAction(null);
                        }}
                    />
                )}
                {action === 'rename' && selectedDataset !== undefined && (
                    <RenameDatasetDialog
                        dataset={selectedDataset}
                        onDone={() => {
                            setAction(null);
                        }}
                    />
                )}
                {action === 'delete' && selectedDataset !== undefined && (
                    <DeleteDatasetDialog
                        dataset={selectedDataset}
                        onDone={() => onDatasetDeleteDone(selectedDataset)}
                    />
                )}
            </DialogContainer>
        </Flex>
    );
};

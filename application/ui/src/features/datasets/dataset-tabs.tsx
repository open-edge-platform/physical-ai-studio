import { Suspense, useState } from 'react';

import { ManagedTab, type ManagedTabAction } from '@geti-ui/blocks';
import { ActionButton, DialogContainer, Flex, Icon, Item, Menu, MenuTrigger, TabList } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';
import { useNavigate } from 'react-router';

import { fetchClient } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { paths } from '../../router';
import { ImportDatasetDialog } from '../../routes/datasets/import/dataset-import-button';
import { NewDatasetForm } from '../../routes/datasets/new-dataset.component';
import { useProjectId } from '../projects/use-project';
import { DeleteDatasetDialog } from './delete-dataset-dialog';
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

    const [action, setAction] = useState<null | 'rename' | 'delete' | 'add' | 'import'>(null);
    const selectedDataset = datasets.find((dataset) => dataset.id === selectedDatasetId);

    const openDatasetDownload = (datasetId: string) => {
        const downloadUrl = fetchClient.PATH('/api/dataset/{dataset_id}/download', {
            params: { path: { dataset_id: datasetId } },
        });

        window.open(downloadUrl, '_blank', 'noopener,noreferrer');
    };

    const onItemAction = (itemAction: string) => {
        switch (itemAction) {
            case 'export':
                if (selectedDatasetId !== undefined) {
                    openDatasetDownload(selectedDatasetId);
                }
                return;
            case 'delete':
            case 'rename':
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
                <MenuTrigger>
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
                    <Menu
                        onAction={(key) => {
                            if (key === 'add') {
                                setAction('add');
                            }
                            if (key === 'import') {
                                setAction('import');
                            }
                        }}
                    >
                        <Item key='add'>Add</Item>
                        <Item key='import'>Import</Item>
                    </Menu>
                </MenuTrigger>
            </div>

            <DialogContainer
                onDismiss={() => {
                    setAction(null);
                }}
            >
                {action === 'import' && <ImportDatasetDialog onClose={() => setAction(null)} />}
                {action === 'add' && (
                    <Suspense>
                        <NewDatasetForm project_id={project_id} onDone={onAddDataset} />
                    </Suspense>
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

import { Suspense, useState } from 'react';

import { ManagedTab, type ManagedTabAction } from '@geti-ui/blocks';
import { ActionButton, DialogContainer, Icon, Item, Menu, MenuTrigger } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';
import { Tab, TabList } from 'react-aria-components';
import { useNavigate } from 'react-router';

import { fetchClient } from '../../api/client';
import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { paths } from '../../router';
import { ImportDatasetDialog } from '../../routes/datasets/import/dataset-import-button';
import { NewDatasetForm } from '../../routes/datasets/new-dataset.component';
import { useProjectId } from '../projects/use-project';
import { DeleteDatasetDialog } from './delete-dataset-dialog';
import { RenameDatasetDialog } from './rename-dataset-dialog';

import styles from './dataset-tabs.module.css';

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
    const navigate = useNavigate();
    const [action, setAction] = useState<null | 'rename' | 'delete' | 'create' | 'import'>(null);
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

    const handleCreateDataset = (dataset: SchemaDatasetOutput | undefined) => {
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
        <>
            <div className={styles.tabBar}>
                <TabList aria-label='Datasets' className={styles.tabList}>
                    {datasets.map((dataset) => {
                        const isSelected = dataset.id === selectedDatasetId;

                        return (
                            <Tab className={styles.tab} id={dataset.id} key={dataset.id}>
                                {isSelected ? (
                                    <ManagedTab
                                        label={dataset.name}
                                        isSelected
                                        actions={ACTIONS}
                                        onAction={onItemAction}
                                    />
                                ) : (
                                    dataset.name
                                )}
                            </Tab>
                        );
                    })}
                </TabList>

                <div className={styles.addActions}>
                    <MenuTrigger>
                        <ActionButton
                            isQuiet
                            aria-label='Create new dataset'
                            onPress={() => {
                                setAction('create');
                            }}
                        >
                            <Icon>
                                <Add />
                            </Icon>
                        </ActionButton>
                        <Menu
                            onAction={(key) => {
                                if (key === 'create') {
                                    setAction('create');
                                }
                                if (key === 'import') {
                                    setAction('import');
                                }
                            }}
                        >
                            <Item key='create'>Create</Item>
                            <Item key='import'>Import</Item>
                        </Menu>
                    </MenuTrigger>
                </div>
            </div>

            <DialogContainer
                onDismiss={() => {
                    setAction(null);
                }}
            >
                {action === 'import' && <ImportDatasetDialog onClose={() => setAction(null)} />}
                {action === 'create' && (
                    <Suspense>
                        <NewDatasetForm project_id={project_id} onDone={handleCreateDataset} />
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
        </>
    );
};

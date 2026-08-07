import { Suspense } from 'react';

import { Content, Flex, Heading, IllustratedMessage, Loading, Text } from '@geti-ui/ui';
import { TabPanel, Tabs } from 'react-aria-components';
import { useNavigate, useParams } from 'react-router';

import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { DatasetTabs } from '../../features/datasets/dataset-tabs';
import { useProject, useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { DatasetProvider } from './dataset-provider';
import { DatasetViewer } from './dataset-viewer';
import { DatasetImportButton } from './import/dataset-import-button';
import { NewDatasetLink } from './new-dataset.component';

interface DatasetsProps {
    datasets: SchemaDatasetOutput[];
}

const Datasets = ({ datasets }: DatasetsProps) => {
    const { project_id } = useProjectId();
    const navigate = useNavigate();
    const params = useParams();
    const dataset_id = params.dataset_id ?? datasets[0]?.id;

    if (datasets.length === 0) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex height='100%'>
                <IllustratedMessage>
                    <EmptyIllustration />
                    <Content> Currently there are no datasets available. </Content>
                    <Text>It&apos;s time to begin recording a dataset. </Text>
                    <Heading>No datasets yet</Heading>
                    <Flex gap='size-100' marginTop={'size-200'}>
                        <NewDatasetLink project_id={project_id} />
                        <DatasetImportButton />
                    </Flex>
                </IllustratedMessage>
            </Flex>
        );
    }

    return (
        <Flex
            height='100%'
            width='100%'
            minWidth={0}
            UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-200)' }}
        >
            <Tabs
                aria-label='Datasets'
                selectedKey={dataset_id}
                onSelectionChange={(key) => {
                    navigate(paths.project.datasets.show({ project_id, dataset_id: String(key) }));
                }}
                style={{
                    display: 'flex',
                    flex: 1,
                    width: '100%',
                    minWidth: 0,
                    flexDirection: 'column',
                }}
            >
                <DatasetTabs datasets={datasets} selectedDatasetId={dataset_id} />
                <TabPanel
                    id={dataset_id}
                    style={{
                        display: 'flex',
                        minHeight: 0,
                        flex: 1,
                        marginTop: 'var(--spectrum-global-dimension-size-200)',
                    }}
                >
                    <Flex height='100%' flex>
                        {dataset_id === undefined ? (
                            <Text>No datasets yet...</Text>
                        ) : (
                            <Suspense fallback={<Loading />}>
                                <DatasetProvider dataset_id={dataset_id}>
                                    <DatasetViewer />
                                </DatasetProvider>
                            </Suspense>
                        )}
                    </Flex>
                </TabPanel>
            </Tabs>
        </Flex>
    );
};

export const Index = () => {
    const project = useProject();
    return <Datasets datasets={project.datasets} />;
};

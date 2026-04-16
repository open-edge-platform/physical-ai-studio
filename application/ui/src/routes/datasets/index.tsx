import { Content, Flex, Heading, IllustratedMessage, Item, TabPanels, Tabs, Text, View } from '@geti-ui/ui';
import { useParams } from 'react-router';

import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { DatasetTabs } from '../../features/datasets/dataset-tabs';
import { useProject, useProjectId } from '../../features/projects/use-project';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { DatasetProvider } from './dataset-provider';
import { DatasetViewer } from './dataset-viewer';
import { NewDatasetLink } from './new-dataset.component';

interface DatasetsProps {
    datasets: SchemaDatasetOutput[];
}

const Datasets = ({ datasets }: DatasetsProps) => {
    const { project_id } = useProjectId();
    const params = useParams();
    const dataset_id = params.dataset_id ?? datasets[0]?.id;

    if (datasets.length === 0) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex>
                <IllustratedMessage>
                    <EmptyIllustration />
                    <Content> Currently there are datasets available. </Content>
                    <Text>It&apos;s time to begin recording a dataset. </Text>
                    <Heading>No datasets yet</Heading>
                    <View margin={'size-100'}>
                        <NewDatasetLink project_id={project_id} />
                    </View>
                </IllustratedMessage>
            </Flex>
        );
    }

    return (
        <Flex height='100%'>
            <Tabs
                aria-label='Datasets'
                selectedKey={dataset_id}
                flex='1'
                margin={'size-200'}
                UNSAFE_style={{
                    '--spectrum-tabs-item-gap': 'var(--spectrum-global-dimension-size-100)',
                }}
            >
                <DatasetTabs datasets={datasets} selectedDatasetId={dataset_id} />
                <TabPanels UNSAFE_style={{ border: 'none' }} marginTop={'size-200'} minHeight={0}>
                    <Item key={dataset_id}>
                        <Flex height='100%' flex>
                            {dataset_id === undefined ? (
                                <Text>No datasets yet...</Text>
                            ) : (
                                <DatasetProvider dataset_id={dataset_id}>
                                    <DatasetViewer />
                                </DatasetProvider>
                            )}
                        </Flex>
                    </Item>
                </TabPanels>
            </Tabs>
        </Flex>
    );
};

export const Index = () => {
    const project = useProject();
    return <Datasets datasets={project.datasets} />;
};

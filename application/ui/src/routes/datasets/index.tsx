import { useState } from 'react';

import {
    Content,
    Flex,
    Heading,
    Icon,
    IllustratedMessage,
    Item,
    Key,
    TabList,
    TabPanels,
    Tabs,
    Text,
    View,
} from '@geti/ui';
import { Add } from '@geti/ui/icons';
import { useParams } from 'react-router';

import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { useProject, useProjectId } from '../../features/projects/use-project';
import { paths } from '../../router';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { DatasetProvider } from './dataset-provider';
import { DatasetViewer } from './dataset-viewer';
import { NewDatasetDialogContainer, NewDatasetLink } from './new-dataset.component';

interface DatasetsProps {
    datasets: SchemaDatasetOutput[];
}

const Datasets = ({ datasets }: DatasetsProps) => {
    const { project_id } = useProjectId();
    const params = useParams();
    const dataset_id = params.dataset_id ?? datasets[0]?.id;

    const [showDialog, setShowDialog] = useState<boolean>(false);

    const onSelectionChange = (key: Key) => {
        if (key.toString() === '#new-dataset') {
            setShowDialog(true);
        }
    };

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
            <Tabs onSelectionChange={onSelectionChange} selectedKey={dataset_id} flex='1' margin={'size-200'}>
                <Flex alignItems={'end'}>
                    <TabList flex={1}>
                        {[
                            ...datasets.map((data) => (
                                <Item
                                    key={data.id}
                                    href={paths.project.datasets.show({ project_id, dataset_id: data.id! })}
                                >
                                    <Text UNSAFE_style={{ fontSize: '16px' }}>{data.name}</Text>
                                </Item>
                            )),
                            <Item key='#new-dataset'>
                                <Icon>
                                    <Add />
                                </Icon>
                            </Item>,
                        ]}
                    </TabList>
                </Flex>
                <TabPanels UNSAFE_style={{ border: 'none' }} marginTop={'size-200'}>
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
            <NewDatasetDialogContainer
                project_id={project_id}
                show={showDialog}
                onDismiss={() => setShowDialog(false)}
            />
        </Flex>
    );
};

export const Index = () => {
    const project = useProject();
    return <Datasets datasets={project.datasets} />;
};

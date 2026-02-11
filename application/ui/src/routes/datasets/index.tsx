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

import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { useProject, useProjectId } from '../../features/projects/use-project';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { DatasetViewer } from './dataset-viewer';
import { NewDatasetDialogContainer, NewDatasetLink } from './new-dataset.component';
import { RecordingProvider } from './recording-provider';
import { Add } from '@geti/ui/icons';

interface DatasetsProps {
    datasets: SchemaDatasetOutput[];
}

const Datasets = ({ datasets }: DatasetsProps) => {
    const { project_id } = useProjectId();
    const [dataset, setDataset] = useState<SchemaDatasetOutput | undefined>(
        datasets.length > 0 ? datasets[0] : undefined
    );

    const [showDialog, setShowDialog] = useState<boolean>(false);

    const onSelectionChange = (key: Key) => {
        if (key.toString() === '#new-dataset') {
            setShowDialog(true);
        } else {
            setDataset(datasets.find((d) => d.id === key.toString()));
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
            <Tabs onSelectionChange={onSelectionChange} selectedKey={dataset?.id} flex='1' margin={'size-200'}>
                <Flex alignItems={'end'}>
                    <TabList flex={1}>
                        {
                            [
                                ...datasets.map((data) => (
                                    <Item key={data.id}>
                                        <Text UNSAFE_style={{fontSize: '16px'}}>{data.name}</Text>
                                    </Item>
                                )),
                                <Item key='#new-dataset'>
                                    <Icon>
                                        <Add />
                                    </Icon>
                                </Item>
                            ]
                        }
                    </TabList>
                </Flex>
                <TabPanels UNSAFE_style={{ border: 'none' }} marginTop={'size-200'}>
                    <Item key={dataset?.id}>
                        <Flex height='100%' flex>
                            {dataset === undefined ? (
                                <Text>No datasets yet...</Text>
                            ) : (
                                <DatasetViewer dataset={dataset} />
                            )}
                        </Flex>
                    </Item>
                </TabPanels>
            </Tabs>
            <NewDatasetDialogContainer project_id={project_id} show={showDialog} onDismiss={() => setShowDialog(false)} />
        </Flex>
    );
};

export const Index = () => {
    const project = useProject();
    return (
        <RecordingProvider>
            <Datasets datasets={project.datasets} />
        </RecordingProvider>
    );
};

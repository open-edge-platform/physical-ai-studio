import { $api } from '../../api/client';
import { ErrorMessage } from '../../components/error-page/error-page';
import { useProject } from '../projects/project.provider';
import { View, Button, Text, Flex, Key, Well, Tabs, TabList, TabPanels, Item, Link } from '@geti/ui'
import { DatasetViewer } from './dataset-viewer';
import { DatasetProvider } from './dataset.provider';
import { Add } from '@geti/ui/icons';
import { useNavigate } from 'react-router';
import { paths } from '../../router';
import { useState } from 'react';

export const Index = () => {
    const navigate = useNavigate();
    const { project } = useProject();
    const datasets = project.datasets;
    const [dataset, setDataset] = useState<string>(datasets.length > 0 ? datasets[0] : "");

    const onSelectionChange = (key: Key) => {
        if (key.toString() === "#new-dataset") {
            navigate(paths.project.datasets.record({ project_id: project.id }));
        } else {
            setDataset(key.toString());
        }
    }

    return (
        <View padding="size-200" flex="1">
            <Flex height="100%">
                <Tabs onSelectionChange={onSelectionChange} flex="1">
                    <Flex alignItems={"end"}>
                        <TabList flex={1}>
                            {[
                                ...datasets.map((dataset) => <Item key={dataset}>{dataset}</Item>),
                                <Item key="#new-dataset"><Add fill="white" height="10px" /> New dataset</Item>
                            ]}
                        </TabList>
                        {dataset !== undefined &&
                            <View padding={"size-30"}>
                                <Link href={paths.project.datasets.record({ project_id: project.id }).concat(`?dataset=${dataset}`)}>
                                    <Button>Start recording</Button>
                                </Link>
                            </View>}
                    </Flex>
                    <TabPanels>
                        <Item key={dataset}>
                            <Flex height="100%">
                                <Well flex={1}>
                                    {dataset === undefined
                                        ? <Text>No datasets yet...</Text>
                                        : (
                                            <DatasetProvider project_id={project.id} repo_id={dataset}>
                                                <DatasetViewer />
                                            </DatasetProvider>
                                        )
                                    }
                                </Well>
                            </Flex>

                        </Item>

                    </TabPanels>

                </Tabs>
            </Flex>
        </View>
    )
};

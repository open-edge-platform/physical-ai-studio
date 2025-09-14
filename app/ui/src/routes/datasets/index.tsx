import { $api } from '../../api/client';
import { ErrorMessage } from '../../components/error-page/error-page';
import { useProject } from '../projects/project.provider';
import { View, Text, Flex, Divider, Well, Tabs, TabList, TabPanels, Item } from '@geti/ui'
import { DatasetViewer } from './dataset-viewer';
import { DatasetProvider } from './dataset.provider';

export const Index = () => {
    const { project } = useProject();
    const dataset = project.datasets[0];
    return (
        <Flex direction={'column'} height={'100%'}>
            <Tabs>
                <TabList>
                    <Item key="Test 1">Test</Item>
                    <Item key="Test 2">Test</Item>
                </TabList>

                <TabPanels>
                    <Item key="Test 1">
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
                  
                    </Item>

                </TabPanels>

            </Tabs>
        </Flex>
    )
};

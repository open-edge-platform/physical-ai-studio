import { useState } from 'react';

import { Button, Flex, Item, Key, Link, TabList, TabPanels, Tabs, Text, View, Well } from '@geti/ui';
import { Add } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { SchemaDataset } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';
import { paths } from '../../router';
import { DatasetViewer } from './dataset-viewer';
import { ImportDataset } from './import/import';

export const Index = () => {
    const navigate = useNavigate();
    const project = useProject();
    const datasets = project.datasets;
    const [dataset, setDataset] = useState<SchemaDataset | undefined>(datasets.length > 0 ? datasets[0] : undefined);

    const onSelectionChange = (key: Key) => {
        if (key.toString() === '#new-dataset') {
            if (datasets.length === 0) {
                setDataset(undefined);
            } else {
                navigate(paths.project.datasets.record_new({ project_id: project.id }));
            }
        } else {
            setDataset(datasets.find((d) => d.id === key.toString()));
        }
    };

    return (
        <Flex height='100%'>
            <Tabs onSelectionChange={onSelectionChange} flex='1' margin={'size-200'}>
                <Flex alignItems={'end'}>
                    <TabList flex={1}>
                        {[
                            ...datasets.map((data) => <Item key={data.id}>{data.name}</Item>),
                            <Item key='#new-dataset'>
                                <Add fill='white' height='10px' /> New dataset
                            </Item>,
                        ]}
                    </TabList>
                    {dataset !== undefined && (
                        <View padding={'size-30'}>
                            <Link
                                href={paths.project.datasets.record({
                                    project_id: project.id,
                                    dataset_id: dataset.id,
                                })}
                            >
                                <Button>Start recording</Button>
                            </Link>
                        </View>
                    )}
                </Flex>
                <TabPanels UNSAFE_style={{border: 'none'}} marginTop={'size-200'}>
                    <Item key={'#new-dataset'}>
                        <ImportDataset />
                    </Item>
                    <Item key={dataset?.id}>
                        <Flex height='100%' flex>
                            {dataset === undefined ? (
                                <Text>No datasets yet...</Text>
                            ) : (
                                <DatasetViewer id={dataset.id!} />
                            )}
                        </Flex>
                    </Item>
                </TabPanels>
            </Tabs>
        </Flex>
    );
};

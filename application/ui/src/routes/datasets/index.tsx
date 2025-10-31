import { useState } from 'react';

import { Button, Divider, DialogTrigger, Content, Dialog, Flex, Heading, Item, Key, TabList, TabPanels, Tabs, Text, View } from '@geti/ui';
import { Add } from '@geti/ui/icons';

import { SchemaDatasetOutput, SchemaTeleoperationConfig } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';
import { DatasetViewer } from './dataset-viewer';
import { ImportDataset } from './import/import';
import { HardwareSetup } from './record/hardware-setup';
import { RecordingProvider, useRecording } from './recording-provider';


const HardwareSetupModal = (close: (config: SchemaTeleoperationConfig | undefined) => void, dataset_id: string | undefined) => {
    return (
        <Dialog>
            <Heading>Teleoperate Setup</Heading>
            <Divider />
            <Content>
                <HardwareSetup dataset_id={dataset_id} onDone={close} />
            </Content>
        </Dialog>
    )
}

interface DatasetsProps {
    datasets: SchemaDatasetOutput[]
}

const Datasets = ({ datasets }: DatasetsProps) => {
    const { isRecording, recordingConfig, setRecordingConfig} = useRecording();
    const [dataset, setDataset] = useState<SchemaDatasetOutput | undefined>(
        datasets.length > 0 ? datasets[0] : undefined
    );

    const onSelectionChange = (key: Key) => {
        if (key.toString() === '#new-dataset') {
            if (datasets.length === 0) {
                setDataset(undefined);
            } else {
                //return;
                //navigate(paths.project.datasets.record_new({ project_id: project.id }));
            }
        } else {
            setDataset(datasets.find((d) => d.id === key.toString()));
        }
    };

    return (
        <Flex height='100%'>
            <Tabs onSelectionChange={onSelectionChange} selectedKey={undefined} flex='1' margin={'size-200'}>
                <Flex alignItems={'end'}>
                    <TabList flex={1}>
                        {[
                            ...datasets.map((data) => <Item key={data.id}>{data.name}</Item>),
                            <Item key='#new-dataset'> <Add fill='white' height='10px' /> Import dataset </Item>,
                        ]}
                    </TabList>

                    { !isRecording &&
                        <View padding={'size-30'}>
                            <DialogTrigger>
                                <Button variant='secondary'>New Dataset</Button>
                                {(close) => HardwareSetupModal((config) => { setRecordingConfig(config); console.log('wat'); close() }, undefined)}
                            </DialogTrigger>


                            <DialogTrigger>
                                <Button variant='accent'>Start recording</Button>
                                {(close) => HardwareSetupModal((config) => { setRecordingConfig(config); close() }, dataset.id)}
                            </DialogTrigger>
                        </View>
                    }
                </Flex>
                <TabPanels UNSAFE_style={{ border: 'none' }} marginTop={'size-200'}>
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
}

export const Index = () => {
    const project = useProject();

    return (
        <RecordingProvider>
            <Datasets datasets={project.datasets} />
        </RecordingProvider>
    );
};

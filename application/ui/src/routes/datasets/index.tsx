import { useState } from 'react';

import {
    Button,
    Content,
    DialogTrigger,
    Flex,
    Heading,
    IllustratedMessage,
    Item,
    Key,
    TabList,
    TabPanels,
    Tabs,
    Text,
    View,
} from '@geti/ui';

import { SchemaDatasetOutput, SchemaProjectConfigOutput } from '../../api/openapi-spec';
import { useProject } from '../../features/projects/use-project';
import { ReactComponent as EmptyIllustration } from './../../assets/illustration.svg';
import { DatasetViewer } from './dataset-viewer';
import { ProjectSetupModal } from './record/project-setup';
import { RecordingProvider, useRecording } from './recording-provider';
import { TeleoperationSetupModal } from '../../features/configuration/teleoperation/teleoperation';

interface DatasetsProps {
    datasets: SchemaDatasetOutput[];
    projectConfig?: SchemaProjectConfigOutput;
}

const Datasets = ({ datasets, projectConfig }: DatasetsProps) => {
    const { isRecording, setRecordingConfig } = useRecording();
    const [dataset, setDataset] = useState<SchemaDatasetOutput | undefined>(
        datasets.length > 0 ? datasets[0] : undefined
    );

    const onSelectionChange = (key: Key) => {
        if (key.toString() === '#new-dataset') {
            if (datasets.length === 0) {
                setDataset(undefined);
            }
        } else {
            setDataset(datasets.find((d) => d.id === key.toString()));
        }
    };

    if (!projectConfig) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex>
                <IllustratedMessage>
                    <EmptyIllustration />
                    <Content> Please setup your project. </Content>
                    <Heading>No Project Setup.</Heading>
                    <View margin={'size-100'}>
                        <DialogTrigger>
                            <Button variant='accent'>Setup project</Button>
                            {ProjectSetupModal}
                        </DialogTrigger>
                    </View>
                </IllustratedMessage>
            </Flex>
        );
    }

    if (datasets.length === 0) {
        return (
            <Flex margin={'size-200'} direction={'column'} flex>
                <IllustratedMessage>
                    <EmptyIllustration />
                    <Content> Currently there are datasets available. </Content>
                    <Text>It&apos;s time to begin recording a dataset. </Text>
                    <Heading>No datasets yet</Heading>
                    <View margin={'size-100'}>
                        <DialogTrigger>
                            <Button variant='accent'>Start recording a new dataset</Button>
                            {(close) =>
                                TeleoperationSetupModal((config) => {
                                    setRecordingConfig(config);
                                    close();
                                }, undefined)
                            }
                        </DialogTrigger>
                    </View>
                </IllustratedMessage>
            </Flex>
        );
    }

    return (
        <Flex height='100%'>
            <Tabs onSelectionChange={onSelectionChange} flex='1' margin={'size-200'}>
                <Flex alignItems={'end'}>
                    <TabList flex={1}>
                        {datasets.map((data) => (
                            <Item key={data.id}>{data.name}</Item>
                        ))}
                    </TabList>

                    {!isRecording && (
                        <View padding={'size-30'}>
                            <DialogTrigger>
                                <Button variant='secondary'>New Dataset</Button>
                                {(close) =>
                                    TeleoperationSetupModal((config) => {
                                        setRecordingConfig(config);
                                        close();
                                    }, undefined)
                                }
                            </DialogTrigger>

                            <DialogTrigger>
                                <Button variant='accent'>Start recording</Button>
                                {(close) =>
                                    TeleoperationSetupModal((config) => {
                                        setRecordingConfig(config);
                                        close();
                                    }, dataset?.id)
                                }
                            </DialogTrigger>
                        </View>
                    )}
                </Flex>
                <TabPanels UNSAFE_style={{ border: 'none' }} marginTop={'size-200'}>
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

export const Index = () => {
    const project = useProject();

    return (
        <RecordingProvider>
            <Datasets
                datasets={project.datasets}
                projectConfig={project.config === null ? undefined : project.config}
            />
        </RecordingProvider>
    );
};

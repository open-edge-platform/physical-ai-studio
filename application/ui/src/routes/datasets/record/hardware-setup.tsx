import { Dispatch, SetStateAction, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Flex,
    Form,
    Heading,
    Item,
    Key,
    TabList,
    TabPanels,
    Tabs,
    TextField,
    View,
} from '@geti/ui';
import { useNavigate } from 'react-router-dom';

import { $api } from '../../../api/client';
import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { useProject } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { CameraSetup } from './camera-setup';
import { RobotSetup } from './robot-setup';

interface HardwareSetupProps {
    config: SchemaTeleoperationConfig;
    setConfig: Dispatch<SetStateAction<SchemaTeleoperationConfig>>;
}
export const HardwareSetup = ({ config, setConfig }: HardwareSetupProps) => {
    const project = useProject();
    const datasetName = config.dataset_id && project.datasets.find((d) => d.id === config.dataset_id)?.name;

    const { data: availableCameras } = $api.useQuery('get', '/api/hardware/cameras');
    const { data: foundRobots } = $api.useQuery('get', '/api/hardware/robots');
    const isNewDataset = datasetName === undefined;
    const [dataset, setDataset] = useState<string>(datasetName ?? '');
    const [task, setTask] = useState<string>('');

    const navigate = useNavigate();

    const updateCamera = (name: string, id: string, oldId: string) => {
        setConfig({
            ...config,
            cameras: config.cameras.map((c) => {
                if (c.name === name) {
                    return { ...c, id };
                } else if (c.id === id) {
                    return { ...c, id: oldId };
                } else {
                    return c;
                }
            }),
        });
    };

    const [activeTab, setActiveTab] = useState<string>('cameras');

    const onTabSwitch = (key: Key) => {
        setActiveTab(key.toString());
    };

    const onNext = () => {
        if (activeTab === 'cameras') {
            setActiveTab('robots');
        } else {
            //lets start!
        }
    };

    const onBack = () => {
        if (activeTab === 'robots') {
            setActiveTab('cameras');
        } else {
            navigate(paths.project.datasets.index({ project_id: project.id! }));
        }
    };

    const isValid = () => {
        const datasetNameValid = dataset !== '';
        const taskValid = task !== '';
        return datasetNameValid && taskValid;
    };

    return (
        <Flex justifyContent={'center'} flex='1'>
            <View flex='1' padding='size-150' maxWidth={'640px'}>
                <View paddingTop={'size-100'} paddingBottom={'size-100'}>
                    <Heading>Hardware setup</Heading>
                </View>
                <View backgroundColor={'gray-200'} padding={'size-200'}>
                    <Form>
                        <TextField
                            validationState={dataset === '' ? 'invalid' : 'valid'}
                            isRequired
                            label='Dataset Name'
                            value={dataset}
                            isDisabled={!isNewDataset}
                            onChange={setDataset}
                        />
                        <TextField
                            validationState={task === '' ? 'invalid' : 'valid'}
                            isRequired
                            label='Task'
                            value={task}
                            onChange={setTask}
                        />
                    </Form>
                    <View height={'330px'}>
                        <Tabs onSelectionChange={onTabSwitch} selectedKey={activeTab}>
                            <TabList>
                                <Item key='cameras'>Cameras</Item>
                                <Item key='robots'>Robots</Item>
                            </TabList>
                            <TabPanels>
                                <Item key='cameras'>
                                    <Flex gap='40px'>
                                        {config.cameras.map((camera) => (
                                            <CameraSetup
                                                key={camera.name}
                                                camera={camera}
                                                availableCameras={availableCameras ?? []}
                                                updateCamera={updateCamera}
                                            />
                                        ))}
                                    </Flex>
                                </Item>
                                <Item key='robots'>
                                    <Flex gap='40px'>
                                        {config.robots.map((robot) => (
                                            <RobotSetup
                                                key={robot.serial_id}
                                                config={robot}
                                                portInfos={foundRobots ?? []}
                                            />
                                        ))}
                                    </Flex>
                                </Item>
                            </TabPanels>
                        </Tabs>
                    </View>
                    <Flex justifyContent={'end'}>
                        <View paddingTop={'size-300'}>
                            <ButtonGroup>
                                <Button onPress={onBack} variant='secondary'>
                                    {activeTab === 'robots' ? 'Back' : 'Cancel'}
                                </Button>
                                <Button onPress={onNext} isDisabled={activeTab == 'robots' && !isValid()}>
                                    {activeTab === 'robots' ? 'Start' : 'Next'}
                                </Button>
                            </ButtonGroup>
                        </View>
                    </Flex>
                </View>
            </View>
        </Flex>
    );
};

import { Dispatch, SetStateAction, useState } from 'react';

import {
    Button,
    ComboBox,
    Section,
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
import { SchemaRobotConfig, SchemaTeleoperationConfig } from '../../../api/openapi-spec';
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

    const { data: availableCameras, refetch: refreshCameras } = $api.useQuery('get', '/api/hardware/cameras');
    const { data: foundRobots, refetch: refreshRobots  } = $api.useQuery('get', '/api/hardware/robots');
    const { data: availableCalibrations, refetch: refreshCalibrations  } = $api.useQuery('get', '/api/hardware/calibrations');
    const isNewDataset = datasetName === undefined;
    const [dataset, setDataset] = useState<string>(datasetName ?? '');

    const { data: projectTasks } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/tasks', {
        params: {
            path: { project_id: project.id! },
        },
    });

    const initialTask = isNewDataset ? '' : Object.values(projectTasks).flat()[0];
    const [task, setTask] = useState<string>(initialTask);

    const navigate = useNavigate();

    console.log(config);

    const updateCamera = (name: string, port_or_device_id: string, oldId: string) => {
        setConfig({
            ...config,
            cameras: config.cameras.map((c) => {
                if (c.name === name) {
                    return { ...c, port_or_device_id };
                } else if (c.port_or_device_id === port_or_device_id) {
                    return { ...c, port_or_device_id: oldId };
                } else {
                    return c;
                }
            }),
        });
    }

    const updateRobot = (type: "leader" | "follower", robot_config: SchemaRobotConfig) => {
        //Update robot, but importantly swap the serial ids if the id was already selected by other robot config
        const other = type == "leader" ? "follower" : "leader";

        setConfig((config) => ({
            ...config,
            [other]: {...config[other], serial_id: config[other].serial_id == robot_config.serial_id ? config[type].serial_id : config[other].serial_id},
            [type]: robot_config,
        }));
    }

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

    const onRefresh = () => {
        refreshCameras();
        refreshRobots();
        refreshCalibrations();
    }

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
                        <ComboBox
                            validationState={task === '' ? 'invalid' : 'valid'}
                            isRequired
                            label='Task'
                            allowsCustomValue
                            inputValue={task}
                            onInputChange={setTask}
                        >
                            {Object.keys(projectTasks).map((datasetName) => (
                                <Section key={datasetName} title={datasetName}>
                                    {projectTasks[datasetName].map((task) => (
                                        <Item key={task}>{task}</Item>
                                    ))}
                                </Section>
                            ))}
                        </ComboBox>
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
                                                key={`${camera.name}${camera.port_or_device_id}`}
                                                camera={camera}
                                                availableCameras={availableCameras ?? []}
                                                updateCamera={updateCamera}
                                            />
                                        ))}
                                    </Flex>
                                </Item>
                                <Item key='robots'>
                                    <Flex gap='40px'>
                                        <RobotSetup
                                            key={"leader"}
                                            config={config.leader}
                                            portInfos={foundRobots ?? []}
                                            calibrations={availableCalibrations ?? []}
                                            setConfig={(config) => updateRobot('leader', config)}
                                        />
                                        <RobotSetup
                                            key={"follower"}
                                            config={config.follower}
                                            portInfos={foundRobots ?? []}
                                            calibrations={availableCalibrations ?? []}
                                            setConfig={(config) => updateRobot('follower', config)}
                                        />
                                    </Flex>
                                </Item>
                            </TabPanels>
                        </Tabs>
                    </View>
                    <Flex justifyContent={'end'}>
                        <View paddingTop={'size-300'}>
                            <ButtonGroup>
                                <Button onPress={onRefresh} variant='secondary'>
                                    Refresh
                                </Button>
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

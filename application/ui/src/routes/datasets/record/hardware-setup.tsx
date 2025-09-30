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
import { useNavigate, useParams } from 'react-router-dom';

import { $api } from '../../../api/client';
import { SchemaRobotConfig, SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { useProject } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { CameraSetup } from './camera-setup';
import { RobotSetup } from './robot-setup';

interface HardwareSetupProps {
    onDone: (config: SchemaTeleoperationConfig) => void;
}
export const HardwareSetup = ({ onDone }: HardwareSetupProps) => {
    const { dataset_id } = useParams<{ dataset_id: string }>();
    const project = useProject();
    const { data: projectTasks } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/tasks', {
        params: {
            path: { project_id: project.id! },
        },
    });


    const isNewDataset = false; //TODO: Implement new dataset...
    const initialTask = isNewDataset ? '' : Object.values(projectTasks).flat()[0];

    const [config, setConfig] = useState<SchemaTeleoperationConfig>({
        task: initialTask,
        fps: project.config!.fps,
        dataset: project.datasets.find((d) => d.id === dataset_id) ?? null,
        cameras: project.config?.cameras ?? [],
        follower: {
            id: "",
            robot_type: project.config?.robot_type ?? "",
            serial_id: "",
            port: "",
            type: "follower",
        },
        leader: {
            id: "",
            robot_type: project.config?.robot_type ?? "",
            serial_id: "",
            port: "",
            type: "leader",
        },
    });



    const { data: availableCameras, refetch: refreshCameras } = $api.useQuery('get', '/api/hardware/cameras');
    const { data: foundRobots, refetch: refreshRobots  } = $api.useQuery('get', '/api/hardware/robots');
    const { data: availableCalibrations, refetch: refreshCalibrations  } = $api.useQuery('get', '/api/hardware/calibrations');

    const navigate = useNavigate();

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
            console.log("lets start")
            onDone(config);
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
        const datasetNameValid = config.dataset !== null;
        const taskValid = config.task !== '';
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
                            validationState={config.dataset === null ? 'invalid' : 'valid'}
                            isRequired
                            label='Dataset Name'
                            value={config.dataset?.name}
                            isDisabled={!isNewDataset}
                        />
                        <ComboBox
                            validationState={config.task === '' ? 'invalid' : 'valid'}
                            isRequired
                            label='Task'
                            allowsCustomValue
                            inputValue={config.task}
                            onInputChange={(task) => setConfig((c) => ({...c, task}))}
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

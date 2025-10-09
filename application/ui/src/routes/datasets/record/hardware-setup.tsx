import { useState } from 'react';
import { v4 as uuidv4 } from 'uuid'

import {
    Button,
    ButtonGroup,
    ComboBox,
    Flex,
    Form,
    Heading,
    Item,
    Key,
    Section,
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
import { useSettings } from '../../../components/settings/use-settings';

interface HardwareSetupProps {
    onDone: (config: SchemaTeleoperationConfig) => void;
}
export const HardwareSetup = ({ onDone }: HardwareSetupProps) => {
    const { dataset_id } = useParams<{ dataset_id: string }>();
    const [activeTab, setActiveTab] = useState<string>('cameras');
    const project = useProject();
    const { data: projectTasks } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/tasks', {
        params: {
            path: { project_id: project.id! },
        },
    });

    const {geti_action_dataset_path} = useSettings();

    const isNewDataset = !dataset_id 
    const initialTask = Object.values(projectTasks).flat()[0];

    const [config, setConfig] = useState<SchemaTeleoperationConfig>({
        task: initialTask,
        fps: project.config!.fps,
        dataset: project.datasets.find((d) => d.id === dataset_id) ?? {
            project_id: project.id!,
            name: '',
            path: '',
            id: uuidv4()
        },
        cameras: project.config?.cameras ?? [],
        follower: {
            id: '',
            robot_type: project.config?.robot_type ?? '',
            serial_id: '',
            port: '',
            type: 'follower',
        },
        leader: {
            id: '',
            robot_type: project.config?.robot_type ?? '',
            serial_id: '',
            port: '',
            type: 'leader',
        },
    });

    console.log(config.dataset);

    const { data: availableCameras, refetch: refreshCameras } = $api.useQuery('get', '/api/hardware/cameras');
    const { data: foundRobots, refetch: refreshRobots } = $api.useQuery('get', '/api/hardware/robots');
    const { data: availableCalibrations, refetch: refreshCalibrations } = $api.useQuery(
        'get',
        '/api/hardware/calibrations'
    );

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
    };

    const updateRobot = (type: 'leader' | 'follower', robot_config: SchemaRobotConfig) => {
        //Update robot, but importantly swap the serial ids if the id was already selected by other robot config
        const other = type == 'leader' ? 'follower' : 'leader';

        setConfig((c) => ({
            ...c,
            [other]: {
                ...c[other],
                serial_id: c[other].serial_id == robot_config.serial_id ? c[type].serial_id : c[other].serial_id,
            },
            [type]: robot_config,
        }));
    };

    console.log(config);
    const updateDataset = (name: string) =>{
        setConfig((c) => ({
            ...c,
            dataset: {
                ...c.dataset,
                name,
                path: `${geti_action_dataset_path}/${name}`,
            }
        }))
    }

    const onTabSwitch = (key: Key) => {
        setActiveTab(key.toString());
    };

    const onNext = () => {
        if (activeTab === 'cameras') {
            setActiveTab('robots');
        } else {
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
                            validationState={config.dataset.name == '' ? 'invalid' : 'valid'}
                            isRequired
                            label='Name'
                            width={"100%"}
                            value={config.dataset.name}
                            isDisabled={!isNewDataset}
                            onChange={updateDataset}
                        />
                        <ComboBox
                            validationState={config.task === '' ? 'invalid' : 'valid'}
                            isRequired
                            label='Task'
                            allowsCustomValue
                            inputValue={config.task}
                            onInputChange={(task) => setConfig((c) => ({ ...c, task }))}
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
                                            key={'leader'}
                                            config={config.leader}
                                            portInfos={foundRobots ?? []}
                                            calibrations={availableCalibrations ?? []}
                                            setConfig={(c) => updateRobot('leader', c)}
                                        />
                                        <RobotSetup
                                            key={'follower'}
                                            config={config.follower}
                                            portInfos={foundRobots ?? []}
                                            calibrations={availableCalibrations ?? []}
                                            setConfig={(c) => updateRobot('follower', c)}
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

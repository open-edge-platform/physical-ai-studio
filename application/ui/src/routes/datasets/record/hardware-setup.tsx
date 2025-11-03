import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    ComboBox,
    Content,
    Dialog,
    Divider,
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
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../api/client';
import {
    SchemaProjectConfigOutput,
    SchemaProjectOutput,
    SchemaRobotConfig,
    SchemaTeleoperationConfig,
} from '../../../api/openapi-spec';
import { useSettings } from '../../../components/settings/use-settings';
import { useProject } from '../../../features/projects/use-project';
import { CameraSetup } from './camera-setup';
import { RobotSetup } from './robot-setup';

interface HardwareSetupProps {
    onDone: (config: SchemaTeleoperationConfig | undefined) => void;
    dataset_id: string | undefined;
}

const makeNameSafeForPath = (name: string): string => {
    return name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
};

const TELEOPERATION_CONFIG_CACHE_KEY = 'teleoperation_config';

const storeConfigToCache = (config: SchemaTeleoperationConfig) => {
    localStorage.setItem(TELEOPERATION_CONFIG_CACHE_KEY, JSON.stringify(config));
};

const teleoperateCacheMatchesProject = (
    cache: SchemaTeleoperationConfig,
    projectConfig?: SchemaProjectConfigOutput | null
): boolean => {
    if (projectConfig === undefined || projectConfig === null) {
        return false;
    }

    const problemsInCacheCamera =
        cache.cameras.find((cachedCamera) => {
            const projectCamera = projectConfig.cameras.find((m) => m.name === cachedCamera.name);
            if (projectCamera === undefined) {
                return true;
            }
            const sameProps =
                projectCamera.name === cachedCamera.name &&
                projectCamera.width === cachedCamera.width &&
                projectCamera.height === cachedCamera.height &&
                projectCamera.fps === cachedCamera.fps &&
                projectCamera.use_depth === cachedCamera.use_depth &&
                projectCamera.driver === cachedCamera.driver;

            return !sameProps;
        }) !== undefined;

    const problemsInCacheRobot =
        cache.follower.robot_type !== projectConfig.robot_type || cache.leader.robot_type !== projectConfig.robot_type;

    return !problemsInCacheCamera && !problemsInCacheRobot;
};

const initialTeleoperationConfig = (
    initialTask: string,
    project: SchemaProjectOutput,
    dataset_id: string | undefined
): SchemaTeleoperationConfig => {
    const cachedConfig = localStorage.getItem(TELEOPERATION_CONFIG_CACHE_KEY);
    const config: SchemaTeleoperationConfig = {
        task: initialTask,
        fps: project.config?.fps ?? 30,
        dataset: project.datasets.find((d) => d.id === dataset_id) ?? {
            project_id: project.id!,
            name: '',
            path: '',
            id: uuidv4(),
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
    };
    if (cachedConfig !== null) {
        const cache = JSON.parse(cachedConfig) as SchemaTeleoperationConfig;
        if (teleoperateCacheMatchesProject(cache, project.config)) {
            const { follower, leader, cameras } = cache;
            return { ...config, follower, leader, cameras };
        }
    }
    return config;
};

export const HardwareSetup = ({ onDone, dataset_id }: HardwareSetupProps) => {
    const [activeTab, setActiveTab] = useState<string>('cameras');
    const project = useProject();
    const { data: projectTasks } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/tasks', {
        params: {
            path: { project_id: project.id! },
        },
    });

    const createDatasetMutation = $api.useMutation('post', '/api/dataset');
    const { geti_action_dataset_path } = useSettings();

    const isNewDataset = !dataset_id;
    const initialTask = Object.values(projectTasks).flat()[0];

    const [config, setConfig] = useState<SchemaTeleoperationConfig>(
        initialTeleoperationConfig(initialTask, project, dataset_id)
    );

    const { data: availableCameras, refetch: refreshCameras } = $api.useQuery('get', '/api/hardware/cameras');
    const { data: foundRobots, refetch: refreshRobots } = $api.useQuery('get', '/api/hardware/robots');
    const { data: availableCalibrations, refetch: refreshCalibrations } = $api.useQuery(
        'get',
        '/api/hardware/calibrations'
    );

    const updateCamera = (name: string, id: string, oldId: string, driver: string, oldDriver: string) => {
        setConfig({
            ...config,
            cameras: config.cameras.map((c) => {
                if (c.name === name) {
                    return { ...c, port_or_device_id: id, driver };
                } else if (c.port_or_device_id === id && c.driver === driver) {
                    return { ...c, port_or_device_id: oldId, driver: oldDriver };
                } else {
                    return c;
                }
            }),
        });
    };

    const updateRobot = (type: 'leader' | 'follower', robot_config: SchemaRobotConfig) => {
        setConfig((c) => ({
            ...c,
            [type]: robot_config,
        }));
    };

    const updateDataset = (name: string) => {
        setConfig((c) => ({
            ...c,
            dataset: {
                ...c.dataset,
                name,
                path: `${geti_action_dataset_path}/${makeNameSafeForPath(name)}`,
            },
        }));
    };

    const onTabSwitch = (key: Key) => {
        setActiveTab(key.toString());
    };

    const onNext = async () => {
        if (activeTab === 'cameras') {
            setActiveTab('robots');
        } else {
            if (isNewDataset) {
                await createDatasetMutation.mutateAsync({ body: config.dataset });
            }
            storeConfigToCache(config);
            onDone(config);
        }
    };

    const onBack = () => {
        if (activeTab === 'robots') {
            setActiveTab('cameras');
        } else {
            onDone(undefined);
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
        <View>
            <Form>
                <TextField
                    validationState={config.dataset.name == '' ? 'invalid' : 'valid'}
                    isRequired
                    label='Dataset Name'
                    width={'100%'}
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
                                <Item key={`${datasetName}-${task}`}>{task}</Item>
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
    );
};

export const HardwareSetupModal = (
    close: (config: SchemaTeleoperationConfig | undefined) => void,
    dataset_id: string | undefined
) => {
    return (
        <Dialog>
            <Heading>Teleoperate Setup</Heading>
            <Divider />
            <Content>
                <HardwareSetup dataset_id={dataset_id} onDone={close} />
            </Content>
        </Dialog>
    );
};

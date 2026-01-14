import { Suspense, useState } from 'react';

import {
    Button,
    ButtonGroup,
    ComboBox,
    Content,
    Dialog,
    Divider,
    Flex,
    Heading,
    Item,
    Key,
    Loading,
    Section,
    TabList,
    TabPanels,
    Tabs,
    TextField,
    View,
} from '@geti/ui';

import { $api } from '../../../api/client';
import { SchemaRobotConfig, SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { useSettings } from '../../../components/settings/use-settings';
import {
    initialTeleoperationConfig,
    makeNameSafeForPath,
    storeConfigToCache,
} from '../../../routes/datasets/record/utils';
import { useProject } from '../../projects/use-project';
import { CameraSetup } from '../shared/camera-setup';
import { RobotSetup } from '../shared/robot-setup';

interface TeleoperationSetupProps {
    onDone: (config: SchemaTeleoperationConfig | undefined) => void;
    dataset_id: string | undefined;
}

export const TeleoperationSetup = ({ dataset_id, onDone }: TeleoperationSetupProps) => {
    const [activeTab, setActiveTab] = useState<string>('cameras');
    const project = useProject();
    const { data: projectTasks } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/tasks', {
        params: {
            path: { project_id: project.id! },
        },
    });
    const { data: availableCameras, refetch: refreshCameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const { data: foundRobots, refetch: refreshRobots } = $api.useSuspenseQuery('get', '/api/hardware/robots');
    const { data: availableCalibrations, refetch: refreshCalibrations } = $api.useSuspenseQuery(
        'get',
        '/api/hardware/calibrations'
    );

    const createDatasetMutation = $api.useMutation('post', '/api/dataset');
    const { geti_action_dataset_path } = useSettings();

    const isNewDataset = !dataset_id;
    const initialTask = Object.values(projectTasks).flat()[0];

    const [config, setConfig] = useState<SchemaTeleoperationConfig>(
        initialTeleoperationConfig(initialTask, project, dataset_id, foundRobots)
    );

    const updateCamera = (name: string, id: string, oldId: string, driver: string, oldDriver: string) => {
        setConfig({
            ...config,
            cameras: config.cameras.map((c) => {
                if (c.name === name) {
                    return { ...c, fingerprint: id, driver };
                } else if (c.fingerprint === id && c.driver === driver) {
                    return { ...c, fingerprint: oldId, driver: oldDriver };
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
                width={'100%'}
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

export const TeleoperationSetupModal = (
    close: (config: SchemaTeleoperationConfig | undefined) => void,
    dataset_id: string | undefined
) => {
    return (
        <Dialog>
            <Heading>Teleoperate Setup</Heading>
            <Divider />
            <Content>
                <Suspense fallback={<Loading mode='overlay' />}>
                    <TeleoperationSetup dataset_id={dataset_id} onDone={close} />
                </Suspense>
            </Content>
        </Dialog>
    );
};
